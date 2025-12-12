from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import app_db
from .config import DATASTORE_DIR, DEFAULT_VERSION, DOCS_DIR, LMSTUDIO_BASE_URL
from .datastore_search import SearchEngine
from .lmstudio import LMStudioError, chat_completion
from .logging_utils import get_logger
from .schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    ImageResult,
    MessagesResponse,
    SearchRequest,
    SearchResponse,
    SessionCreateRequest,
    SessionsResponse,
)

log = get_logger(__name__)

app = FastAPI(title="luxriot-mcp-backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_engine = SearchEngine(version=DEFAULT_VERSION, datastore_dir=DATASTORE_DIR)


def _safe_resolve(base: Path, unsafe_path: str) -> Path:
    candidate = (base / unsafe_path).resolve()
    if not str(candidate).startswith(str(base.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return candidate


@app.on_event("startup")
def _startup() -> None:
    app_db.init_db()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "docs_version": DEFAULT_VERSION,
        "datastore_ready": search_engine.is_ready(),
        "lmstudio_base_url": LMSTUDIO_BASE_URL,
    }


@app.get("/assets/{version}/{path:path}")
def assets(version: str, path: str) -> FileResponse:
    base = DATASTORE_DIR / version / "assets"
    if not base.exists():
        raise HTTPException(status_code=404, detail=f"Assets not found for version '{version}'")
    file_path = _safe_resolve(base, path)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(str(file_path))


@app.get("/rawdocs/{version}/{path:path}")
def rawdocs(version: str, path: str) -> FileResponse:
    if version != DEFAULT_VERSION:
        raise HTTPException(status_code=404, detail=f"Unknown version '{version}'")
    base = DOCS_DIR
    if not base.exists():
        raise HTTPException(status_code=404, detail="Docs directory not found")
    file_path = _safe_resolve(base, path)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Doc file not found")
    return FileResponse(str(file_path))


def _build_citations(results: list[dict[str, Any]], version: str) -> list[Citation]:
    seen: set[tuple[str, str]] = set()
    citations: list[Citation] = []
    for r in results:
        key = (r["doc_id"], r["page_id"])
        if key in seen:
            continue
        seen.add(key)
        title = (r.get("heading_path") or [r["page_id"]])[-1]
        source_rel = r.get("source_path") or ""
        citations.append(
            Citation(
                title=title,
                doc_id=r["doc_id"],
                page_id=r["page_id"],
                anchor=r.get("anchor"),
                source_path=f"/rawdocs/{version}/{source_rel}",
            )
        )
        if len(citations) >= 8:
            break
    return citations


def _build_images(results: list[dict[str, Any]]) -> list[ImageResult]:
    urls: list[ImageResult] = []
    seen: set[str] = set()
    for r in results:
        near = (r.get("heading_path") or [None])[-1]
        for url in r.get("images") or []:
            if url in seen:
                continue
            seen.add(url)
            urls.append(
                ImageResult(
                    url=url,
                    doc_id=r["doc_id"],
                    page_id=r["page_id"],
                    near_heading=near,
                )
            )
            if len(urls) >= 6:
                return urls
    return urls


@app.post("/docs/search", response_model=SearchResponse)
def docs_search(req: SearchRequest) -> SearchResponse:
    if not search_engine.is_ready():
        log.error("Search index missing; run ingestion CLI to create datastore/%s/index.sqlite", DEFAULT_VERSION)
        raise HTTPException(
            status_code=503,
            detail=f"Search index not found for version '{DEFAULT_VERSION}'. Run the ingestion CLI first.",
        )

    results = search_engine.search(req.query, k=req.k)
    chunks = [
        {
            "doc_id": r["doc_id"],
            "page_id": r["page_id"],
            "heading_path": r["heading_path"],
            "text": r["text"],
            "score": r["score"],
        }
        for r in results
    ]
    citations = _build_citations(results, DEFAULT_VERSION)
    images = _build_images(results)
    return SearchResponse(chunks=chunks, citations=citations, images=images)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if not search_engine.is_ready():
        log.error("Chat requested but index missing; run ingestion CLI.")
        raise HTTPException(
            status_code=503,
            detail=f"Search index not found for version '{DEFAULT_VERSION}'. Run the ingestion CLI first.",
        )

    session = None
    if req.session_id:
        session = app_db.get_session(req.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        session_id = req.session_id
    else:
        session = app_db.create_session(title=req.message[:60])
        session_id = session["session_id"]

    history = app_db.list_messages(session_id, limit=20)
    app_db.insert_message(session_id=session_id, role="user", content=req.message)

    retrieval = search_engine.search(req.message, k=req.k)
    citations = _build_citations(retrieval, DEFAULT_VERSION)
    images = _build_images(retrieval)

    context_blocks = []
    for i, r in enumerate(retrieval, start=1):
        hp = " > ".join(r.get("heading_path") or [])
        source = f"{r['doc_id']}/{r['page_id']}"
        context_blocks.append(f"[{i}] {source} | {hp}\n{r['text']}")

    system_prompt = (
        "You are a Luxriot EVO 1.32 assistant. Answer ONLY using the provided documentation context.\n"
        "If the context does not contain the answer, say you cannot find it in the docs and ask for clarification.\n"
        "Be concise and practical. When you rely on a context item, cite it using bracketed numbers like [1] or [1][3].\n"
    )
    context_prompt = "DOCUMENTATION CONTEXT:\n\n" + ("\n\n---\n\n".join(context_blocks) if context_blocks else "(no matches)")

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt + "\n" + context_prompt}]
    for m in history[-10:]:
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": req.message})

    try:
        answer = await chat_completion(messages=messages)
    except LMStudioError as e:
        log.exception("LM Studio error")
        raise HTTPException(status_code=502, detail=str(e)) from e

    app_db.insert_message(session_id=session_id, role="assistant", content=answer)
    return ChatResponse(answer=answer, citations=citations, images=images, session_id=session_id)


@app.get("/sessions", response_model=SessionsResponse)
def sessions_list() -> SessionsResponse:
    sessions = app_db.list_sessions(limit=100)
    return SessionsResponse(sessions=sessions)


@app.post("/sessions", response_model=dict)
def sessions_create(req: SessionCreateRequest) -> dict[str, Any]:
    return app_db.create_session(title=req.title)


@app.get("/sessions/{session_id}/messages", response_model=MessagesResponse)
def sessions_messages(session_id: str) -> MessagesResponse:
    sess = app_db.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return MessagesResponse(messages=app_db.list_messages(session_id, limit=500))
