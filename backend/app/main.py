from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import app_db
from .config import DATASTORE_DIR, DEFAULT_VERSION, DOCS_DIR, LMSTUDIO_BASE_URL, REPO_ROOT
from .datastore_search import SearchEngine
from .docs_store import DocsStore
from .lmstudio import LMStudioError, chat_completion
from .logging_utils import get_logger
from .prompting import PromptTemplateError, render_template
from .settings import SettingsError, ensure_defaults, get_settings_bundle, update_settings
from .web_tools import WebToolError, duckduckgo_search, extract_urls, fetch_url, parse_search_query
from .schemas import (
    AdminSettingsResponse,
    AdminSettingsUpdateRequest,
    ChatRequest,
    ChatResponse,
    Citation,
    DocCatalogDetailResponse,
    DocPageResponse,
    DocsCatalogResponse,
    ImageResult,
    MessagesResponse,
    PageImage,
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
docs_store = DocsStore(version=DEFAULT_VERSION, datastore_dir=DATASTORE_DIR)


def _safe_resolve(base: Path, unsafe_path: str) -> Path:
    candidate = (base / unsafe_path).resolve()
    if not str(candidate).startswith(str(base.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return candidate


@app.on_event("startup")
def _startup() -> None:
    app_db.init_db()
    ensure_defaults()


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        effective = get_settings_bundle()["effective"]
        retrieval = effective.get("retrieval") if isinstance(effective.get("retrieval"), dict) else {}
        retrieval_mode = str(retrieval.get("mode", "bm25"))
    except Exception:
        retrieval_mode = "unknown"
    return {
        "status": "ok",
        "docs_version": DEFAULT_VERSION,
        "datastore_ready": search_engine.is_ready(),
        "embeddings_ready": search_engine.embeddings_ready() if search_engine.is_ready() else False,
        "retrieval_mode": retrieval_mode,
        "lmstudio_base_url": LMSTUDIO_BASE_URL,
    }


@app.get("/")
def ui_root() -> FileResponse:
    index_path = REPO_ROOT / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found (missing index.html at repo root)")
    return FileResponse(str(index_path), media_type="text/html")


@app.get("/index.html")
def ui_index() -> FileResponse:
    return ui_root()


@app.get("/admin/settings", response_model=AdminSettingsResponse)
def admin_get_settings() -> dict[str, Any]:
    try:
        return get_settings_bundle()
    except SettingsError as e:
        log.exception("Settings error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/admin/settings", response_model=AdminSettingsResponse)
def admin_update_settings(req: AdminSettingsUpdateRequest) -> dict[str, Any]:
    try:
        if "system_prompt_template" in req.settings:
            tmpl = req.settings.get("system_prompt_template")
            if not isinstance(tmpl, str) or not tmpl.strip():
                raise HTTPException(status_code=400, detail="system_prompt_template must be a non-empty string")
            if "{{context}}" not in tmpl:
                raise HTTPException(status_code=400, detail="system_prompt_template must include required placeholder {{context}}")
        return update_settings(req.settings)
    except SettingsError as e:
        log.exception("Settings error")
        raise HTTPException(status_code=500, detail=str(e)) from e


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


@app.get("/docs/catalog", response_model=DocsCatalogResponse)
def docs_catalog() -> DocsCatalogResponse:
    if not docs_store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Docs catalog not found for version '{DEFAULT_VERSION}'. Run the ingestion CLI first.",
        )
    try:
        return DocsCatalogResponse(docs=docs_store.list_docs())
    except Exception as e:
        log.exception("Docs catalog error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/docs/catalog/{doc_id}", response_model=DocCatalogDetailResponse)
def docs_catalog_doc(doc_id: str) -> dict[str, Any]:
    if not docs_store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Docs catalog not found for version '{DEFAULT_VERSION}'. Run the ingestion CLI first.",
        )
    try:
        return docs_store.list_pages(doc_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail="Doc not found") from e
    except Exception as e:
        log.exception("Docs catalog doc error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/docs/page/{doc_id}/{page_id}", response_model=DocPageResponse)
def docs_page(doc_id: str, page_id: str) -> DocPageResponse:
    if not docs_store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Docs catalog not found for version '{DEFAULT_VERSION}'. Run the ingestion CLI first.",
        )
    try:
        page = docs_store.get_page(doc_id, page_id)
        md_text = docs_store.read_markdown(page)
    except KeyError as e:
        raise HTTPException(status_code=404, detail="Page not found") from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        log.exception("Docs page error")
        raise HTTPException(status_code=500, detail=str(e)) from e

    images: list[PageImage] = []
    for img in page.images:
        if not isinstance(img, dict):
            continue
        url = str(img.get("url") or "").strip()
        if not url:
            continue
        original = str(img.get("original") or "").strip()
        alt = str(img.get("alt") or "").strip() or None
        images.append(PageImage(original=original, url=url, alt=alt))

    return DocPageResponse(
        version=DEFAULT_VERSION,
        doc_id=page.doc_id,
        doc_title=page.doc_title,
        page_id=page.page_id,
        page_title=page.page_title,
        heading_path=page.heading_path,
        anchor=page.anchor,
        source_path=page.source_path,
        markdown=md_text,
        images=images,
    )


def _build_citations(results: list[dict[str, Any]], version: str, max_citations: int) -> list[Citation]:
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
        if len(citations) >= max_citations:
            break
    return citations


def _build_images(results: list[dict[str, Any]], max_images: int) -> list[ImageResult]:
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
            if len(urls) >= max_images:
                return urls
    return urls


@app.post("/docs/search", response_model=SearchResponse, response_model_exclude_none=True)
async def docs_search(req: SearchRequest) -> SearchResponse:
    if not search_engine.is_ready():
        log.error("Search index missing; run ingestion CLI to create datastore/%s/index.sqlite", DEFAULT_VERSION)
        raise HTTPException(
            status_code=503,
            detail=f"Search index not found for version '{DEFAULT_VERSION}'. Run the ingestion CLI first.",
        )

    settings = get_settings_bundle()["effective"]
    retrieval = settings.get("retrieval") if isinstance(settings.get("retrieval"), dict) else {}
    max_citations = int(retrieval.get("max_citations", 8))
    max_images = int(retrieval.get("max_images", 6))

    mode = str(retrieval.get("mode", "bm25"))
    doc_priority = retrieval.get("doc_priority") if isinstance(retrieval.get("doc_priority"), list) else []
    doc_priority_boost = float(retrieval.get("doc_priority_boost", 0.0) or 0.0)
    heading_boost = float(retrieval.get("heading_boost", 0.0) or 0.0)
    dedupe = retrieval.get("dedupe") if isinstance(retrieval.get("dedupe"), dict) else {}
    max_per_page = int(dedupe.get("max_per_page", 0) or 0)
    max_per_doc = int(dedupe.get("max_per_doc", 0) or 0)
    hybrid = retrieval.get("hybrid") if isinstance(retrieval.get("hybrid"), dict) else {}
    bm25_candidates = int(hybrid.get("bm25_candidates", 0) or 0) or None
    embedding_candidates = int(hybrid.get("embedding_candidates", 0) or 0) or None
    rrf_k = int(hybrid.get("rrf_k", 60) or 60)
    bm25_weight = float(hybrid.get("bm25_weight", 1.0) or 1.0)
    embedding_weight = float(hybrid.get("embedding_weight", 1.0) or 1.0)

    mmr = retrieval.get("mmr") if isinstance(retrieval.get("mmr"), dict) else {}
    mmr_enabled = bool(mmr.get("enabled", False))
    mmr_lambda = float(mmr.get("lambda", 0.7) or 0.7)
    mmr_candidates = int(mmr.get("candidates", 0) or 0) or None
    mmr_use_embeddings = bool(mmr.get("use_embeddings", True))

    expand = retrieval.get("expand") if isinstance(retrieval.get("expand"), dict) else {}
    expand_neighbors = int(expand.get("neighbors", 0) or 0)
    expand_max_chars = int(expand.get("max_chars", 0) or 0)

    try:
        debug: dict[str, Any] | None = {} if bool(getattr(req, "debug", False)) else None
        results = await search_engine.search(
            req.query,
            k=req.k,
            mode=mode,
            mmr_enabled=mmr_enabled,
            mmr_lambda=mmr_lambda,
            mmr_candidates=mmr_candidates,
            mmr_use_embeddings=mmr_use_embeddings,
            expand_neighbors=expand_neighbors,
            expand_max_chars=expand_max_chars,
            heading_boost=heading_boost,
            bm25_candidates=bm25_candidates,
            embedding_candidates=embedding_candidates,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            embedding_weight=embedding_weight,
            doc_priority=[str(x) for x in doc_priority],
            doc_priority_boost=doc_priority_boost,
            max_per_page=max_per_page,
            max_per_doc=max_per_doc,
            debug_out=debug,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
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
    citations = _build_citations(results, DEFAULT_VERSION, max_citations=max_citations)
    images = _build_images(results, max_images=max_images)
    return SearchResponse(chunks=chunks, citations=citations, images=images, debug=debug)


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

    settings = get_settings_bundle()["effective"]
    required_placeholders = settings.get("required_placeholders")
    if not isinstance(required_placeholders, list):
        required_placeholders = ["context"]

    template = settings.get("system_prompt_template")
    if not isinstance(template, str) or not template.strip():
        raise HTTPException(status_code=500, detail="System prompt template is missing. Set it via /admin/settings.")

    retrieval_cfg = settings.get("retrieval") if isinstance(settings.get("retrieval"), dict) else {}
    retrieval_mode = str(retrieval_cfg.get("mode", "bm25"))
    doc_priority = retrieval_cfg.get("doc_priority")
    if isinstance(doc_priority, list):
        doc_priority_str = ", ".join(str(x) for x in doc_priority)
    else:
        doc_priority_str = ""

    llm_cfg = settings.get("llm") if isinstance(settings.get("llm"), dict) else {}
    llm_model = llm_cfg.get("model")
    llm_model_id = str(llm_model).strip() if isinstance(llm_model, str) else ""
    if not llm_model_id:
        llm_model_id = None
    timeout_s = float(llm_cfg.get("timeout_s", 60.0) or 60.0)
    temperature = float(llm_cfg.get("temperature", 0.2))
    max_tokens = int(llm_cfg.get("max_tokens", 800))

    history = app_db.list_messages(session_id, limit=20)
    app_db.insert_message(session_id=session_id, role="user", content=req.message)

    doc_priority_list = [str(x) for x in doc_priority] if isinstance(doc_priority, list) else []
    doc_priority_boost = float(retrieval_cfg.get("doc_priority_boost", 0.0) or 0.0)
    heading_boost = float(retrieval_cfg.get("heading_boost", 0.0) or 0.0)
    dedupe = retrieval_cfg.get("dedupe") if isinstance(retrieval_cfg.get("dedupe"), dict) else {}
    max_per_page = int(dedupe.get("max_per_page", 0) or 0)
    max_per_doc = int(dedupe.get("max_per_doc", 0) or 0)
    hybrid = retrieval_cfg.get("hybrid") if isinstance(retrieval_cfg.get("hybrid"), dict) else {}
    bm25_candidates = int(hybrid.get("bm25_candidates", 0) or 0) or None
    embedding_candidates = int(hybrid.get("embedding_candidates", 0) or 0) or None
    rrf_k = int(hybrid.get("rrf_k", 60) or 60)
    bm25_weight = float(hybrid.get("bm25_weight", 1.0) or 1.0)
    embedding_weight = float(hybrid.get("embedding_weight", 1.0) or 1.0)

    mmr = retrieval_cfg.get("mmr") if isinstance(retrieval_cfg.get("mmr"), dict) else {}
    mmr_enabled = bool(mmr.get("enabled", False))
    mmr_lambda = float(mmr.get("lambda", 0.7) or 0.7)
    mmr_candidates = int(mmr.get("candidates", 0) or 0) or None
    mmr_use_embeddings = bool(mmr.get("use_embeddings", True))

    expand = retrieval_cfg.get("expand") if isinstance(retrieval_cfg.get("expand"), dict) else {}
    expand_neighbors = int(expand.get("neighbors", 0) or 0)
    expand_max_chars = int(expand.get("max_chars", 0) or 0)

    try:
        retrieval = await search_engine.search(
            req.message,
            k=req.k,
            mode=retrieval_mode,
            mmr_enabled=mmr_enabled,
            mmr_lambda=mmr_lambda,
            mmr_candidates=mmr_candidates,
            mmr_use_embeddings=mmr_use_embeddings,
            expand_neighbors=expand_neighbors,
            expand_max_chars=expand_max_chars,
            heading_boost=heading_boost,
            bm25_candidates=bm25_candidates,
            embedding_candidates=embedding_candidates,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            embedding_weight=embedding_weight,
            doc_priority=doc_priority_list,
            doc_priority_boost=doc_priority_boost,
            max_per_page=max_per_page,
            max_per_doc=max_per_doc,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    web_cfg = settings.get("web") if isinstance(settings.get("web"), dict) else {}
    web_enabled = bool(web_cfg.get("enabled", False))
    web_timeout_s = float(web_cfg.get("timeout_s", 15) or 15)
    web_max_bytes = int(web_cfg.get("max_bytes", 1_000_000) or 1_000_000)
    web_max_chars = int(web_cfg.get("max_chars", 20_000) or 20_000)
    web_max_urls = int(web_cfg.get("max_urls_per_message", 2) or 2)
    web_search_k = int(web_cfg.get("search_k", 5) or 5)

    web_context_blocks: list[str] = []
    if web_enabled:
        urls = extract_urls(req.message, max_urls=web_max_urls)
        search_q = parse_search_query(req.message)
        if urls:
            tasks = [
                fetch_url(u, timeout_s=web_timeout_s, max_bytes=web_max_bytes, max_chars=web_max_chars) for u in urls
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for u, res in zip(urls, results):
                if isinstance(res, Exception):
                    detail = str(res).strip() or repr(res)
                    web_context_blocks.append(f"[W{len(web_context_blocks)+1}] ERROR fetching {u}: {detail}")
                    continue
                title = f" | {res.title}" if getattr(res, "title", None) else ""
                web_context_blocks.append(f"[W{len(web_context_blocks)+1}] {res.final_url}{title}\n{res.text}")
        elif search_q:
            try:
                found = await duckduckgo_search(
                    search_q, k=web_search_k, timeout_s=web_timeout_s, max_bytes=web_max_bytes
                )
            except WebToolError as e:
                web_context_blocks.append(f"[W{len(web_context_blocks)+1}] ERROR web search: {e}")
                found = []
            if found:
                for r in found:
                    title = str(r.get("title") or "").strip()
                    url = str(r.get("url") or "").strip()
                    snippet = str(r.get("snippet") or "").strip()
                    if not url:
                        continue
                    title_part = f" | {title}" if title else ""
                    snippet_part = f"\n{snippet}" if snippet else ""
                    web_context_blocks.append(f"[W{len(web_context_blocks)+1}] {url}{title_part}{snippet_part}")
    max_citations = int(retrieval_cfg.get("max_citations", 8))
    max_images = int(retrieval_cfg.get("max_images", 6))
    citations = _build_citations(retrieval, DEFAULT_VERSION, max_citations=max_citations)
    images = _build_images(retrieval, max_images=max_images)

    context_blocks = []
    for i, r in enumerate(retrieval, start=1):
        hp = " > ".join(r.get("heading_path") or [])
        source = f"{r['doc_id']}/{r['page_id']}"
        context_blocks.append(f"[{i}] {source} | {hp}\n{r['text']}")

    if web_context_blocks:
        context_blocks.append("EXTERNAL WEB CONTEXT (not Luxriot EVO docs):")
        context_blocks.extend(web_context_blocks)

    context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no matches)"

    try:
        system_prompt = render_template(
            template,
            variables={
                "docs_version": DEFAULT_VERSION,
                "retrieval_mode": retrieval_mode,
                "retrieval_k": str(req.k),
                "doc_priority": doc_priority_str,
                "web_enabled": str(web_enabled),
                "context": context_text,
            },
            required_placeholders=[str(x) for x in required_placeholders],
        )
    except PromptTemplateError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for m in history[-10:]:
        if m["role"] in ("user", "assistant"):
            messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": req.message})

    try:
        answer = await chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            model=llm_model_id,
        )
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
