from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from . import app_db
from .auth import (
    AUTH_COOKIE,
    AuthContext,
    apply_auth_cookies,
    create_login_session,
    docs_allowed_for_role,
    ensure_bootstrap_admin,
    logout_session,
    resolve_auth,
    require_role,
)
from .config import DATASTORE_DIR, DEFAULT_VERSION, DOCS_DIR, LMSTUDIO_BASE_URL, REPO_ROOT
from .datastore_search import SearchEngine
from .docs_store import DocsStore
from .lmstudio import LMStudioError, chat_completion, chat_completion_stream
from .logging_utils import get_logger
from .prompting import PromptTemplateError, render_template
from .settings import SettingsError, ensure_defaults, get_settings_bundle, update_settings
from .web_tools import WebToolError, duckduckgo_search, extract_urls, fetch_url, parse_search_query
from .schemas import (
    AdminSettingsResponse,
    AdminSettingsUpdateRequest,
    AuthMeResponse,
    ChatRequest,
    ChatResponse,
    Citation,
    DocCatalogDetailResponse,
    DocPageResponse,
    DocsCatalogResponse,
    ImageResult,
    MessagesResponse,
    PageImage,
    ReindexJob,
    ReindexRequest,
    ReindexStatusResponse,
    LoginRequest,
    LoginResponse,
    OkResponse,
    PasswordChangeRequest,
    PasswordResetRequest,
    SearchRequest,
    SearchResponse,
    SessionCreateRequest,
    SessionsResponse,
    UserCreateRequest,
    UserInfo,
    UserUpdateRequest,
    UsersResponse,
    WebSource,
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

_search_lock = asyncio.Lock()
_reindex_lock = asyncio.Lock()
_reindex_job: dict[str, Any] | None = None
_reindex_task: asyncio.Task[None] | None = None


def _safe_resolve(base: Path, unsafe_path: str) -> Path:
    candidate = (base / unsafe_path).resolve()
    if not str(candidate).startswith(str(base.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    return candidate


@app.on_event("startup")
def _startup() -> None:
    app_db.init_db()
    ensure_defaults()
    ensure_bootstrap_admin()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.get("/auth/me", response_model=AuthMeResponse)
def auth_me(response: Response, ctx: AuthContext = Depends(resolve_auth)) -> AuthMeResponse:
    apply_auth_cookies(response, ctx)
    p = ctx.principal
    return AuthMeResponse(
        authenticated=bool(p.authenticated),
        role=p.role,
        username=p.username,
        email=p.email,
        greeting=p.greeting,
    )


@app.post("/auth/login", response_model=LoginResponse)
def auth_login(req: LoginRequest, response: Response) -> LoginResponse:
    principal, token = create_login_session(username=req.username, password=req.password)
    response.set_cookie(
        AUTH_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=int(60 * 60 * 24 * 30),
    )
    return LoginResponse(
        user=AuthMeResponse(
            authenticated=True,
            role=principal.role,
            username=principal.username,
            email=principal.email,
            greeting=principal.greeting,
        )
    )


@app.post("/auth/logout", response_model=AuthMeResponse)
def auth_logout(request: Request, response: Response, ctx: AuthContext = Depends(resolve_auth)) -> AuthMeResponse:
    if ctx.principal.authenticated:
        logout_session(request)
    response.delete_cookie(AUTH_COOKIE)
    anon_ctx = resolve_auth(request)
    apply_auth_cookies(response, anon_ctx)
    p = anon_ctx.principal
    return AuthMeResponse(authenticated=False, role=p.role, username=None, email=None, greeting=p.greeting)


@app.post("/auth/password/change", response_model=OkResponse)
def auth_change_password(req: PasswordChangeRequest, ctx: AuthContext = Depends(resolve_auth)) -> OkResponse:
    if not ctx.principal.authenticated or not ctx.principal.user_id:
        raise HTTPException(status_code=403, detail="Login required")
    user = app_db.get_user(str(ctx.principal.user_id))
    if not user:
        raise HTTPException(status_code=401, detail="Invalid session")

    from .auth import hash_password, verify_password

    if not verify_password(req.current_password, str(user.get("password_hash") or "")):
        raise HTTPException(status_code=401, detail="Invalid current password")
    app_db.update_user(user_id=str(user["user_id"]), password_hash=hash_password(req.new_password))
    return OkResponse()


@app.post("/auth/users/{user_id}/password/reset", response_model=OkResponse)
def auth_reset_password(user_id: str, req: PasswordResetRequest, ctx: AuthContext = Depends(resolve_auth)) -> OkResponse:
    require_role(ctx, {"admin"})
    user = app_db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    from .auth import hash_password

    app_db.update_user(user_id=str(user["user_id"]), password_hash=hash_password(req.new_password))
    return OkResponse()


@app.get("/auth/users", response_model=UsersResponse)
def auth_list_users(ctx: AuthContext = Depends(resolve_auth)) -> UsersResponse:
    require_role(ctx, {"admin"})
    users = [UserInfo(**u) for u in app_db.list_users(limit=500)]
    return UsersResponse(users=users)


@app.post("/auth/users", response_model=UserInfo)
def auth_create_user(req: UserCreateRequest, ctx: AuthContext = Depends(resolve_auth)) -> UserInfo:
    require_role(ctx, {"admin"})
    from .auth import hash_password

    username = str(req.username).strip()
    email = str(req.email).strip() if req.email is not None else None
    if email == "":
        email = None
    role = str(req.role)
    greeting = str(req.greeting).strip() if req.greeting is not None else None
    try:
        rec = app_db.create_user(
            username=username,
            email=email or None,
            password_hash=hash_password(req.password),
            role=role,
            greeting=greeting,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create user: {e}") from e
    return UserInfo(**rec)


@app.patch("/auth/users/{user_id}", response_model=UserInfo)
def auth_update_user(user_id: str, req: UserUpdateRequest, ctx: AuthContext = Depends(resolve_auth)) -> UserInfo:
    require_role(ctx, {"admin"})
    from .auth import hash_password

    target = app_db.get_user(user_id)
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    password_hash = hash_password(req.password) if req.password else None
    update_kwargs: dict[str, Any] = {}
    if "email" in req.model_fields_set:
        email = str(req.email).strip() if req.email is not None else None
        if email == "":
            email = None
        update_kwargs["email"] = email
    if "disabled" in req.model_fields_set:
        target_disabled = bool(str(target.get("disabled_at") or "").strip())
        desired_disabled = bool(req.disabled)
        if desired_disabled != target_disabled:
            if desired_disabled and str(ctx.principal.user_id or "") == str(user_id):
                raise HTTPException(status_code=400, detail="Cannot disable currently logged-in user")
            if (
                desired_disabled
                and str(target.get("role") or "") == "admin"
                and (not target_disabled)
                and app_db.count_active_admins() <= 1
            ):
                raise HTTPException(status_code=400, detail="Cannot disable the last active admin")
            update_kwargs["disabled_at"] = _utc_now() if desired_disabled else None

    greeting = str(req.greeting).strip() if req.greeting is not None else None
    if req.role is not None and req.role != "admin":
        target_disabled = bool(str(target.get("disabled_at") or "").strip())
        if (
            str(target.get("role") or "") == "admin"
            and (not target_disabled)
            and app_db.count_active_admins() <= 1
        ):
            raise HTTPException(status_code=400, detail="Cannot remove the last active admin")

    rec = app_db.update_user(user_id=user_id, role=req.role, greeting=greeting, password_hash=password_hash, **update_kwargs)
    if not rec:
        raise HTTPException(status_code=404, detail="User not found")
    return UserInfo(
        **{
            k: rec[k]
            for k in ("user_id", "username", "email", "role", "greeting", "disabled_at", "created_at", "updated_at")
        }
    )


def _job_defaults() -> dict[str, Any]:
    return {
        "docs_dir": str(DOCS_DIR),
        "version": DEFAULT_VERSION,
        "datastore_dir": str((DATASTORE_DIR / DEFAULT_VERSION).resolve()),
        "lmstudio_base_url": LMSTUDIO_BASE_URL,
        "embedding_max_chars": 448,
        "embedding_batch_size": 8,
    }


def _append_log(job: dict[str, Any], line: str) -> None:
    logs: list[str] = job.get("logs_tail") if isinstance(job.get("logs_tail"), list) else []
    logs.append(line)
    if len(logs) > 220:
        logs = logs[-220:]
    job["logs_tail"] = logs
    job["updated_at"] = _utc_now()


def _update_phase_from_log(job: dict[str, Any], line: str) -> None:
    s = str(line or "")
    if "Ingesting doc:" in s:
        job["phase"] = "converting_html_to_md"
    elif s.strip().startswith("Done:") or s.strip().startswith("Done."):
        # Keep 'finalizing'/'done' for the outer controller.
        pass
    elif s.startswith("Indexing "):
        job["phase"] = "indexing_bm25"
    elif s.startswith("Computing embeddings "):
        job["phase"] = "computing_embeddings"
    elif s.startswith("ERROR:"):
        job["phase"] = "failed"
        job["error"] = s
    if s.startswith("  Done:"):
        try:
            job["doc_done"] = int(job.get("doc_done") or 0) + 1
        except Exception:
            job["doc_done"] = None
        job["updated_at"] = _utc_now()


async def _pump_stream(job: dict[str, Any], stream: asyncio.StreamReader) -> None:
    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip()
        _append_log(job, text)
        _update_phase_from_log(job, text)


async def _run_reindex_job(job: dict[str, Any]) -> None:
    global _reindex_job, _reindex_task

    job["status"] = "running"
    job["phase"] = "starting"
    job["updated_at"] = _utc_now()

    docs_dir = Path(str(job.get("docs_dir") or "")).expanduser()
    version = str(job.get("version") or DEFAULT_VERSION)
    build_dir = Path(str(job.get("build_dir") or "")).expanduser()
    target_dir = DATASTORE_DIR / version

    script = REPO_ROOT / "backend" / "cli" / "ingest_evo_1_32.py"
    if not script.exists():
        job["status"] = "failed"
        job["phase"] = "failed"
        job["error"] = f"Missing ingestion CLI: {script}"
        job["updated_at"] = _utc_now()
        return

    if build_dir.exists():
        try:
            shutil.rmtree(build_dir)
        except Exception as e:
            job["status"] = "failed"
            job["phase"] = "failed"
            job["error"] = f"Failed to remove build dir {build_dir}: {e}"
            job["updated_at"] = _utc_now()
            return

    cmd = [
        sys.executable,
        str(script),
        "--docs-dir",
        str(docs_dir),
        "--out-dir",
        str(build_dir),
        "--version",
        version,
        "--lmstudio-base-url",
        LMSTUDIO_BASE_URL,
        "--embedding-max-chars",
        str(int(job.get("embedding_max_chars") or 448)),
        "--embedding-batch-size",
        str(int(job.get("embedding_batch_size") or 8)),
        "--clean",
    ]
    if not bool(job.get("compute_embeddings", True)):
        cmd.append("--no-embeddings")

    _append_log(job, f"Running: {' '.join(cmd)}")
    job["phase"] = "running_ingest"

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(REPO_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as e:
        job["status"] = "failed"
        job["phase"] = "failed"
        job["error"] = f"Failed to start ingestion process: {e}"
        job["updated_at"] = _utc_now()
        return

    try:
        job["pid"] = int(proc.pid) if proc.pid else None
    except Exception:
        job["pid"] = None

    stdout = proc.stdout or asyncio.StreamReader()
    stderr = proc.stderr or asyncio.StreamReader()

    await asyncio.gather(_pump_stream(job, stdout), _pump_stream(job, stderr))
    exit_code = await proc.wait()

    job["exit_code"] = int(exit_code)
    job["updated_at"] = _utc_now()
    if exit_code != 0:
        job["status"] = "failed"
        job["phase"] = "failed"
        if not job.get("error"):
            job["error"] = f"Ingestion failed with exit code {exit_code}"
        return

    job["phase"] = "swapping"
    job["updated_at"] = _utc_now()

    async with _search_lock:
        search_engine.close()
        try:
            if not build_dir.exists():
                raise RuntimeError(f"Build output missing: {build_dir}")

            backup_dir = None
            if target_dir.exists():
                suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                backup_dir = target_dir.with_name(f"{target_dir.name}.bak_{suffix}")
                target_dir.rename(backup_dir)

            build_dir.rename(target_dir)
            docs_store.invalidate()
            if backup_dir and backup_dir.exists():
                pass
        except Exception as e:
            job["status"] = "failed"
            job["phase"] = "failed"
            job["error"] = f"Swap failed: {e}"
            job["updated_at"] = _utc_now()
            return

    job["status"] = "succeeded"
    job["phase"] = "done"
    job["updated_at"] = _utc_now()

    async with _reindex_lock:
        _reindex_task = None
        _reindex_job = job


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


@app.get("/admin/reindex", response_model=ReindexStatusResponse, response_model_exclude_none=True)
async def admin_reindex_status(ctx: AuthContext = Depends(resolve_auth)) -> ReindexStatusResponse:
    require_role(ctx, {"admin"})
    async with _reindex_lock:
        job = _reindex_job
    return ReindexStatusResponse(defaults=_job_defaults(), job=ReindexJob(**job) if job else None)


@app.post("/admin/reindex", response_model=ReindexStatusResponse, response_model_exclude_none=True)
async def admin_reindex_start(req: ReindexRequest, ctx: AuthContext = Depends(resolve_auth)) -> ReindexStatusResponse:
    require_role(ctx, {"admin"})
    global _reindex_job, _reindex_task

    async with _reindex_lock:
        if _reindex_job and _reindex_job.get("status") == "running":
            raise HTTPException(status_code=409, detail="Reindex is already running")

        docs_dir = Path(str(req.docs_dir or DOCS_DIR)).expanduser()
        if not docs_dir.exists() or not docs_dir.is_dir():
            raise HTTPException(status_code=400, detail=f"docs_dir not found or not a directory: {docs_dir}")

        doc_total = len([p for p in docs_dir.iterdir() if p.is_dir()])
        job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(4)
        build_dir = (DATASTORE_DIR / f".build_{DEFAULT_VERSION}_{job_id}").resolve()

        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "running",
            "phase": "queued",
            "docs_dir": str(docs_dir),
            "version": DEFAULT_VERSION,
            "compute_embeddings": bool(req.compute_embeddings),
            "embedding_max_chars": int(req.embedding_max_chars),
            "embedding_batch_size": int(req.embedding_batch_size),
            "started_at": _utc_now(),
            "updated_at": _utc_now(),
            "doc_total": int(doc_total),
            "doc_done": 0,
            "exit_code": None,
            "error": None,
            "logs_tail": [],
            "build_dir": str(build_dir),
        }

        _reindex_job = job
        _reindex_task = asyncio.create_task(_run_reindex_job(job))

    return ReindexStatusResponse(defaults=_job_defaults(), job=ReindexJob(**job))


@app.get("/")
def ui_root(request: Request) -> FileResponse:
    index_path = REPO_ROOT / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found (missing index.html at repo root)")
    ctx = resolve_auth(request)
    resp = FileResponse(str(index_path), media_type="text/html")
    apply_auth_cookies(resp, ctx)
    return resp


@app.get("/index.html")
def ui_index(request: Request) -> FileResponse:
    return ui_root(request)


@app.get("/admin/settings", response_model=AdminSettingsResponse)
def admin_get_settings(ctx: AuthContext = Depends(resolve_auth)) -> dict[str, Any]:
    require_role(ctx, {"admin"})
    try:
        return get_settings_bundle()
    except SettingsError as e:
        log.exception("Settings error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/admin/settings", response_model=AdminSettingsResponse)
def admin_update_settings(req: AdminSettingsUpdateRequest, ctx: AuthContext = Depends(resolve_auth)) -> dict[str, Any]:
    require_role(ctx, {"admin"})
    try:
        if "system_prompt_template" in req.settings:
            tmpl = req.settings.get("system_prompt_template")
            if not isinstance(tmpl, str) or not tmpl.strip():
                raise HTTPException(status_code=400, detail="system_prompt_template must be a non-empty string")
            if "{{context}}" not in tmpl:
                raise HTTPException(status_code=400, detail="system_prompt_template must include required placeholder {{context}}")
        if "system_prompt_templates" in req.settings:
            tmpls = req.settings.get("system_prompt_templates")
            if not isinstance(tmpls, dict):
                raise HTTPException(status_code=400, detail="system_prompt_templates must be an object/dict")
            for role, tmpl in tmpls.items():
                if not isinstance(role, str) or not role.strip():
                    raise HTTPException(status_code=400, detail="system_prompt_templates keys must be non-empty strings")
                if not isinstance(tmpl, str):
                    raise HTTPException(
                        status_code=400, detail=f"system_prompt_templates['{role}'] must be a string (or empty to inherit)"
                    )
                if tmpl.strip() and "{{context}}" not in tmpl:
                    raise HTTPException(
                        status_code=400,
                        detail=f"system_prompt_templates['{role}'] must include required placeholder {{context}}",
                    )
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


def _doc_allowed(doc_id: str, allow: set[str] | None, deny: set[str]) -> bool:
    did = str(doc_id or "").strip()
    if not did:
        return False
    if allow is not None and did not in allow:
        return False
    if did in deny:
        return False
    return True


def _select_system_prompt_template(settings: dict[str, Any], *, role: str) -> str:
    role = str(role or "").strip() or "anonymous"
    tmpls = settings.get("system_prompt_templates")
    if isinstance(tmpls, dict):
        t = tmpls.get(role)
        if isinstance(t, str) and t.strip():
            return t
    t = settings.get("system_prompt_template")
    if isinstance(t, str) and t.strip():
        return t
    raise HTTPException(status_code=500, detail="System prompt template is missing. Set it via /admin/settings.")


@app.get("/docs/catalog", response_model=DocsCatalogResponse)
def docs_catalog(ctx: AuthContext = Depends(resolve_auth)) -> DocsCatalogResponse:
    if not docs_store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Docs catalog not found for version '{DEFAULT_VERSION}'. Run the ingestion CLI first.",
        )
    try:
        allow, deny = docs_allowed_for_role(ctx.principal.role)
        docs = [d for d in docs_store.list_docs() if _doc_allowed(d.get("doc_id"), allow, deny)]
        return DocsCatalogResponse(docs=docs)
    except Exception as e:
        log.exception("Docs catalog error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/docs/catalog/{doc_id}", response_model=DocCatalogDetailResponse)
def docs_catalog_doc(doc_id: str, ctx: AuthContext = Depends(resolve_auth)) -> dict[str, Any]:
    if not docs_store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Docs catalog not found for version '{DEFAULT_VERSION}'. Run the ingestion CLI first.",
        )
    allow, deny = docs_allowed_for_role(ctx.principal.role)
    if not _doc_allowed(doc_id, allow, deny):
        raise HTTPException(status_code=403, detail="Doc is not available for this user")
    try:
        return docs_store.list_pages(doc_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail="Doc not found") from e
    except Exception as e:
        log.exception("Docs catalog doc error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/docs/page/{doc_id}/{page_id}", response_model=DocPageResponse)
def docs_page(doc_id: str, page_id: str, ctx: AuthContext = Depends(resolve_auth)) -> DocPageResponse:
    if not docs_store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Docs catalog not found for version '{DEFAULT_VERSION}'. Run the ingestion CLI first.",
        )
    allow, deny = docs_allowed_for_role(ctx.principal.role)
    if not _doc_allowed(doc_id, allow, deny):
        raise HTTPException(status_code=403, detail="Doc is not available for this user")
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
async def docs_search(req: SearchRequest, ctx: AuthContext = Depends(resolve_auth)) -> SearchResponse:
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

    allow, deny = docs_allowed_for_role(ctx.principal.role)
    search_k = int(req.k)
    buffer_k = min(max(search_k * 5, search_k + 10), 200)

    try:
        debug: dict[str, Any] | None = {} if bool(getattr(req, "debug", False)) else None
        async with _search_lock:
            results = await search_engine.search(
                req.query,
                k=buffer_k,
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
    results = [r for r in results if _doc_allowed(r.get("doc_id"), allow, deny)][:search_k]
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


def _assert_session_access(ctx: AuthContext, sess: dict[str, Any]) -> None:
    owner = str(sess.get("owner_id") or "legacy")
    if owner == "legacy":
        if ctx.principal.role != "admin":
            raise HTTPException(status_code=404, detail="Session not found")
        return
    if owner != ctx.principal.owner_id:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, response: Response, ctx: AuthContext = Depends(resolve_auth)) -> ChatResponse:
    apply_auth_cookies(response, ctx)
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
        _assert_session_access(ctx, session)
        session_id = req.session_id
    else:
        session = app_db.create_session(owner_id=ctx.principal.owner_id, title=req.message[:60])
        session_id = session["session_id"]

    settings = get_settings_bundle()["effective"]
    required_placeholders = settings.get("required_placeholders")
    if not isinstance(required_placeholders, list):
        required_placeholders = ["context"]

    template = _select_system_prompt_template(settings, role=ctx.principal.role)

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

    allow, deny = docs_allowed_for_role(ctx.principal.role)
    search_k = int(req.k)
    buffer_k = min(max(search_k * 5, search_k + 10), 200)

    try:
        async with _search_lock:
            retrieval = await search_engine.search(
                req.message,
                k=buffer_k,
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
    retrieval = [r for r in retrieval if _doc_allowed(r.get("doc_id"), allow, deny)][:search_k]

    web_cfg = settings.get("web") if isinstance(settings.get("web"), dict) else {}
    web_enabled = bool(web_cfg.get("enabled", False))
    if ctx.principal.role in ("anonymous", "client"):
        web_enabled = False
    web_timeout_s = float(web_cfg.get("timeout_s", 15) or 15)
    web_max_bytes = int(web_cfg.get("max_bytes", 1_000_000) or 1_000_000)
    web_max_chars = int(web_cfg.get("max_chars", 20_000) or 20_000)
    web_max_urls = int(web_cfg.get("max_urls_per_message", 2) or 2)
    web_search_k = int(web_cfg.get("search_k", 5) or 5)

    web_context_blocks: list[str] = []
    web_sources: list[WebSource] = []
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
                    web_sources.append(WebSource(kind="fetch", url=u, error=detail))
                    continue
                title = f" | {res.title}" if getattr(res, "title", None) else ""
                web_context_blocks.append(f"[W{len(web_context_blocks)+1}] {res.final_url}{title}\n{res.text}")
                web_sources.append(
                    WebSource(
                        kind="fetch",
                        url=u,
                        final_url=str(res.final_url),
                        title=getattr(res, "title", None),
                        status=int(getattr(res, "status", 0) or 0) or None,
                        truncated=bool(getattr(res, "truncated", False)),
                    )
                )
        elif search_q:
            ddg_query_url = "https://html.duckduckgo.com/html/?q=" + quote_plus(search_q)
            try:
                found = await duckduckgo_search(
                    search_q, k=web_search_k, timeout_s=web_timeout_s, max_bytes=web_max_bytes
                )
            except WebToolError as e:
                err = str(e)
                web_context_blocks.append(f"[W{len(web_context_blocks)+1}] ERROR web search: {err}")
                web_sources.append(WebSource(kind="search", url=ddg_query_url, error=err))
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
                    web_sources.append(WebSource(kind="search", url=url, title=title or None, snippet=snippet or None))
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
                "user_role": str(ctx.principal.role),
                "username": str(ctx.principal.username or ""),
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
    return ChatResponse(answer=answer, citations=citations, images=images, web_sources=web_sources, session_id=session_id)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, ctx: AuthContext = Depends(resolve_auth)) -> StreamingResponse:
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
        _assert_session_access(ctx, session)
        session_id = req.session_id
    else:
        session = app_db.create_session(owner_id=ctx.principal.owner_id, title=req.message[:60])
        session_id = session["session_id"]

    settings = get_settings_bundle()["effective"]
    required_placeholders = settings.get("required_placeholders")
    if not isinstance(required_placeholders, list):
        required_placeholders = ["context"]

    template = _select_system_prompt_template(settings, role=ctx.principal.role)

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

    allow, deny = docs_allowed_for_role(ctx.principal.role)
    search_k = int(req.k)
    buffer_k = min(max(search_k * 5, search_k + 10), 200)

    def sse(event: str, data: Any) -> bytes:
        if isinstance(data, str):
            payload = data
        else:
            payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")

    async def gen() -> Any:
        try:
            yield sse("status", {"phase": "starting", "session_id": session_id})

            yield sse(
                "status",
                {
                    "phase": "retrieving_docs",
                    "message": f"Searching documentation ({retrieval_mode})…",
                    "k": int(req.k),
                },
            )
            try:
                async with _search_lock:
                    retrieval = await search_engine.search(
                        req.message,
                        k=buffer_k,
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
                yield sse("error", {"error": "Bad request", "detail": str(e)})
                return
            except RuntimeError as e:
                yield sse("error", {"error": "Search unavailable", "detail": str(e)})
                return
            retrieval = [r for r in retrieval if _doc_allowed(r.get("doc_id"), allow, deny)][:search_k]

            yield sse(
                "status",
                {
                    "phase": "retrieved_docs",
                    "message": f"Found {len(retrieval)} relevant sections.",
                    "found": int(len(retrieval)),
                },
            )

            web_cfg = settings.get("web") if isinstance(settings.get("web"), dict) else {}
            web_enabled = bool(web_cfg.get("enabled", False))
            if ctx.principal.role in ("anonymous", "client"):
                web_enabled = False
            web_timeout_s = float(web_cfg.get("timeout_s", 15) or 15)
            web_max_bytes = int(web_cfg.get("max_bytes", 1_000_000) or 1_000_000)
            web_max_chars = int(web_cfg.get("max_chars", 20_000) or 20_000)
            web_max_urls = int(web_cfg.get("max_urls_per_message", 2) or 2)
            web_search_k = int(web_cfg.get("search_k", 5) or 5)

            web_context_blocks: list[str] = []
            web_sources: list[WebSource] = []
            if web_enabled:
                urls = extract_urls(req.message, max_urls=web_max_urls)
                search_q = parse_search_query(req.message)
                if urls:
                    yield sse("status", {"phase": "fetching_web", "message": f"Fetching {len(urls)} URL(s)…"})
                    tasks = [
                        fetch_url(u, timeout_s=web_timeout_s, max_bytes=web_max_bytes, max_chars=web_max_chars)
                        for u in urls
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for u, res in zip(urls, results):
                        if isinstance(res, Exception):
                            detail = str(res).strip() or repr(res)
                            web_context_blocks.append(f"[W{len(web_context_blocks)+1}] ERROR fetching {u}: {detail}")
                            web_sources.append(WebSource(kind="fetch", url=u, error=detail))
                            continue
                        title = f" | {res.title}" if getattr(res, "title", None) else ""
                        web_context_blocks.append(f"[W{len(web_context_blocks)+1}] {res.final_url}{title}\n{res.text}")
                        web_sources.append(
                            WebSource(
                                kind="fetch",
                                url=u,
                                final_url=str(res.final_url),
                                title=getattr(res, "title", None),
                                status=int(getattr(res, "status", 0) or 0) or None,
                                truncated=bool(getattr(res, "truncated", False)),
                            )
                        )
                elif search_q:
                    yield sse("status", {"phase": "searching_web", "message": f"Searching the web ({web_search_k})…"})
                    ddg_query_url = "https://html.duckduckgo.com/html/?q=" + quote_plus(search_q)
                    try:
                        found = await duckduckgo_search(
                            search_q, k=web_search_k, timeout_s=web_timeout_s, max_bytes=web_max_bytes
                        )
                    except WebToolError as e:
                        err = str(e)
                        web_context_blocks.append(f"[W{len(web_context_blocks)+1}] ERROR web search: {err}")
                        web_sources.append(WebSource(kind="search", url=ddg_query_url, error=err))
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
                            web_sources.append(
                                WebSource(kind="search", url=url, title=title or None, snippet=snippet or None)
                            )

            max_citations = int(retrieval_cfg.get("max_citations", 8))
            max_images = int(retrieval_cfg.get("max_images", 6))
            citations = _build_citations(retrieval, DEFAULT_VERSION, max_citations=max_citations)
            images = _build_images(retrieval, max_images=max_images)

            retrieval_pack = SearchResponse(
                chunks=[
                    {
                        "doc_id": r["doc_id"],
                        "page_id": r["page_id"],
                        "heading_path": r["heading_path"],
                        "text": r["text"],
                        "score": r["score"],
                    }
                    for r in retrieval
                ],
                citations=citations,
                images=images,
            ).model_dump()
            retrieval_pack["meta"] = {
                "docs_version": DEFAULT_VERSION,
                "retrieval_mode": retrieval_mode,
                "k": int(req.k),
                "web_enabled": bool(web_enabled),
            }
            if web_sources:
                retrieval_pack["web_sources"] = [ws.model_dump() for ws in web_sources]
            yield sse("retrieval", retrieval_pack)

            context_blocks: list[str] = []
            for i, r in enumerate(retrieval, start=1):
                hp = " > ".join(r.get("heading_path") or [])
                source = f"{r['doc_id']}/{r['page_id']}"
                context_blocks.append(f"[{i}] {source} | {hp}\n{r['text']}")

            if web_context_blocks:
                context_blocks.append("EXTERNAL WEB CONTEXT (not Luxriot EVO docs):")
                context_blocks.extend(web_context_blocks)

            context_text = "\n\n---\n\n".join(context_blocks) if context_blocks else "(no matches)"

            yield sse("status", {"phase": "building_prompt", "message": "Building prompt…"})
            try:
                system_prompt = render_template(
                    template,
                    variables={
                        "user_role": str(ctx.principal.role),
                        "username": str(ctx.principal.username or ""),
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
                yield sse("error", {"error": "Prompt template error", "detail": str(e)})
                return

            messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
            for m in history[-10:]:
                if m["role"] in ("user", "assistant"):
                    messages.append({"role": m["role"], "content": m["content"]})
            messages.append({"role": "user", "content": req.message})

            model_label = llm_model_id or "LM Studio (auto)"
            yield sse("status", {"phase": "calling_llm", "message": f"Generating answer ({model_label})…"})

            parts: list[str] = []
            llm_started = time.monotonic()
            q: asyncio.Queue[str] = asyncio.Queue()
            done = asyncio.Event()
            stream_err: Exception | None = None

            async def reader() -> None:
                nonlocal stream_err
                try:
                    async for delta in chat_completion_stream(
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout_s=timeout_s,
                        model=llm_model_id,
                    ):
                        await q.put(delta)
                except Exception as e:
                    stream_err = e
                finally:
                    done.set()

            task = asyncio.create_task(reader())
            try:
                while True:
                    try:
                        delta = await asyncio.wait_for(q.get(), timeout=3.0)
                        parts.append(delta)
                        yield sse("delta", {"delta": delta})
                    except asyncio.TimeoutError:
                        if done.is_set():
                            break
                        elapsed_s = int(time.monotonic() - llm_started)
                        yield sse("ping", {"phase": "calling_llm", "elapsed_s": elapsed_s})

                # Drain any remaining queued deltas after completion.
                while not q.empty():
                    try:
                        delta = q.get_nowait()
                    except Exception:
                        break
                    parts.append(delta)
                    yield sse("delta", {"delta": delta})

                if stream_err:
                    raise stream_err
            except asyncio.CancelledError:
                raise
            except LMStudioError as e:
                log.exception("LM Studio error (stream)")
                yield sse("error", {"error": "LM Studio error", "detail": str(e)})
                return
            except Exception as e:
                log.exception("LM Studio stream error")
                yield sse("error", {"error": "LM Studio stream error", "detail": str(e)})
                return
            finally:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(Exception):
                        await task

            answer = "".join(parts)
            yield sse("status", {"phase": "saving", "message": "Saving answer to session…"})
            app_db.insert_message(session_id=session_id, role="assistant", content=answer)

            yield sse(
                "final",
                ChatResponse(
                    answer=answer,
                    citations=citations,
                    images=images,
                    web_sources=web_sources,
                    session_id=session_id,
                ).model_dump(),
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception("Chat stream error")
            yield sse("error", {"error": "Chat stream failed", "detail": str(e)})

    resp = StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
    apply_auth_cookies(resp, ctx)
    return resp


@app.get("/sessions", response_model=SessionsResponse)
def sessions_list(ctx: AuthContext = Depends(resolve_auth)) -> SessionsResponse:
    if ctx.principal.role == "anonymous":
        raise HTTPException(status_code=403, detail="Login required")
    sessions = app_db.list_sessions(owner_id=ctx.principal.owner_id, limit=100)
    if ctx.principal.role == "admin":
        sessions.extend(app_db.list_sessions(owner_id="legacy", limit=100))
        uniq: dict[str, dict[str, Any]] = {}
        for s in sessions:
            sid = str(s.get("session_id") or "")
            if sid:
                uniq[sid] = s
        sessions = list(uniq.values())
        sessions.sort(
            key=lambda x: str(x.get("last_message_at") or x.get("created_at") or ""),
            reverse=True,
        )
        sessions = sessions[:100]
    return SessionsResponse(sessions=sessions)


@app.post("/sessions", response_model=dict)
def sessions_create(req: SessionCreateRequest, ctx: AuthContext = Depends(resolve_auth)) -> dict[str, Any]:
    if ctx.principal.role == "anonymous":
        raise HTTPException(status_code=403, detail="Login required")
    return app_db.create_session(owner_id=ctx.principal.owner_id, title=req.title)


@app.get("/sessions/{session_id}/messages", response_model=MessagesResponse)
def sessions_messages(session_id: str, ctx: AuthContext = Depends(resolve_auth)) -> MessagesResponse:
    if ctx.principal.role == "anonymous":
        raise HTTPException(status_code=403, detail="Login required")
    sess = app_db.get_session(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_access(ctx, sess)
    return MessagesResponse(messages=app_db.list_messages(session_id, limit=500))
