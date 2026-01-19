from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import datetime
import re
import secrets
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
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
from .config import APP_DB_PATH, APP_VERSION, DATASTORE_DIR, DEFAULT_VERSION, DOCS_DIR, LMSTUDIO_BASE_URL, REPO_ROOT
from .datastore_search import SearchEngine
from .docs_store import DocsStore
from .lmstudio import LMStudioError, chat_completion, chat_completion_stream
from .logging_utils import get_logger
from .pdf_export import render_markdown_to_pdf
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
    ContextChunk,
    DocCatalogDetailResponse,
    DocCreateResponse,
    DocEditInfo,
    DocEditDeleteRequest,
    DocEditRequest,
    DocEditResponse,
    DocPublishRequest,
    PublishRequestDecision,
    PublishRequestInfo,
    PublishRequestItem,
    PublishRequestListResponse,
    DocDeleteRequest,
    DocExcludeRequest,
    DocExcludeResponse,
    DocGuideCreateRequest,
    DocAssetUploadResponse,
    DocPageResponse,
    DocPdfRequest,
    DocsStyleResponse,
    DocsStyleUpdateRequest,
    DocsCatalogResponse,
    DocsVersionsResponse,
    DocSectionCreateRequest,
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

_search_engines: dict[str, SearchEngine] = {}
_docs_stores: dict[str, DocsStore] = {}

_search_lock = asyncio.Lock()
_reindex_lock = asyncio.Lock()
_reindex_job: dict[str, Any] | None = None
_reindex_task: asyncio.Task[None] | None = None
_REINDEX_BACKUP_KEEP = 8


def _available_versions() -> list[str]:
    versions: list[str] = []
    if DATASTORE_DIR.exists():
        for entry in DATASTORE_DIR.iterdir():
            if not entry.is_dir():
                continue
            name = entry.name
            if name.startswith("."):
                continue
            if (entry / "pages.jsonl").exists():
                versions.append(name)
    versions = sorted(set(versions))
    if DEFAULT_VERSION in versions:
        versions = [DEFAULT_VERSION] + [v for v in versions if v != DEFAULT_VERSION]
    return versions


def _require_version(version: str | None) -> str:
    ver = str(version or DEFAULT_VERSION).strip() or DEFAULT_VERSION
    versions = _available_versions()
    if ver not in versions:
        raise HTTPException(status_code=404, detail=f"Unknown docs version '{ver}'")
    return ver


def _get_docs_store(version: str) -> DocsStore:
    store = _docs_stores.get(version)
    if not store:
        store = DocsStore(version=version, datastore_dir=DATASTORE_DIR)
        _docs_stores[version] = store
    return store


def _get_search_engine(version: str) -> SearchEngine:
    engine = _search_engines.get(version)
    if not engine:
        engine = SearchEngine(version=version, datastore_dir=DATASTORE_DIR)
        _search_engines[version] = engine
    return engine


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
        user_id=p.user_id,
        username=p.username,
        email=p.email,
        greeting=p.greeting,
        docs_edit=bool(getattr(p, "docs_edit", False)),
        docs_publish=bool(getattr(p, "docs_publish", False)),
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
            user_id=principal.user_id,
            username=principal.username,
            email=principal.email,
            greeting=principal.greeting,
            docs_edit=bool(getattr(principal, "docs_edit", False)),
            docs_publish=bool(getattr(principal, "docs_publish", False)),
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
    return AuthMeResponse(
        authenticated=False,
        role=p.role,
        user_id=None,
        username=None,
        email=None,
        greeting=p.greeting,
        docs_edit=False,
        docs_publish=False,
    )


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
            docs_edit=req.docs_edit,
            docs_publish=req.docs_publish,
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
    if "docs_edit" in req.model_fields_set:
        update_kwargs["docs_edit"] = bool(req.docs_edit)
    if "docs_publish" in req.model_fields_set:
        update_kwargs["docs_publish"] = bool(req.docs_publish)

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
            for k in (
                "user_id",
                "username",
                "email",
                "role",
                "docs_edit",
                "docs_publish",
                "greeting",
                "disabled_at",
                "created_at",
                "updated_at",
            )
        }
    )


def _job_defaults() -> dict[str, Any]:
    summary = {}
    try:
        effective = get_settings_bundle()["effective"]
        retrieval = effective.get("retrieval") if isinstance(effective.get("retrieval"), dict) else {}
        summary = retrieval.get("summary") if isinstance(retrieval.get("summary"), dict) else {}
    except Exception:
        summary = {}
    return {
        "docs_dir": str(DOCS_DIR),
        "version": DEFAULT_VERSION,
        "datastore_dir": str((DATASTORE_DIR / DEFAULT_VERSION).resolve()),
        "lmstudio_base_url": LMSTUDIO_BASE_URL,
        "embedding_max_chars": 448,
        "embedding_batch_size": 8,
        "include_edits": True,
        "summary_enabled": bool(summary.get("enabled", False)),
        "summary_model": str(summary.get("model") or "") or None,
        "summary_max_input_chars": int(summary.get("max_input_chars", 6000) or 6000),
        "summary_max_output_tokens": int(summary.get("max_output_tokens", 280) or 280),
        "summary_unit_max_tokens": int(summary.get("unit_max_tokens", 900) or 900),
    }


def _count_datastore_docs(store_dir: Path) -> int | None:
    pages_jsonl = store_dir / "pages.jsonl"
    if not pages_jsonl.exists():
        return None
    doc_ids: set[str] = set()
    try:
        with pages_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = str(rec.get("doc_id") or "").strip()
                if doc_id:
                    doc_ids.add(doc_id)
    except Exception:
        return None
    return len(doc_ids)


def _append_log(job: dict[str, Any], line: str) -> None:
    logs: list[str] = job.get("logs_tail") if isinstance(job.get("logs_tail"), list) else []
    logs.append(line)
    if len(logs) > 220:
        logs = logs[-220:]
    job["logs_tail"] = logs
    job["updated_at"] = _utc_now()


def _prune_reindex_backups(version: str, keep: int = _REINDEX_BACKUP_KEEP) -> None:
    if keep <= 0:
        return
    prefix = f"{version}.bak_"
    try:
        candidates = [p for p in DATASTORE_DIR.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    except FileNotFoundError:
        return
    candidates.sort(key=lambda p: p.name, reverse=True)
    for stale in candidates[keep:]:
        try:
            shutil.rmtree(stale)
        except Exception:
            log.exception("Failed to prune backup %s", stale)


def _update_phase_from_log(job: dict[str, Any], line: str) -> None:
    s = str(line or "")
    if "Ingesting doc:" in s:
        job["phase"] = "converting_html_to_md"
    elif "Indexing from datastore" in s:
        job["phase"] = "indexing_existing_md"
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
    source_datastore = str(job.get("source_datastore") or "").strip()
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
    if source_datastore:
        cmd.extend(["--from-datastore", source_datastore])
    else:
        cmd.extend(["--docs-dir", str(docs_dir)])
    if not bool(job.get("compute_embeddings", True)):
        cmd.append("--no-embeddings")
    if bool(job.get("include_edits", True)):
        cmd.extend(["--include-edits", "--app-db", str(APP_DB_PATH)])
    if bool(job.get("summary_enabled", False)):
        cmd.append("--summary-enabled")
        if job.get("summary_model"):
            cmd.extend(["--summary-model", str(job.get("summary_model"))])
        cmd.extend(
            [
                "--summary-max-input-chars",
                str(int(job.get("summary_max_input_chars") or 6000)),
                "--summary-max-output-tokens",
                str(int(job.get("summary_max_output_tokens") or 280)),
                "--summary-unit-max-tokens",
                str(int(job.get("summary_unit_max_tokens") or 900)),
            ]
        )

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
        _get_search_engine(version).close()
        try:
            if not build_dir.exists():
                raise RuntimeError(f"Build output missing: {build_dir}")

            backup_dir = None
            if target_dir.exists():
                suffix = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                backup_dir = target_dir.with_name(f"{target_dir.name}.bak_{suffix}")
                target_dir.rename(backup_dir)

            user_assets_src = None
            if backup_dir and (backup_dir / "assets" / "user").exists():
                user_assets_src = backup_dir / "assets" / "user"
            elif target_dir.exists() and (target_dir / "assets" / "user").exists():
                user_assets_src = target_dir / "assets" / "user"
            if user_assets_src:
                user_assets_dest = build_dir / "assets" / "user"
                user_assets_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(user_assets_src, user_assets_dest, dirs_exist_ok=True)

            build_dir.rename(target_dir)
            _get_docs_store(version).invalidate()
            _prune_reindex_backups(version)
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
    engine = _get_search_engine(DEFAULT_VERSION)
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "docs_version": DEFAULT_VERSION,
        "datastore_ready": engine.is_ready(),
        "embeddings_ready": engine.embeddings_ready() if engine.is_ready() else False,
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

        defaults = _job_defaults()
        summary_enabled = bool(req.summary_enabled) if req.summary_enabled is not None else bool(defaults.get("summary_enabled"))
        summary_model = req.summary_model if req.summary_model is not None else defaults.get("summary_model")
        summary_max_input_chars = (
            int(req.summary_max_input_chars) if req.summary_max_input_chars is not None else int(defaults.get("summary_max_input_chars") or 6000)
        )
        summary_max_output_tokens = (
            int(req.summary_max_output_tokens) if req.summary_max_output_tokens is not None else int(defaults.get("summary_max_output_tokens") or 280)
        )
        summary_unit_max_tokens = (
            int(req.summary_unit_max_tokens) if req.summary_unit_max_tokens is not None else int(defaults.get("summary_unit_max_tokens") or 900)
        )

        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "running",
            "phase": "queued",
            "docs_dir": str(docs_dir),
            "version": DEFAULT_VERSION,
            "compute_embeddings": bool(req.compute_embeddings),
            "embedding_max_chars": int(req.embedding_max_chars),
            "embedding_batch_size": int(req.embedding_batch_size),
            "include_edits": bool(req.include_edits),
            "summary_enabled": summary_enabled,
            "summary_model": summary_model,
            "summary_max_input_chars": summary_max_input_chars,
            "summary_max_output_tokens": summary_max_output_tokens,
            "summary_unit_max_tokens": summary_unit_max_tokens,
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


@app.post("/admin/refresh", response_model=ReindexStatusResponse, response_model_exclude_none=True)
async def admin_refresh_start(req: ReindexRequest, ctx: AuthContext = Depends(resolve_auth)) -> ReindexStatusResponse:
    require_role(ctx, {"admin"})
    global _reindex_job, _reindex_task

    source_store = (DATASTORE_DIR / DEFAULT_VERSION).resolve()
    if not source_store.exists() or not (source_store / "pages.jsonl").exists():
        raise HTTPException(status_code=400, detail="Datastore not found; run full reindex first.")

    async with _reindex_lock:
        if _reindex_job and _reindex_job.get("status") == "running":
            raise HTTPException(status_code=409, detail="Reindex is already running")

        job_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(4)
        build_dir = (DATASTORE_DIR / f".build_{DEFAULT_VERSION}_{job_id}").resolve()

        defaults = _job_defaults()
        summary_enabled = bool(req.summary_enabled) if req.summary_enabled is not None else bool(defaults.get("summary_enabled"))
        summary_model = req.summary_model if req.summary_model is not None else defaults.get("summary_model")
        summary_max_input_chars = (
            int(req.summary_max_input_chars) if req.summary_max_input_chars is not None else int(defaults.get("summary_max_input_chars") or 6000)
        )
        summary_max_output_tokens = (
            int(req.summary_max_output_tokens) if req.summary_max_output_tokens is not None else int(defaults.get("summary_max_output_tokens") or 280)
        )
        summary_unit_max_tokens = (
            int(req.summary_unit_max_tokens) if req.summary_unit_max_tokens is not None else int(defaults.get("summary_unit_max_tokens") or 900)
        )

        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "running",
            "phase": "queued",
            "mode": "refresh",
            "docs_dir": str(DOCS_DIR),
            "source_datastore": str(source_store),
            "version": DEFAULT_VERSION,
            "compute_embeddings": bool(req.compute_embeddings),
            "embedding_max_chars": int(req.embedding_max_chars),
            "embedding_batch_size": int(req.embedding_batch_size),
            "include_edits": bool(req.include_edits),
            "summary_enabled": summary_enabled,
            "summary_model": summary_model,
            "summary_max_input_chars": summary_max_input_chars,
            "summary_max_output_tokens": summary_max_output_tokens,
            "summary_unit_max_tokens": summary_unit_max_tokens,
            "started_at": _utc_now(),
            "updated_at": _utc_now(),
            "doc_total": _count_datastore_docs(source_store),
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


def _get_doc_title(version: str, doc_id: str) -> str | None:
    store = _get_docs_store(version)
    if store.is_ready():
        try:
            data = store.list_pages(doc_id)
            return str(data.get("doc_title") or doc_id)
        except KeyError:
            pass
    pages = app_db.list_doc_pages(version=version, doc_id=doc_id)
    if pages:
        return str(pages[0].get("doc_title") or doc_id)
    return None


def _get_page_title(version: str, doc_id: str, page_id: str) -> str | None:
    store = _get_docs_store(version)
    if store.is_ready():
        try:
            page = store.get_page(doc_id, page_id)
            return str(page.page_title or page_id)
        except KeyError:
            pass
    custom = app_db.get_doc_page(version=version, doc_id=doc_id, page_id=page_id)
    if custom:
        return str(custom.get("page_title") or page_id)
    return None


def _resolve_doc_page(version: str, doc_id: str, page_id: str) -> tuple[dict[str, Any], str, list[PageImage]]:
    store = _get_docs_store(version)
    if store.is_ready():
        try:
            page = store.get_page(doc_id, page_id)
            md_text = store.read_markdown(page)
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
            return (
                {
                    "doc_id": page.doc_id,
                    "doc_title": page.doc_title,
                    "page_id": page.page_id,
                    "page_title": page.page_title,
                    "heading_path": page.heading_path,
                    "anchor": page.anchor,
                    "source_path": page.source_path,
                    "custom": False,
                    "author_id": None,
                },
                md_text,
                images,
            )
        except KeyError:
            pass

    custom = app_db.get_doc_page(version=version, doc_id=doc_id, page_id=page_id)
    if not custom:
        raise HTTPException(status_code=404, detail="Page not found")
    doc_title = _get_doc_title(version, doc_id) or str(custom.get("doc_title") or doc_id)
    page_title = str(custom.get("page_title") or page_id)
    heading_path = custom.get("heading_path") or [doc_title, page_title]
    return (
        {
            "doc_id": doc_id,
            "doc_title": doc_title,
            "page_id": page_id,
            "page_title": page_title,
            "heading_path": heading_path,
            "anchor": custom.get("anchor"),
            "source_path": str(custom.get("source_path") or ""),
            "custom": True,
            "author_id": custom.get("author_id"),
        },
        str(custom.get("base_markdown") or ""),
        [],
    )


def _doc_exists(version: str, doc_id: str) -> bool:
    store = _get_docs_store(version)
    if store.is_ready():
        try:
            store.list_pages(doc_id)
            return True
        except KeyError:
            pass
    return bool(app_db.list_doc_pages(version=version, doc_id=doc_id))


def _doc_is_ingested(version: str, doc_id: str) -> bool:
    store = _get_docs_store(version)
    if store.is_ready():
        try:
            store.list_pages(doc_id)
            return True
        except KeyError:
            return False
    return False


def _doc_exclusions(version: str) -> set[str]:
    try:
        rows = app_db.list_doc_exclusions(version=version)
    except Exception:
        return set()
    return {str(r.get("doc_id") or "") for r in rows if str(r.get("doc_id") or "").strip()}


def _page_exists(version: str, doc_id: str, page_id: str) -> bool:
    store = _get_docs_store(version)
    if store.is_ready():
        try:
            store.get_page(doc_id, page_id)
            return True
        except KeyError:
            pass
    return bool(app_db.get_doc_page(version=version, doc_id=doc_id, page_id=page_id))


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


def _get_docs_style() -> dict[str, str | None]:
    try:
        effective = get_settings_bundle()["effective"]
    except SettingsError:
        return {
            "heading_font": None,
            "body_font": None,
            "cover_type": "Guide",
            "cover_image": None,
            "cover_text": None,
            "cover_copyright": None,
        }
    style = effective.get("docs_style") if isinstance(effective.get("docs_style"), dict) else {}
    heading = str(style.get("heading_font") or "").strip()
    body = str(style.get("body_font") or "").strip()
    cover_type = str(style.get("cover_type") or "").strip()
    cover_image = str(style.get("cover_image") or "").strip()
    cover_text = str(style.get("cover_text") or "").strip()
    cover_copyright = str(style.get("cover_copyright") or "").strip()
    return {
        "heading_font": heading or None,
        "body_font": body or None,
        "cover_type": cover_type or "Guide",
        "cover_image": cover_image or None,
        "cover_text": cover_text or None,
        "cover_copyright": cover_copyright or None,
    }


def _normalize_font_value(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) > 64:
        raise HTTPException(status_code=400, detail="Font value too long")
    return text


def _normalize_cover_value(value: str | None, *, max_len: int, field: str) -> str | None:
    if value is None:
        return None
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) > max_len:
        raise HTTPException(status_code=400, detail=f"{field} is too long")
    return text


@app.get("/docs/versions", response_model=DocsVersionsResponse)
def docs_versions(ctx: AuthContext = Depends(resolve_auth)) -> DocsVersionsResponse:
    versions = _available_versions()
    if not versions:
        raise HTTPException(status_code=503, detail="Docs catalog is not available. Run ingestion first.")
    return DocsVersionsResponse(default_version=DEFAULT_VERSION, versions=versions)


@app.get("/docs/style", response_model=DocsStyleResponse)
def docs_style(ctx: AuthContext = Depends(resolve_auth)) -> DocsStyleResponse:
    style = _get_docs_style()
    return DocsStyleResponse(
        heading_font=style.get("heading_font"),
        body_font=style.get("body_font"),
        cover_type=style.get("cover_type"),
        cover_image=style.get("cover_image"),
        cover_text=style.get("cover_text"),
        cover_copyright=style.get("cover_copyright"),
    )


@app.post("/docs/style", response_model=DocsStyleResponse)
def docs_style_update(req: DocsStyleUpdateRequest, ctx: AuthContext = Depends(resolve_auth)) -> DocsStyleResponse:
    require_role(ctx, {"admin"})
    style = _get_docs_style()
    heading = _normalize_font_value(req.heading_font)
    body = _normalize_font_value(req.body_font)
    if heading is not None:
        style["heading_font"] = heading
    if body is not None:
        style["body_font"] = body
    if req.cover_type is not None:
        style["cover_type"] = _normalize_cover_value(req.cover_type, max_len=60, field="cover_type")
    if req.cover_image is not None:
        style["cover_image"] = _normalize_cover_value(req.cover_image, max_len=300, field="cover_image")
    if req.cover_text is not None:
        style["cover_text"] = _normalize_cover_value(req.cover_text, max_len=800, field="cover_text")
    if req.cover_copyright is not None:
        style["cover_copyright"] = _normalize_cover_value(
            req.cover_copyright, max_len=120, field="cover_copyright"
        )
    update_settings({"docs_style": style})
    return DocsStyleResponse(
        heading_font=style.get("heading_font"),
        body_font=style.get("body_font"),
        cover_type=style.get("cover_type"),
        cover_image=style.get("cover_image"),
        cover_text=style.get("cover_text"),
        cover_copyright=style.get("cover_copyright"),
    )


@app.get("/docs/catalog", response_model=DocsCatalogResponse)
def docs_catalog(version: str | None = None, ctx: AuthContext = Depends(resolve_auth)) -> DocsCatalogResponse:
    ver = _require_version(version)
    store = _get_docs_store(ver)
    if not store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Docs catalog not found for version '{ver}'. Run the ingestion CLI first.",
        )
    try:
        allow, deny = docs_allowed_for_role(ctx.principal.role)
        docs = [d for d in store.list_docs() if _doc_allowed(d.get("doc_id"), allow, deny)]
        excluded = _doc_exclusions(ver)
        custom_pages = app_db.list_doc_pages(version=ver)
        doc_map: dict[str, dict[str, Any]] = {}
        for d in docs:
            doc_id = str(d.get("doc_id") or "")
            entry = dict(d)
            entry["origin"] = "ingested"
            entry["rag_excluded"] = doc_id in excluded
            doc_map[doc_id] = entry
        ordered = [d["doc_id"] for d in docs]
        for page in custom_pages:
            doc_id = str(page.get("doc_id") or "")
            if not _doc_allowed(doc_id, allow, deny):
                continue
            if doc_id in doc_map:
                doc_map[doc_id]["page_count"] = int(doc_map[doc_id].get("page_count") or 0) + 1
            else:
                doc_map[doc_id] = {
                    "doc_id": doc_id,
                    "doc_title": str(page.get("doc_title") or doc_id),
                    "page_count": 1,
                    "origin": "custom",
                    "rag_excluded": doc_id in excluded,
                }
        custom_only = [d for d in doc_map.keys() if d not in ordered]
        custom_only.sort(key=lambda d: str(doc_map[d].get("doc_title") or d).lower())
        docs_out = [doc_map[d] for d in ordered if d in doc_map] + [doc_map[d] for d in custom_only]
        return DocsCatalogResponse(docs=docs_out)
    except Exception as e:
        log.exception("Docs catalog error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/docs/catalog/{doc_id}", response_model=DocCatalogDetailResponse)
def docs_catalog_doc(doc_id: str, version: str | None = None, ctx: AuthContext = Depends(resolve_auth)) -> dict[str, Any]:
    ver = _require_version(version)
    store = _get_docs_store(ver)
    if not store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Docs catalog not found for version '{ver}'. Run the ingestion CLI first.",
        )
    allow, deny = docs_allowed_for_role(ctx.principal.role)
    excluded_docs = _doc_exclusions(DEFAULT_VERSION)
    if not _doc_allowed(doc_id, allow, deny):
        raise HTTPException(status_code=403, detail="Doc is not available for this user")
    try:
        pages: list[dict[str, Any]] = []
        doc_title = None
        try:
            data = store.list_pages(doc_id)
            doc_title = data.get("doc_title")
            for p in data.get("pages") or []:
                entry = dict(p)
                entry["custom"] = False
                entry["author_id"] = None
                pages.append(entry)
        except KeyError:
            pass

        custom_pages = app_db.list_doc_pages(version=ver, doc_id=doc_id)
        if not doc_title and custom_pages:
            doc_title = str(custom_pages[0].get("doc_title") or doc_id)

        existing_ids = {p.get("page_id") for p in pages}
        for page in custom_pages:
            pid = str(page.get("page_id") or "")
            if pid in existing_ids:
                continue
            heading_path = page.get("heading_path") or [doc_title or doc_id, page.get("page_title") or pid]
            pages.append(
                {
                    "page_id": pid,
                    "page_title": page.get("page_title") or pid,
                    "heading_path": heading_path,
                    "source_path": page.get("source_path") or "",
                    "anchor": page.get("anchor"),
                    "custom": True,
                    "author_id": page.get("author_id"),
                }
            )

        if not pages:
            raise HTTPException(status_code=404, detail="Doc not found")
        return {"doc_id": doc_id, "doc_title": doc_title or doc_id, "pages": pages}
    except Exception as e:
        log.exception("Docs catalog doc error")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/docs/page/{doc_id}/{page_id}", response_model=DocPageResponse)
def docs_page(
    doc_id: str, page_id: str, version: str | None = None, ctx: AuthContext = Depends(resolve_auth)
) -> DocPageResponse:
    ver = _require_version(version)
    store = _get_docs_store(ver)
    if not store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Docs catalog not found for version '{ver}'. Run the ingestion CLI first.",
        )
    allow, deny = docs_allowed_for_role(ctx.principal.role)
    if not _doc_allowed(doc_id, allow, deny):
        raise HTTPException(status_code=403, detail="Doc is not available for this user")
    try:
        page_meta, md_text, images = _resolve_doc_page(ver, doc_id, page_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Docs page error")
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        published = app_db.get_doc_edit(
            version=ver, doc_id=page_meta["doc_id"], page_id=page_meta["page_id"], status="published"
        )
        if published and str(published.get("content_md") or "").strip():
            md_text = str(published["content_md"])
    except Exception:
        log.exception("Failed to load published doc edit")
    _, md_text = _split_front_matter_text(md_text)

    return DocPageResponse(
        version=ver,
        doc_id=page_meta["doc_id"],
        doc_title=page_meta["doc_title"],
        page_id=page_meta["page_id"],
        page_title=page_meta["page_title"],
        heading_path=page_meta["heading_path"],
        anchor=page_meta.get("anchor"),
        source_path=page_meta.get("source_path") or "",
        markdown=md_text,
        images=images,
        custom=bool(page_meta.get("custom")),
        author_id=page_meta.get("author_id"),
    )


def _get_effective_markdown(version: str, doc_id: str, page_id: str) -> tuple[dict[str, Any], str]:
    meta, md_text, _ = _resolve_doc_page(version, doc_id, page_id)
    try:
        published = app_db.get_doc_edit(version=version, doc_id=meta["doc_id"], page_id=meta["page_id"], status="published")
        if published and str(published.get("content_md") or "").strip():
            md_text = str(published["content_md"])
    except Exception:
        log.exception("Failed to load published doc edit for PDF export")
    return meta, md_text


def _can_edit_docs(ctx: AuthContext) -> bool:
    if ctx.principal.role == "admin":
        return True
    return bool(getattr(ctx.principal, "docs_edit", False))


def _can_publish_docs(ctx: AuthContext) -> bool:
    if ctx.principal.role == "admin":
        return True
    return bool(getattr(ctx.principal, "docs_publish", False))


def _assert_edit_role(ctx: AuthContext) -> None:
    if not _can_edit_docs(ctx):
        raise HTTPException(status_code=403, detail="Editing requires permission")


def _select_edit_version(version: str | None) -> str:
    return _require_version(version)


def _to_doc_edit_info(rec: dict[str, Any] | None) -> DocEditInfo | None:
    if not rec:
        return None
    return DocEditInfo(
        edit_id=str(rec.get("edit_id") or ""),
        status=str(rec.get("status") or "draft"),  # type: ignore[arg-type]
        content_md=str(rec.get("content_md") or ""),
        author_id=str(rec.get("author_id") or ""),
        created_at=str(rec.get("created_at") or ""),
        updated_at=str(rec.get("updated_at") or ""),
    )


def _to_publish_request_info(rec: dict[str, Any] | None) -> PublishRequestInfo | None:
    if not rec:
        return None
    return PublishRequestInfo(
        status=str(rec.get("status") or "pending"),  # type: ignore[arg-type]
        content_md=str(rec.get("content_md") or ""),
        author_id=str(rec.get("author_id") or ""),
        created_at=str(rec.get("created_at") or ""),
        updated_at=str(rec.get("updated_at") or ""),
        reviewed_by=str(rec.get("reviewed_by") or "") or None,
        reviewed_at=str(rec.get("reviewed_at") or "") or None,
        review_note=str(rec.get("review_note") or "") or None,
    )


def _to_publish_request_item(rec: dict[str, Any], user_map: dict[str, str]) -> PublishRequestItem:
    version = str(rec.get("version") or DEFAULT_VERSION)
    doc_id = str(rec.get("doc_id") or "")
    page_id = str(rec.get("page_id") or "")
    author_id = str(rec.get("author_id") or "")
    return PublishRequestItem(
        version=version,
        doc_id=doc_id,
        doc_title=_get_doc_title(version, doc_id) or doc_id,
        page_id=page_id,
        page_title=_get_page_title(version, doc_id, page_id) or page_id,
        status=str(rec.get("status") or "pending"),  # type: ignore[arg-type]
        content_md=str(rec.get("content_md") or ""),
        author_id=author_id,
        author_username=user_map.get(author_id),
        created_at=str(rec.get("created_at") or ""),
        updated_at=str(rec.get("updated_at") or ""),
        reviewed_by=str(rec.get("reviewed_by") or "") or None,
        reviewed_at=str(rec.get("reviewed_at") or "") or None,
        review_note=str(rec.get("review_note") or "") or None,
    )


@app.get("/docs/page/{doc_id}/{page_id}/edit", response_model=DocEditResponse)
def docs_page_edit(
    doc_id: str, page_id: str, version: str | None = None, ctx: AuthContext = Depends(resolve_auth)
) -> DocEditResponse:
    _assert_edit_role(ctx)
    ver = _select_edit_version(version)
    store = _get_docs_store(ver)
    if not store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Docs catalog not found for version '{ver}'. Run the ingestion CLI first.",
        )
    allow, deny = docs_allowed_for_role(ctx.principal.role)
    if not _doc_allowed(doc_id, allow, deny):
        raise HTTPException(status_code=403, detail="Doc is not available for this user")
    try:
        page_meta, base_md, _ = _resolve_doc_page(ver, doc_id, page_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Docs page edit error")
        raise HTTPException(status_code=500, detail=str(e)) from e

    draft = app_db.get_doc_edit(version=ver, doc_id=page_meta["doc_id"], page_id=page_meta["page_id"], status="draft")
    published = app_db.get_doc_edit(
        version=ver, doc_id=page_meta["doc_id"], page_id=page_meta["page_id"], status="published"
    )
    publish_req = app_db.get_doc_publish_request(
        version=ver, doc_id=page_meta["doc_id"], page_id=page_meta["page_id"]
    )
    effective_md = str(published.get("content_md")) if published else base_md
    can_edit = _can_edit_docs(ctx)
    can_publish = _can_publish_docs(ctx)
    can_request_publish = bool(can_edit and not can_publish)

    return DocEditResponse(
        version=ver,
        doc_id=page_meta["doc_id"],
        doc_title=page_meta["doc_title"],
        page_id=page_meta["page_id"],
        page_title=page_meta["page_title"],
        heading_path=page_meta["heading_path"],
        base_markdown=base_md,
        effective_markdown=effective_md,
        draft=_to_doc_edit_info(draft),
        published=_to_doc_edit_info(published),
        publish_request=_to_publish_request_info(publish_req),
        can_edit=bool(can_edit),
        can_publish=bool(can_publish),
        can_request_publish=can_request_publish,
    )


@app.post("/docs/page/{doc_id}/{page_id}/edit", response_model=DocEditResponse)
def docs_page_edit_save(
    doc_id: str, page_id: str, req: DocEditRequest, ctx: AuthContext = Depends(resolve_auth)
) -> DocEditResponse:
    _assert_edit_role(ctx)
    ver = _select_edit_version(req.version)
    store = _get_docs_store(ver)
    if not store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Docs catalog not found for version '{ver}'. Run the ingestion CLI first.",
        )
    allow, deny = docs_allowed_for_role(ctx.principal.role)
    if not _doc_allowed(doc_id, allow, deny):
        raise HTTPException(status_code=403, detail="Doc is not available for this user")
    status = str(req.status or "draft").strip()
    if status not in ("draft", "published"):
        raise HTTPException(status_code=400, detail="Invalid edit status")
    if status == "published" and not _can_publish_docs(ctx):
        raise HTTPException(status_code=403, detail="Publishing requires permission")
    content = str(req.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Content is empty")
    try:
        page_meta, _, _ = _resolve_doc_page(ver, doc_id, page_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    app_db.upsert_doc_edit(
        version=ver,
        doc_id=page_meta["doc_id"],
        page_id=page_meta["page_id"],
        status=status,
        content_md=content,
        author_id=str(ctx.principal.user_id or "unknown"),
    )
    # Return latest edit view
    return docs_page_edit(doc_id, page_id, version=ver, ctx=ctx)


@app.post("/docs/page/{doc_id}/{page_id}/edit/delete", response_model=DocEditResponse)
def docs_page_edit_delete(
    doc_id: str, page_id: str, req: DocEditDeleteRequest, ctx: AuthContext = Depends(resolve_auth)
) -> DocEditResponse:
    _assert_edit_role(ctx)
    status = str(req.status or "draft").strip()
    if status not in ("draft", "published"):
        raise HTTPException(status_code=400, detail="Invalid edit status")
    if status == "published" and not _can_publish_docs(ctx):
        raise HTTPException(status_code=403, detail="Publishing requires permission")
    ver = _select_edit_version(req.version)
    app_db.delete_doc_edit(version=ver, doc_id=doc_id, page_id=page_id, status=status)
    return docs_page_edit(doc_id, page_id, version=ver, ctx=ctx)


@app.post("/docs/page/{doc_id}/{page_id}/publish/request", response_model=DocEditResponse)
def docs_page_publish_request(
    doc_id: str, page_id: str, req: DocPublishRequest, ctx: AuthContext = Depends(resolve_auth)
) -> DocEditResponse:
    _assert_edit_role(ctx)
    ver = _select_edit_version(req.version)
    store = _get_docs_store(ver)
    if not store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Docs catalog not found for version '{ver}'. Run the ingestion CLI first.",
        )
    allow, deny = docs_allowed_for_role(ctx.principal.role)
    if not _doc_allowed(doc_id, allow, deny):
        raise HTTPException(status_code=403, detail="Doc is not available for this user")
    content = str(req.content or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Content is empty")
    try:
        page_meta, _, _ = _resolve_doc_page(ver, doc_id, page_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    app_db.upsert_doc_publish_request(
        version=ver,
        doc_id=page_meta["doc_id"],
        page_id=page_meta["page_id"],
        content_md=content,
        author_id=str(ctx.principal.user_id or "unknown"),
    )
    return docs_page_edit(doc_id, page_id, version=ver, ctx=ctx)


@app.post("/docs/page/{doc_id}/{page_id}/delete", response_model=OkResponse)
def docs_page_delete(
    doc_id: str, page_id: str, req: DocDeleteRequest, ctx: AuthContext = Depends(resolve_auth)
) -> OkResponse:
    _assert_edit_role(ctx)
    ver = _select_edit_version(req.version)
    custom = app_db.get_doc_page(version=ver, doc_id=doc_id, page_id=page_id)
    if not custom:
        store = _get_docs_store(ver)
        if store.is_ready():
            try:
                store.get_page(doc_id, page_id)
                raise HTTPException(
                    status_code=400,
                    detail="Cannot delete an ingested page. Remove edits or exclude the guide from RAG.",
                )
            except KeyError:
                pass
        raise HTTPException(status_code=404, detail="Page not found")
    if ctx.principal.role != "admin":
        if not ctx.principal.user_id or str(custom.get("author_id") or "") != str(ctx.principal.user_id):
            raise HTTPException(status_code=403, detail="Only the owner or admin can delete this page")
    app_db.delete_doc_edits_for_page(version=ver, doc_id=doc_id, page_id=page_id)
    app_db.delete_doc_page(version=ver, doc_id=doc_id, page_id=page_id)
    return OkResponse()


@app.get("/admin/publish-requests", response_model=PublishRequestListResponse)
def admin_list_publish_requests(
    status: str | None = None,
    version: str | None = None,
    ctx: AuthContext = Depends(resolve_auth),
) -> PublishRequestListResponse:
    require_role(ctx, {"admin"})
    ver = _select_edit_version(version)
    desired = str(status or "pending").strip().lower()
    if desired not in ("pending", "approved", "rejected", "all"):
        raise HTTPException(status_code=400, detail="Invalid status filter")
    user_map = {str(u.get("user_id") or ""): str(u.get("username") or "") for u in app_db.list_users(limit=1000)}
    rows = app_db.list_doc_publish_requests(version=ver, status=None if desired == "all" else desired)
    items = [_to_publish_request_item(r, user_map) for r in rows]
    return PublishRequestListResponse(requests=items)


@app.post("/admin/publish-requests/{doc_id}/{page_id}/approve", response_model=PublishRequestItem)
def admin_approve_publish_request(
    doc_id: str, page_id: str, req: PublishRequestDecision, ctx: AuthContext = Depends(resolve_auth)
) -> PublishRequestItem:
    require_role(ctx, {"admin"})
    ver = _select_edit_version(req.version)
    rec = app_db.get_doc_publish_request(version=ver, doc_id=doc_id, page_id=page_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Publish request not found")
    if str(rec.get("status") or "") != "pending":
        raise HTTPException(status_code=409, detail="Publish request is not pending")
    content = str(rec.get("content_md") or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Publish request content is empty")
    app_db.upsert_doc_edit(
        version=ver,
        doc_id=str(rec.get("doc_id") or doc_id),
        page_id=str(rec.get("page_id") or page_id),
        status="published",
        content_md=content,
        author_id=str(rec.get("author_id") or ctx.principal.user_id or "unknown"),
    )
    updated = app_db.update_doc_publish_request_status(
        version=ver,
        doc_id=doc_id,
        page_id=page_id,
        status="approved",
        reviewed_by=str(ctx.principal.user_id or "admin"),
        review_note=req.note,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Publish request not found")
    user_map = {str(u.get("user_id") or ""): str(u.get("username") or "") for u in app_db.list_users(limit=1000)}
    return _to_publish_request_item(updated, user_map)


@app.post("/admin/publish-requests/{doc_id}/{page_id}/reject", response_model=PublishRequestItem)
def admin_reject_publish_request(
    doc_id: str, page_id: str, req: PublishRequestDecision, ctx: AuthContext = Depends(resolve_auth)
) -> PublishRequestItem:
    require_role(ctx, {"admin"})
    ver = _select_edit_version(req.version)
    rec = app_db.get_doc_publish_request(version=ver, doc_id=doc_id, page_id=page_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Publish request not found")
    if str(rec.get("status") or "") != "pending":
        raise HTTPException(status_code=409, detail="Publish request is not pending")
    updated = app_db.update_doc_publish_request_status(
        version=ver,
        doc_id=doc_id,
        page_id=page_id,
        status="rejected",
        reviewed_by=str(ctx.principal.user_id or "admin"),
        review_note=req.note,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Publish request not found")
    user_map = {str(u.get("user_id") or ""): str(u.get("username") or "") for u in app_db.list_users(limit=1000)}
    return _to_publish_request_item(updated, user_map)


@app.post("/docs/guide/{doc_id}/delete", response_model=OkResponse)
def docs_guide_delete(doc_id: str, req: DocDeleteRequest, ctx: AuthContext = Depends(resolve_auth)) -> OkResponse:
    _assert_edit_role(ctx)
    ver = _select_edit_version(req.version)
    if _doc_is_ingested(ver, doc_id):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete an ingested guide. Use 'exclude from RAG' instead.",
        )
    pages = app_db.list_doc_pages(version=ver, doc_id=doc_id)
    if not pages:
        raise HTTPException(status_code=404, detail="Guide not found")
    if ctx.principal.role != "admin":
        if not ctx.principal.user_id:
            raise HTTPException(status_code=403, detail="Only the owner or admin can delete this guide")
        owner_id = str(ctx.principal.user_id)
        for page in pages:
            if str(page.get("author_id") or "") != owner_id:
                raise HTTPException(status_code=403, detail="Only the owner or admin can delete this guide")
    app_db.delete_doc_edits_for_doc(version=ver, doc_id=doc_id)
    app_db.delete_doc_pages_for_doc(version=ver, doc_id=doc_id)
    app_db.delete_doc_exclusion(version=ver, doc_id=doc_id)
    return OkResponse()


@app.post("/docs/guide/{doc_id}/exclude", response_model=DocExcludeResponse)
def docs_guide_exclude(doc_id: str, req: DocExcludeRequest, ctx: AuthContext = Depends(resolve_auth)) -> DocExcludeResponse:
    require_role(ctx, {"admin"})
    ver = _require_version(req.version)
    exclude = bool(req.excluded)
    if exclude:
        app_db.upsert_doc_exclusion(
            version=ver,
            doc_id=doc_id,
            excluded_by=str(ctx.principal.user_id or "admin"),
            reason=str(req.reason or "").strip() or None,
        )
    else:
        app_db.delete_doc_exclusion(version=ver, doc_id=doc_id)
    return DocExcludeResponse(doc_id=doc_id, rag_excluded=exclude)


@app.post("/docs/pdf")
def docs_pdf(req: DocPdfRequest, ctx: AuthContext = Depends(resolve_auth)) -> Response:
    _assert_edit_role(ctx)
    ver = _select_edit_version(req.version)
    content = str(req.markdown or "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Content is empty")
    title = str(req.title or "document").strip() or "document"
    style = _get_docs_style()
    cover = {
        "guide_type": style.get("cover_type") or "Guide",
        "title": title,
        "image": style.get("cover_image"),
        "text": style.get("cover_text"),
        "version": ver,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "copyright": style.get("cover_copyright"),
    }
    pdf_bytes = render_markdown_to_pdf(
        content,
        title=title,
        version=ver,
        heading_font=style.get("heading_font"),
        body_font=style.get("body_font"),
        cover=cover,
    )
    filename = f"{_safe_slug(title)}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/docs/guide/{doc_id}/pdf")
def docs_guide_pdf(
    doc_id: str, version: str | None = None, ctx: AuthContext = Depends(resolve_auth)
) -> Response:
    _assert_edit_role(ctx)
    ver = _select_edit_version(version)
    store = _get_docs_store(ver)
    if not store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Docs catalog not found for version '{ver}'. Run the ingestion CLI first.",
        )
    allow, deny = docs_allowed_for_role(ctx.principal.role)
    if not _doc_allowed(doc_id, allow, deny):
        raise HTTPException(status_code=403, detail="Doc is not available for this user")

    pages: list[dict[str, Any]] = []
    doc_title = _get_doc_title(ver, doc_id) or doc_id
    if store.is_ready():
        try:
            data = store.list_pages(doc_id)
            doc_title = str(data.get("doc_title") or doc_title)
            pages = list(data.get("pages") or [])
        except KeyError:
            pages = []

    custom_pages = app_db.list_doc_pages(version=ver, doc_id=doc_id)
    existing = {str(p.get("page_id") or "") for p in pages}
    for rec in custom_pages:
        pid = str(rec.get("page_id") or "")
        if not pid or pid in existing:
            continue
        pages.append(
            {
                "page_id": pid,
                "page_title": str(rec.get("page_title") or pid),
            }
        )
        existing.add(pid)

    if not pages:
        raise HTTPException(status_code=404, detail="Doc not found")

    combined: list[str] = []
    for idx, p in enumerate(pages):
        pid = str(p.get("page_id") or "")
        if not pid:
            continue
        meta, md_text = _get_effective_markdown(ver, doc_id, pid)
        page_title = str(p.get("page_title") or meta.get("page_title") or pid)
        content = str(md_text or "").strip()
        if idx == 0:
            front, body = _split_front_matter_text(content)
            if front:
                combined.append(front.rstrip())
            content = body.strip()
        if page_title:
            combined.append(f"[[DOC_SECTION: {page_title}]]")
        if content:
            combined.append(content)
        combined.append("")

    style = _get_docs_style()
    cover = {
        "guide_type": style.get("cover_type") or "Guide",
        "title": doc_title,
        "image": style.get("cover_image"),
        "text": style.get("cover_text"),
        "version": ver,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "copyright": style.get("cover_copyright"),
    }
    pdf_bytes = render_markdown_to_pdf(
        "\n".join(combined),
        title=doc_title,
        version=ver,
        heading_font=style.get("heading_font"),
        body_font=style.get("body_font"),
        cover=cover,
    )
    filename = f"{_safe_slug(doc_title)}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _content_starts_with_heading(markdown: str, heading: str) -> bool:
    normalized_heading = re.sub(r"\s+", " ", str(heading or "")).strip().lower()
    if not normalized_heading:
        return False
    for raw in str(markdown or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if not match:
            return False
        text = re.sub(r"\s+", " ", match.group(2)).strip().lower()
        return text == normalized_heading
    return False


def _split_front_matter_text(markdown: str) -> tuple[str | None, str]:
    text = str(markdown or "")
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    if not text.startswith("---"):
        return None, markdown
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, markdown
    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return None, markdown
    front = "\n".join(lines[: end_idx + 1]) + "\n"
    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    return front, body


def _safe_slug(value: str) -> str:
    import re

    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "").strip())
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "item"


def _unique_doc_id(version: str, base: str) -> str:
    root = _safe_slug(base)
    candidate = root
    idx = 1
    while _doc_exists(version, candidate):
        candidate = f"{root}-{idx}"
        idx += 1
    return candidate


def _unique_page_id(version: str, doc_id: str, base: str) -> str:
    root = _safe_slug(base)
    candidate = root
    idx = 1
    while _page_exists(version, doc_id, candidate):
        candidate = f"{root}-{idx}"
        idx += 1
    return candidate


@app.post("/docs/guide", response_model=DocCreateResponse)
def docs_guide_create(req: DocGuideCreateRequest, ctx: AuthContext = Depends(resolve_auth)) -> DocCreateResponse:
    _assert_edit_role(ctx)
    ver = _select_edit_version(req.version)
    store = _get_docs_store(ver)
    if not store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Docs catalog not found for version '{ver}'. Run the ingestion CLI first.",
        )

    doc_title = str(req.doc_title or "").strip()
    if not doc_title:
        raise HTTPException(status_code=400, detail="doc_title is required")
    doc_id = _unique_doc_id(ver, doc_title)
    page_title = str(req.page_title or "").strip() or "Overview"
    page_id = _unique_page_id(ver, doc_id, page_title)
    base_md = f"# {page_title}\n\n"
    heading_path = [doc_title, page_title]
    source_path = f"custom/{doc_id}/{page_id}.md"

    try:
        rec = app_db.create_doc_page(
            version=ver,
            doc_id=doc_id,
            page_id=page_id,
            doc_title=doc_title,
            page_title=page_title,
            heading_path=heading_path,
            source_path=source_path,
            base_markdown=base_md,
            author_id=str(ctx.principal.user_id or "unknown"),
        )
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail="Guide already exists") from e

    return DocCreateResponse(
        version=ver,
        doc_id=doc_id,
        doc_title=doc_title,
        page_id=rec.get("page_id") or page_id,
        page_title=rec.get("page_title") or page_title,
    )


@app.post("/docs/section", response_model=DocCreateResponse)
def docs_section_create(req: DocSectionCreateRequest, ctx: AuthContext = Depends(resolve_auth)) -> DocCreateResponse:
    _assert_edit_role(ctx)
    ver = _select_edit_version(req.version)
    store = _get_docs_store(ver)
    if not store.is_ready():
        raise HTTPException(
            status_code=503,
            detail=f"Docs catalog not found for version '{ver}'. Run the ingestion CLI first.",
        )
    doc_id = str(req.doc_id or "").strip()
    if not doc_id:
        raise HTTPException(status_code=400, detail="doc_id is required")
    doc_title = _get_doc_title(ver, doc_id)
    if not doc_title:
        raise HTTPException(status_code=404, detail="Doc not found")

    page_title = str(req.page_title or "").strip()
    if not page_title:
        raise HTTPException(status_code=400, detail="page_title is required")
    page_id = _unique_page_id(ver, doc_id, page_title)
    base_md = f"# {page_title}\n\n"
    heading_path = [doc_title, page_title]
    source_path = f"custom/{doc_id}/{page_id}.md"

    try:
        rec = app_db.create_doc_page(
            version=ver,
            doc_id=doc_id,
            page_id=page_id,
            doc_title=doc_title,
            page_title=page_title,
            heading_path=heading_path,
            source_path=source_path,
            base_markdown=base_md,
            author_id=str(ctx.principal.user_id or "unknown"),
        )
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail="Section already exists") from e

    return DocCreateResponse(
        version=ver,
        doc_id=doc_id,
        doc_title=doc_title,
        page_id=rec.get("page_id") or page_id,
        page_title=rec.get("page_title") or page_title,
    )


@app.post("/docs/assets/upload", response_model=DocAssetUploadResponse)
async def docs_asset_upload(
    file: UploadFile = File(...),
    doc_id: str = Form(""),
    page_id: str = Form(""),
    version: str | None = Form(None),
    ctx: AuthContext = Depends(resolve_auth),
) -> DocAssetUploadResponse:
    _assert_edit_role(ctx)
    ver = _select_edit_version(version)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    if not str(file.content_type or "").lower().startswith("image/"):
        raise HTTPException(status_code=400, detail="Unsupported file type")

    safe_doc = _safe_slug(doc_id)
    safe_page = _safe_slug(page_id)
    safe_name = _safe_slug(Path(file.filename or "image").stem) + ext

    assets_dir = DATASTORE_DIR / ver / "assets" / "user" / safe_doc / safe_page
    assets_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_name = f"{ts}_{safe_name}"
    out_path = assets_dir / out_name

    max_bytes = 5 * 1024 * 1024
    size = 0
    try:
        with out_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=400, detail="File too large (max 5MB)")
                f.write(chunk)
    finally:
        await file.close()

    url = f"/assets/{ver}/user/{safe_doc}/{safe_page}/{out_name}"
    return DocAssetUploadResponse(url=url, filename=out_name, version=ver)


def _doc_source_path(version: str, doc_id: str, page_id: str, source_rel: str | None) -> str:
    source_rel = str(source_rel or "")
    if source_rel and not source_rel.startswith("custom/"):
        return f"/rawdocs/{version}/{source_rel}"
    return f"/docs/page/{doc_id}/{page_id}?version={version}"


def _build_citations(results: list[dict[str, Any]], version: str, max_citations: int) -> list[Citation]:
    seen: set[tuple[str, str]] = set()
    citations: list[Citation] = []
    for r in results:
        key = (r["doc_id"], r["page_id"])
        if key in seen:
            continue
        seen.add(key)
        title = (r.get("heading_path") or [r["page_id"]])[-1]
        source_path = _doc_source_path(version, r["doc_id"], r["page_id"], r.get("source_path"))
        citations.append(
            Citation(
                title=title,
                doc_id=r["doc_id"],
                page_id=r["page_id"],
                anchor=r.get("anchor"),
                source_path=source_path,
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


def _build_context_chunks(results: list[dict[str, Any]], version: str) -> list[ContextChunk]:
    out: list[ContextChunk] = []
    for idx, r in enumerate(results, start=1):
        out.append(
            ContextChunk(
                index=idx,
                chunk_id=str(r.get("chunk_id") or ""),
                doc_id=r["doc_id"],
                page_id=r["page_id"],
                heading_path=list(r.get("heading_path") or []),
                anchor=r.get("anchor"),
                source_path=_doc_source_path(version, r["doc_id"], r["page_id"], r.get("source_path")),
                images=list(r.get("images") or []),
                score=float(r["score"]) if r.get("score") is not None else None,
            )
        )
    return out


_REQUEST_MORE_CONTEXT = "REQUEST_MORE_CONTEXT"


def _strip_json_fence(payload: str) -> str:
    text = payload.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    if not lines[-1].strip().startswith("```"):
        return text
    return "\n".join(lines[1:-1]).strip()


def _parse_context_request(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    lines = text.strip().splitlines()
    if not lines:
        return None
    if lines[0].strip() != _REQUEST_MORE_CONTEXT:
        return None
    payload = "\n".join(lines[1:]).strip()
    if not payload:
        return None
    payload = _strip_json_fence(payload)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    query = str(data.get("query") or "").strip()
    if not query:
        return None
    req: dict[str, Any] = {"query": query}
    reason = str(data.get("reason") or "").strip()
    if reason:
        req["reason"] = reason
    if "k" in data:
        try:
            req["k"] = int(data.get("k") or 0)
        except (TypeError, ValueError):
            req["k"] = 0
    doc_ids = data.get("doc_ids")
    if isinstance(doc_ids, str):
        doc_ids = [doc_ids]
    if isinstance(doc_ids, list):
        req["doc_ids"] = [str(x).strip() for x in doc_ids if str(x).strip()]
    page_ids = data.get("page_ids")
    if isinstance(page_ids, str):
        page_ids = [page_ids]
    if isinstance(page_ids, list):
        req["page_ids"] = [str(x).strip() for x in page_ids if str(x).strip()]
    return req


def _merge_retrieval_results(
    dest: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    seen: set[str],
) -> int:
    added = 0
    for row in incoming:
        cid = str(row.get("chunk_id") or "").strip()
        key = cid or f"{row.get('doc_id')}:{row.get('page_id')}:{row.get('heading_path')}"
        if key in seen:
            continue
        seen.add(key)
        dest.append(row)
        added += 1
    return added


def _build_context_text(results: list[dict[str, Any]], web_context_blocks: list[str]) -> str:
    context_blocks: list[str] = []
    for i, r in enumerate(results, start=1):
        hp = " > ".join(r.get("heading_path") or [])
        source = f"{r['doc_id']}/{r['page_id']}"
        context_blocks.append(f"[{i}] {source} | {hp}\n{r['text']}")
    if web_context_blocks:
        context_blocks.append("EXTERNAL WEB CONTEXT (not Luxriot EVO docs):")
        context_blocks.extend(web_context_blocks)
    return "\n\n---\n\n".join(context_blocks) if context_blocks else "(no matches)"


@app.post("/docs/search", response_model=SearchResponse, response_model_exclude_none=True)
async def docs_search(req: SearchRequest, ctx: AuthContext = Depends(resolve_auth)) -> SearchResponse:
    ver = _require_version(req.version)
    engine = _get_search_engine(ver)
    if not engine.is_ready():
        log.error("Search index missing; run ingestion CLI to create datastore/%s/index.sqlite", ver)
        raise HTTPException(
            status_code=503,
            detail=f"Search index not found for version '{ver}'. Run the ingestion CLI first.",
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
    expand_include_images = bool(expand.get("include_images", False))
    summary = retrieval.get("summary") if isinstance(retrieval.get("summary"), dict) else {}
    summary_enabled = bool(summary.get("enabled", False))
    summary_k = int(summary.get("k", 0) or 0)
    summary_max_pages = int(summary.get("max_pages", 0) or 0)

    allow, deny = docs_allowed_for_role(ctx.principal.role)
    excluded_docs = _doc_exclusions(ver)
    search_k = int(req.k)
    buffer_k = min(max(search_k * 5, search_k + 10), 200)

    try:
        debug: dict[str, Any] | None = {} if bool(getattr(req, "debug", False)) else None
        async with _search_lock:
            results = await engine.search(
                req.query,
                k=buffer_k,
                mode=mode,
                mmr_enabled=mmr_enabled,
                mmr_lambda=mmr_lambda,
                mmr_candidates=mmr_candidates,
                mmr_use_embeddings=mmr_use_embeddings,
                expand_neighbors=expand_neighbors,
                expand_max_chars=expand_max_chars,
                expand_include_images=expand_include_images,
                heading_boost=heading_boost,
                bm25_candidates=bm25_candidates,
                embedding_candidates=embedding_candidates,
                rrf_k=rrf_k,
                bm25_weight=bm25_weight,
                embedding_weight=embedding_weight,
                doc_priority=[str(x) for x in doc_priority],
                doc_priority_boost=doc_priority_boost,
                summary_enabled=summary_enabled,
                summary_k=summary_k,
                summary_max_pages=summary_max_pages,
                max_per_page=max_per_page,
                max_per_doc=max_per_doc,
                debug_out=debug,
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    results = [
        r
        for r in results
        if _doc_allowed(r.get("doc_id"), allow, deny)
        and str(r.get("doc_id") or "") not in excluded_docs
    ][:search_k]
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
    citations = _build_citations(results, ver, max_citations=max_citations)
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
    engine = _get_search_engine(DEFAULT_VERSION)
    if not engine.is_ready():
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
    expand_include_images = bool(expand.get("include_images", False))
    summary = retrieval_cfg.get("summary") if isinstance(retrieval_cfg.get("summary"), dict) else {}
    summary_enabled = bool(summary.get("enabled", False))
    summary_k = int(summary.get("k", 0) or 0)
    summary_max_pages = int(summary.get("max_pages", 0) or 0)

    allow, deny = docs_allowed_for_role(ctx.principal.role)
    search_k = int(req.k)
    tool_cfg = retrieval_cfg.get("tool_calls") if isinstance(retrieval_cfg.get("tool_calls"), dict) else {}
    tool_calls_enabled = bool(tool_cfg.get("enabled", False))
    tool_calls_limit = int(tool_cfg.get("max_calls", 0) or 0)

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

    async def run_retrieval(
        query: str,
        k: int,
        *,
        doc_ids: list[str] | None = None,
        page_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        buffer_k = min(max(k * 5, k + 10), 200)
        try:
            async with _search_lock:
                rows = await engine.search(
                    query,
                    k=buffer_k,
                    mode=retrieval_mode,
                    mmr_enabled=mmr_enabled,
                    mmr_lambda=mmr_lambda,
                    mmr_candidates=mmr_candidates,
                    mmr_use_embeddings=mmr_use_embeddings,
                    expand_neighbors=expand_neighbors,
                    expand_max_chars=expand_max_chars,
                    expand_include_images=expand_include_images,
                    heading_boost=heading_boost,
                    bm25_candidates=bm25_candidates,
                    embedding_candidates=embedding_candidates,
                    rrf_k=rrf_k,
                    bm25_weight=bm25_weight,
                    embedding_weight=embedding_weight,
                    doc_priority=doc_priority_list,
                    doc_priority_boost=doc_priority_boost,
                    summary_enabled=summary_enabled,
                    summary_k=summary_k,
                    summary_max_pages=summary_max_pages,
                    max_per_page=max_per_page,
                    max_per_doc=max_per_doc,
                )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        rows = [
            r
            for r in rows
            if _doc_allowed(r.get("doc_id"), allow, deny)
            and str(r.get("doc_id") or "") not in excluded_docs
        ]
        if doc_ids:
            allowed_docs = {str(x) for x in doc_ids}
            rows = [r for r in rows if str(r.get("doc_id") or "") in allowed_docs]
        if page_ids:
            allowed_pages = {str(x) for x in page_ids}
            rows = [r for r in rows if str(r.get("page_id") or "") in allowed_pages]
        return rows[:k]

    all_results: list[dict[str, Any]] = []
    seen_results: set[str] = set()
    tool_calls_used = 0
    current_query = req.message
    current_k = search_k
    current_doc_ids: list[str] | None = None
    current_page_ids: list[str] | None = None

    while True:
        retrieval = await run_retrieval(
            current_query,
            current_k,
            doc_ids=current_doc_ids,
            page_ids=current_page_ids,
        )
        log.info(
            "chat retrieval pass query=%r k=%s results=%s",
            current_query,
            current_k,
            len(retrieval),
        )
        _merge_retrieval_results(all_results, retrieval, seen_results)

        context_text = _build_context_text(all_results, web_context_blocks)
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
                    "tool_call_limit": str(tool_calls_limit),
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

        request = _parse_context_request(answer) if tool_calls_enabled else None
        if not request:
            break

        tool_calls_used += 1
        if tool_calls_limit > 0 and tool_calls_used >= tool_calls_limit:
            log.warning("chat tool call limit reached (%s)", tool_calls_limit)
            raise HTTPException(status_code=429, detail="Tool call limit reached for this request.")

        log.info(
            "chat model requested more context reason=%r query=%r k=%s doc_ids=%s page_ids=%s",
            request.get("reason"),
            request.get("query"),
            request.get("k"),
            request.get("doc_ids"),
            request.get("page_ids"),
        )
        current_query = str(request.get("query") or req.message).strip() or req.message
        requested_k = int(request.get("k") or 0)
        if requested_k <= 0:
            current_k = search_k
        else:
            current_k = max(1, min(25, requested_k))
        current_doc_ids = request.get("doc_ids") or None
        current_page_ids = request.get("page_ids") or None

    max_citations = int(retrieval_cfg.get("max_citations", 8))
    max_images = int(retrieval_cfg.get("max_images", 6))
    citations = _build_citations(all_results, DEFAULT_VERSION, max_citations=max_citations)
    images = _build_images(all_results, max_images=max_images)
    context_chunks = _build_context_chunks(all_results, DEFAULT_VERSION)

    app_db.insert_message(session_id=session_id, role="assistant", content=answer)
    return ChatResponse(
        answer=answer,
        citations=citations,
        images=images,
        context_chunks=context_chunks,
        web_sources=web_sources,
        session_id=session_id,
    )


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request, ctx: AuthContext = Depends(resolve_auth)) -> StreamingResponse:
    engine = _get_search_engine(DEFAULT_VERSION)
    if not engine.is_ready():
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
    expand_include_images = bool(expand.get("include_images", False))
    summary = retrieval_cfg.get("summary") if isinstance(retrieval_cfg.get("summary"), dict) else {}
    summary_enabled = bool(summary.get("enabled", False))
    summary_k = int(summary.get("k", 0) or 0)
    summary_max_pages = int(summary.get("max_pages", 0) or 0)

    allow, deny = docs_allowed_for_role(ctx.principal.role)
    search_k = int(req.k)
    tool_cfg = retrieval_cfg.get("tool_calls") if isinstance(retrieval_cfg.get("tool_calls"), dict) else {}
    tool_calls_enabled = bool(tool_cfg.get("enabled", False))
    tool_calls_limit = int(tool_cfg.get("max_calls", 0) or 0)

    def sse(event: str, data: Any) -> bytes:
        if isinstance(data, str):
            payload = data
        else:
            payload = json.dumps(data, ensure_ascii=False)
        return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")

    async def gen() -> Any:
        try:
            yield sse("status", {"phase": "starting", "session_id": session_id})

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

            async def run_retrieval(
                query: str,
                k: int,
                *,
                doc_ids: list[str] | None = None,
                page_ids: list[str] | None = None,
            ) -> list[dict[str, Any]]:
                buffer_k = min(max(k * 5, k + 10), 200)
                async with _search_lock:
                    rows = await engine.search(
                        query,
                        k=buffer_k,
                        mode=retrieval_mode,
                        mmr_enabled=mmr_enabled,
                        mmr_lambda=mmr_lambda,
                        mmr_candidates=mmr_candidates,
                        mmr_use_embeddings=mmr_use_embeddings,
                        expand_neighbors=expand_neighbors,
                        expand_max_chars=expand_max_chars,
                        expand_include_images=expand_include_images,
                        heading_boost=heading_boost,
                        bm25_candidates=bm25_candidates,
                        embedding_candidates=embedding_candidates,
                        rrf_k=rrf_k,
                        bm25_weight=bm25_weight,
                        embedding_weight=embedding_weight,
                        doc_priority=doc_priority_list,
                        doc_priority_boost=doc_priority_boost,
                        summary_enabled=summary_enabled,
                        summary_k=summary_k,
                        summary_max_pages=summary_max_pages,
                        max_per_page=max_per_page,
                        max_per_doc=max_per_doc,
                    )
                rows = [r for r in rows if _doc_allowed(r.get("doc_id"), allow, deny)]
                if doc_ids:
                    allowed_docs = {str(x) for x in doc_ids}
                    rows = [r for r in rows if str(r.get("doc_id") or "") in allowed_docs]
                if page_ids:
                    allowed_pages = {str(x) for x in page_ids}
                    rows = [r for r in rows if str(r.get("page_id") or "") in allowed_pages]
                return rows[:k]

            all_results: list[dict[str, Any]] = []
            seen_results: set[str] = set()
            tool_calls_used = 0
            current_query = req.message
            current_k = search_k
            current_doc_ids: list[str] | None = None
            current_page_ids: list[str] | None = None
            pass_idx = 0
            answer: str | None = None
            citations: list[Citation] = []
            images: list[ImageResult] = []
            context_chunks: list[ContextChunk] = []

            while True:
                pass_idx += 1
                yield sse(
                    "status",
                    {
                        "phase": "retrieving_docs",
                        "message": f"Searching documentation ({retrieval_mode})…",
                        "k": int(current_k),
                        "pass": int(pass_idx),
                    },
                )
                try:
                    retrieval = await run_retrieval(
                        current_query,
                        current_k,
                        doc_ids=current_doc_ids,
                        page_ids=current_page_ids,
                    )
                except ValueError as e:
                    yield sse("error", {"error": "Bad request", "detail": str(e)})
                    return
                except RuntimeError as e:
                    yield sse("error", {"error": "Search unavailable", "detail": str(e)})
                    return

                _merge_retrieval_results(all_results, retrieval, seen_results)
                log.info(
                    "chat stream retrieval pass query=%r k=%s results=%s total=%s",
                    current_query,
                    current_k,
                    len(retrieval),
                    len(all_results),
                )
                yield sse(
                    "status",
                    {
                        "phase": "retrieved_docs",
                        "message": f"Found {len(retrieval)} relevant sections (total {len(all_results)}).",
                        "found": int(len(retrieval)),
                        "total": int(len(all_results)),
                        "pass": int(pass_idx),
                    },
                )

                max_citations = int(retrieval_cfg.get("max_citations", 8))
                max_images = int(retrieval_cfg.get("max_images", 6))
                citations = _build_citations(all_results, DEFAULT_VERSION, max_citations=max_citations)
                images = _build_images(all_results, max_images=max_images)
                context_chunks = _build_context_chunks(all_results, DEFAULT_VERSION)

                retrieval_pack = SearchResponse(
                    chunks=[
                        {
                            "doc_id": r["doc_id"],
                            "page_id": r["page_id"],
                            "heading_path": r["heading_path"],
                            "text": r["text"],
                            "score": r["score"],
                        }
                        for r in all_results
                    ],
                    citations=citations,
                    images=images,
                ).model_dump()
                retrieval_pack["context_chunks"] = [c.model_dump() for c in context_chunks]
                retrieval_pack["meta"] = {
                    "docs_version": DEFAULT_VERSION,
                    "retrieval_mode": retrieval_mode,
                    "k": int(req.k),
                    "web_enabled": bool(web_enabled),
                    "pass": int(pass_idx),
                    "tool_calls_used": int(tool_calls_used),
                }
                if web_sources:
                    retrieval_pack["web_sources"] = [ws.model_dump() for ws in web_sources]
                yield sse("retrieval", retrieval_pack)

                context_text = _build_context_text(all_results, web_context_blocks)

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
                            "tool_call_limit": str(tool_calls_limit),
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
                request = None
                if tool_calls_enabled:
                    yield sse("status", {"phase": "preflight", "message": "Checking for more context…"})
                    try:
                        preflight = await chat_completion(
                            messages=messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout_s=timeout_s,
                            model=llm_model_id,
                        )
                    except LMStudioError as e:
                        log.exception("LM Studio error")
                        yield sse("error", {"error": "LM Studio error", "detail": str(e)})
                        return
                    except Exception as e:
                        log.exception("LM Studio error")
                        yield sse("error", {"error": "LM Studio error", "detail": str(e)})
                        return
                    request = _parse_context_request(preflight)

                if request:
                    tool_calls_used += 1
                    if tool_calls_limit > 0 and tool_calls_used >= tool_calls_limit:
                        log.warning("chat stream tool call limit reached (%s)", tool_calls_limit)
                        yield sse(
                            "error",
                            {"error": "Tool call limit reached", "detail": "Model requested more context too many times."},
                        )
                        return
                    log.info(
                        "chat stream model requested more context reason=%r query=%r k=%s doc_ids=%s page_ids=%s",
                        request.get("reason"),
                        request.get("query"),
                        request.get("k"),
                        request.get("doc_ids"),
                        request.get("page_ids"),
                    )
                    reason = str(request.get("reason") or "").strip()
                    msg = "Model requested more documentation context."
                    if reason:
                        msg = f"{msg} ({reason})"
                    yield sse(
                        "status",
                        {
                            "phase": "model_request",
                            "message": msg,
                            "tool_calls_used": int(tool_calls_used),
                        },
                    )
                    current_query = str(request.get("query") or req.message).strip() or req.message
                    requested_k = int(request.get("k") or 0)
                    if requested_k <= 0:
                        current_k = search_k
                    else:
                        current_k = max(1, min(25, requested_k))
                    current_doc_ids = request.get("doc_ids") or None
                    current_page_ids = request.get("page_ids") or None
                    continue

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
                break

            if answer is None:
                yield sse("error", {"error": "Chat stream failed", "detail": "No answer generated."})
                return
            yield sse("status", {"phase": "saving", "message": "Saving answer to session…"})
            app_db.insert_message(session_id=session_id, role="assistant", content=answer)

            yield sse(
                "final",
                ChatResponse(
                    answer=answer,
                    citations=citations,
                    images=images,
                    context_chunks=context_chunks,
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
