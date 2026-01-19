from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import HTTPException, Request, Response

from . import app_db
from .logging_utils import get_logger

log = get_logger(__name__)

Role = Literal["admin", "redactor", "support", "client", "anonymous"]

AUTH_COOKIE = "luxriot_session"
ANON_COOKIE = "luxriot_anon"

_AUTH_SECRET = os.getenv("LUXRIOT_AUTH_SECRET")
if not _AUTH_SECRET:
    _AUTH_SECRET = secrets.token_hex(32)
    log.warning("LUXRIOT_AUTH_SECRET is not set; using ephemeral secret (sessions reset on restart).")

_SESSION_TTL_S = int(os.getenv("LUXRIOT_SESSION_TTL_S", "2592000"))  # 30 days


@dataclass(frozen=True)
class Principal:
    authenticated: bool
    role: Role
    owner_id: str
    user_id: str | None = None
    username: str | None = None
    email: str | None = None
    greeting: str | None = None
    docs_edit: bool = False
    docs_publish: bool = False


@dataclass(frozen=True)
class AuthContext:
    principal: Principal
    set_anon_cookie: str | None = None
    clear_auth_cookie: bool = False


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _token_hash(token: str) -> str:
    return hmac.new(_AUTH_SECRET.encode("utf-8"), token.encode("utf-8"), hashlib.sha256).hexdigest()


def hash_password(password: str, *, iterations: int = 200_000) -> str:
    pwd = str(password or "")
    if not pwd:
        raise ValueError("Password is empty")
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", pwd.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, it_s, salt_hex, hash_hex = str(stored or "").split("$", 3)
        if algo != "pbkdf2_sha256":
            return False
        iterations = int(it_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except Exception:
        return False

    dk = hashlib.pbkdf2_hmac("sha256", str(password or "").encode("utf-8"), salt, iterations)
    return hmac.compare_digest(dk, expected)


def ensure_bootstrap_admin() -> None:
    username = str(os.getenv("LUXRIOT_ADMIN_USERNAME") or "admin").strip()
    password = os.getenv("LUXRIOT_ADMIN_PASSWORD")
    reset = str(os.getenv("LUXRIOT_ADMIN_PASSWORD_RESET") or "").strip().lower() in ("1", "true", "yes", "y")
    generated = False

    if app_db.count_users() == 0:
        if not password:
            password = secrets.token_urlsafe(14)
            generated = True

        try:
            rec = app_db.create_user(
                username=username,
                password_hash=hash_password(password),
                role="admin",
                greeting="Welcome, admin.",
            )
        except Exception as e:
            log.exception("Failed to create bootstrap admin user")
            raise RuntimeError(f"Failed to create bootstrap admin user: {e}") from e

        if generated:
            log.warning(
                "BOOTSTRAP admin user created: username=%s password=%s (set LUXRIOT_ADMIN_PASSWORD to override)",
                rec["username"],
                password,
            )
        else:
            log.info("Bootstrap admin user created: username=%s (password from env)", rec["username"])
        return

    # Existing DB: optionally reset/create admin via env var.
    if not reset:
        if password:
            exists = app_db.get_user_by_username(username)
            if exists:
                log.warning(
                    "Admin user '%s' already exists; ignoring LUXRIOT_ADMIN_PASSWORD. "
                    "To reset it, set LUXRIOT_ADMIN_PASSWORD_RESET=1 for one restart.",
                    username,
                )
            else:
                log.warning(
                    "Users already exist; not creating admin user '%s'. "
                    "To create it, set LUXRIOT_ADMIN_PASSWORD_RESET=1 (and LUXRIOT_ADMIN_PASSWORD) for one restart.",
                    username,
                )
        return
    if not password:
        log.warning("LUXRIOT_ADMIN_PASSWORD_RESET is set but LUXRIOT_ADMIN_PASSWORD is empty; skipping.")
        return

    user = app_db.get_user_by_username(username)
    password_hash = hash_password(password)
    if user:
        try:
            app_db.update_user(user_id=str(user["user_id"]), role="admin", password_hash=password_hash)
        except Exception as e:
            log.exception("Failed to reset admin password")
            raise RuntimeError(f"Failed to reset admin password: {e}") from e
        log.warning(
            "Admin password reset via env for username=%s (remove LUXRIOT_ADMIN_PASSWORD_RESET after first run).",
            username,
        )
    else:
        try:
            app_db.create_user(
                username=username,
                password_hash=password_hash,
                role="admin",
                greeting="Welcome, admin.",
            )
        except Exception as e:
            log.exception("Failed to create admin user")
            raise RuntimeError(f"Failed to create admin user: {e}") from e
        log.warning(
            "Admin user created via env: username=%s (remove LUXRIOT_ADMIN_PASSWORD_RESET after first run).",
            username,
        )


def resolve_auth(request: Request) -> AuthContext:
    # Clean up expired sessions opportunistically.
    try:
        app_db.delete_expired_auth_sessions(_utc_now().isoformat())
    except Exception:
        pass

    token = request.cookies.get(AUTH_COOKIE)
    if token:
        th = _token_hash(token)
        sess = app_db.get_auth_session(th)
        if not sess:
            anon_principal, anon_cookie = _anonymous_principal(request, new_cookie=True)
            return AuthContext(
                principal=anon_principal,
                set_anon_cookie=anon_cookie,
                clear_auth_cookie=True,
            )
        try:
            expires_at = datetime.fromisoformat(str(sess.get("expires_at") or ""))
        except Exception:
            expires_at = _utc_now() - timedelta(seconds=1)
        if expires_at <= _utc_now():
            try:
                app_db.delete_auth_session(th)
            except Exception:
                pass
            anon_principal, anon_cookie = _anonymous_principal(request, new_cookie=True)
            return AuthContext(
                principal=anon_principal,
                set_anon_cookie=anon_cookie,
                clear_auth_cookie=True,
            )
        try:
            app_db.touch_auth_session(th)
        except Exception:
            pass
        if str(sess.get("disabled_at") or "").strip():
            try:
                app_db.delete_auth_session(th)
            except Exception:
                pass
            anon_principal, anon_cookie = _anonymous_principal(request, new_cookie=True)
            return AuthContext(
                principal=anon_principal,
                set_anon_cookie=anon_cookie,
                clear_auth_cookie=True,
            )

        role = str(sess.get("role") or "client")  # type: ignore[arg-type]
        if str(sess.get("docs_edit") or "").strip():
            docs_edit = bool(int(sess.get("docs_edit") or 0))
        else:
            docs_edit = role in ("admin", "redactor", "support")
        if str(sess.get("docs_publish") or "").strip():
            docs_publish = bool(int(sess.get("docs_publish") or 0))
        else:
            docs_publish = role == "admin"
        if role == "admin":
            docs_edit = True
            docs_publish = True
        principal = Principal(
            authenticated=True,
            role=role,
            owner_id=str(sess.get("user_id") or ""),
            user_id=str(sess.get("user_id") or ""),
            username=str(sess.get("username") or ""),
            email=str(sess.get("email") or "") or None,
            greeting=str(sess.get("greeting") or "") or None,
            docs_edit=docs_edit,
            docs_publish=docs_publish,
        )
        return AuthContext(principal=principal)

    principal, anon_cookie = _anonymous_principal(request, new_cookie=True)
    return AuthContext(principal=principal, set_anon_cookie=anon_cookie)


def _anonymous_principal(request: Request, *, new_cookie: bool) -> tuple[Principal, str | None]:
    anon_id = str(request.cookies.get(ANON_COOKIE) or "").strip()
    created = None
    if not anon_id and new_cookie:
        anon_id = secrets.token_hex(16)
        created = anon_id
    owner_id = f"anon:{anon_id or 'missing'}"
    principal = Principal(authenticated=False, role="anonymous", owner_id=owner_id, greeting="Welcome.")
    return principal, created


def apply_auth_cookies(response: Response, ctx: AuthContext) -> None:
    if ctx.clear_auth_cookie:
        response.delete_cookie(AUTH_COOKIE)
    if ctx.set_anon_cookie:
        response.set_cookie(
            ANON_COOKIE,
            ctx.set_anon_cookie,
            httponly=True,
            samesite="lax",
            secure=False,
            max_age=60 * 60 * 24 * 365,
        )


def require_role(ctx: AuthContext, allowed: set[Role]) -> None:
    if ctx.principal.role not in allowed:
        raise HTTPException(status_code=403, detail="Forbidden")


def create_login_session(*, username: str, password: str) -> tuple[Principal, str]:
    ident = str(username or "").strip()
    rec = app_db.get_user_by_username(ident)
    if not rec and "@" in ident:
        rec = app_db.get_user_by_email(ident)
    if not rec:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not verify_password(password, str(rec.get("password_hash") or "")):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if str(rec.get("disabled_at") or "").strip():
        raise HTTPException(status_code=403, detail="User is disabled")

    token = secrets.token_urlsafe(32)
    th = _token_hash(token)
    expires_at = (_utc_now() + timedelta(seconds=_SESSION_TTL_S)).isoformat()
    try:
        app_db.create_auth_session(token_hash=th, user_id=str(rec["user_id"]), expires_at=expires_at)
    except Exception as e:
        log.exception("Failed to create auth session")
        raise HTTPException(status_code=500, detail=f"Failed to create auth session: {e}") from e

    role = str(rec.get("role") or "client")  # type: ignore[arg-type]
    if str(rec.get("docs_edit") or "").strip():
        docs_edit = bool(int(rec.get("docs_edit") or 0))
    else:
        docs_edit = role in ("admin", "redactor", "support")
    if str(rec.get("docs_publish") or "").strip():
        docs_publish = bool(int(rec.get("docs_publish") or 0))
    else:
        docs_publish = role == "admin"
    if role == "admin":
        docs_edit = True
        docs_publish = True
    principal = Principal(
        authenticated=True,
        role=role,
        owner_id=str(rec["user_id"]),
        user_id=str(rec["user_id"]),
        username=str(rec["username"]),
        email=str(rec.get("email") or "") or None,
        greeting=str(rec.get("greeting") or "") or None,
        docs_edit=docs_edit,
        docs_publish=docs_publish,
    )
    return principal, token


def logout_session(request: Request) -> None:
    token = request.cookies.get(AUTH_COOKIE)
    if not token:
        return
    try:
        app_db.delete_auth_session(_token_hash(token))
    except Exception:
        pass


def docs_allowed_for_role(role: Role) -> tuple[set[str] | None, set[str]]:
    # allowlist=None means "all docs allowed"
    if role in ("admin", "redactor", "support"):
        return None, set()
    # client + anonymous: exclude API docs by default
    deny = {"luxriot-evo-monitor-http-api"}
    return None, deny
