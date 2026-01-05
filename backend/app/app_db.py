from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from .config import APP_DB_PATH
from .logging_utils import get_logger

log = get_logger(__name__)

_UNSET = object()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Path = APP_DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db() -> None:
    conn = _connect()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              session_id TEXT PRIMARY KEY,
              owner_id TEXT,
              title TEXT,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
              message_id TEXT PRIMARY KEY,
              session_id TEXT NOT NULL,
              role TEXT NOT NULL CHECK(role IN ('system','user','assistant')),
              content TEXT NOT NULL,
              created_at TEXT NOT NULL,
              FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS settings (
              key TEXT PRIMARY KEY,
              value_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS users (
              user_id TEXT PRIMARY KEY,
              username TEXT NOT NULL UNIQUE,
              email TEXT UNIQUE,
              password_hash TEXT NOT NULL,
              role TEXT NOT NULL CHECK(role IN ('admin','redactor','support','client')),
              greeting TEXT,
              disabled_at TEXT,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS auth_sessions (
              token_hash TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              created_at TEXT NOT NULL,
              expires_at TEXT NOT NULL,
              last_seen_at TEXT NOT NULL,
              FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS doc_edits (
              edit_id TEXT PRIMARY KEY,
              version TEXT NOT NULL,
              doc_id TEXT NOT NULL,
              page_id TEXT NOT NULL,
              status TEXT NOT NULL CHECK(status IN ('draft','published')),
              content_md TEXT NOT NULL,
              author_id TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS doc_pages (
              version TEXT NOT NULL,
              doc_id TEXT NOT NULL,
              page_id TEXT NOT NULL,
              doc_title TEXT NOT NULL,
              page_title TEXT NOT NULL,
              heading_path_json TEXT NOT NULL,
              source_path TEXT NOT NULL,
              anchor TEXT,
              base_markdown TEXT NOT NULL,
              author_id TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(version, doc_id, page_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session_created
              ON messages(session_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_users_role
              ON users(role);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_doc_edits_key
              ON doc_edits(version, doc_id, page_id, status);

            CREATE INDEX IF NOT EXISTS idx_doc_edits_page
              ON doc_edits(version, doc_id, page_id);

            CREATE INDEX IF NOT EXISTS idx_doc_pages_doc
              ON doc_pages(version, doc_id);
            """
        )
        _migrate_db(conn)
        conn.commit()
    finally:
        conn.close()


def _migrate_db(conn: sqlite3.Connection) -> None:
    # sessions.owner_id was added after the initial schema.
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()]
    if "owner_id" not in cols:
        conn.execute("ALTER TABLE sessions ADD COLUMN owner_id TEXT")
    # Backfill legacy rows.
    conn.execute(
        "UPDATE sessions SET owner_id = COALESCE(NULLIF(owner_id,''), 'legacy') WHERE owner_id IS NULL OR owner_id = ''"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_owner_created ON sessions(owner_id, created_at);")
    _migrate_users(conn)


def _migrate_users(conn: sqlite3.Connection) -> None:
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='users' LIMIT 1"
    ).fetchone()
    if not row:
        return
    users_sql = str(row["sql"] or "")
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
    need_email = "email" not in cols
    need_support_role = "support" not in users_sql
    need_disabled_at = "disabled_at" not in cols
    if not need_email and not need_support_role and not need_disabled_at:
        return
    if not need_email and not need_support_role and need_disabled_at:
        conn.execute("ALTER TABLE users ADD COLUMN disabled_at TEXT")
        return

    conn.execute("PRAGMA foreign_keys=OFF;")
    has_auth = bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='auth_sessions' LIMIT 1"
        ).fetchone()
    )
    if has_auth:
        conn.execute("ALTER TABLE auth_sessions RENAME TO auth_sessions_old;")
    conn.execute("ALTER TABLE users RENAME TO users_old;")

    conn.execute(
        """
        CREATE TABLE users (
          user_id TEXT PRIMARY KEY,
          username TEXT NOT NULL UNIQUE,
          email TEXT UNIQUE,
          password_hash TEXT NOT NULL,
          role TEXT NOT NULL CHECK(role IN ('admin','redactor','support','client')),
          greeting TEXT,
          disabled_at TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        """
    )

    old_cols = [r["name"] for r in conn.execute("PRAGMA table_info(users_old)").fetchall()]
    email_expr = "NULLIF(LOWER(TRIM(email)), '')" if "email" in old_cols else "NULL"
    disabled_expr = "disabled_at" if "disabled_at" in old_cols else "NULL"
    conn.execute(
        f"""
        INSERT INTO users(user_id, username, email, password_hash, role, greeting, disabled_at, created_at, updated_at)
        SELECT user_id,
               username,
               {email_expr},
               password_hash,
               CASE
                 WHEN role IN ('admin','redactor','support','client') THEN role
                 ELSE 'client'
               END AS role,
               greeting,
               {disabled_expr},
               created_at,
               updated_at
        FROM users_old
        """
    )
    conn.execute("DROP TABLE users_old;")

    conn.execute(
        """
        CREATE TABLE auth_sessions (
          token_hash TEXT PRIMARY KEY,
          user_id TEXT NOT NULL,
          created_at TEXT NOT NULL,
          expires_at TEXT NOT NULL,
          last_seen_at TEXT NOT NULL,
          FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
        );
        """
    )
    if has_auth:
        conn.execute(
            """
            INSERT INTO auth_sessions(token_hash, user_id, created_at, expires_at, last_seen_at)
            SELECT token_hash, user_id, created_at, expires_at, last_seen_at
            FROM auth_sessions_old
            """
        )
        conn.execute("DROP TABLE auth_sessions_old;")

    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);")
    conn.execute("PRAGMA foreign_keys=ON;")


def create_session(*, owner_id: str, title: str | None = None) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    created_at = _utc_now()

    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO sessions(session_id, owner_id, title, created_at) VALUES (?,?,?,?)",
            (session_id, owner_id, title, created_at),
        )
        conn.commit()
    finally:
        conn.close()

    return {"session_id": session_id, "owner_id": owner_id, "title": title, "created_at": created_at}


def get_session(session_id: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT session_id, owner_id, title, created_at FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def list_sessions(*, owner_id: str, limit: int = 50) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT s.session_id, s.owner_id, s.title, s.created_at,
                   MAX(m.created_at) AS last_message_at
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.session_id
            WHERE s.owner_id = ?
            GROUP BY s.session_id
            ORDER BY COALESCE(last_message_at, s.created_at) DESC
            LIMIT ?
            """,
            (owner_id, limit),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def insert_message(session_id: str, role: str, content: str) -> dict[str, Any]:
    message_id = str(uuid.uuid4())
    created_at = _utc_now()

    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO messages(message_id, session_id, role, content, created_at) VALUES (?,?,?,?,?)",
            (message_id, session_id, role, content, created_at),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "message_id": message_id,
        "session_id": session_id,
        "role": role,
        "content": content,
        "created_at": created_at,
    }


def list_messages(session_id: str, limit: int = 200) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT message_id, session_id, role, content, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_setting(key: str) -> Any | None:
    conn = _connect()
    try:
        row = conn.execute("SELECT value_json FROM settings WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["value_json"])
        except Exception:
            return row["value_json"]
    finally:
        conn.close()


def list_settings() -> dict[str, Any]:
    conn = _connect()
    try:
        rows = conn.execute("SELECT key, value_json FROM settings ORDER BY key ASC").fetchall()
    finally:
        conn.close()
    out: dict[str, Any] = {}
    for r in rows:
        try:
            out[r["key"]] = json.loads(r["value_json"])
        except Exception:
            out[r["key"]] = r["value_json"]
    return out


def set_setting(key: str, value: Any) -> None:
    now = _utc_now()
    value_json = json.dumps(value, ensure_ascii=False)
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO settings(key, value_json, created_at, updated_at)
            VALUES (?,?,?,?)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
            """,
            (key, value_json, now, now),
        )
        conn.commit()
    finally:
        conn.close()


def set_settings(values: dict[str, Any]) -> None:
    now = _utc_now()
    rows = [(k, json.dumps(v, ensure_ascii=False), now, now) for k, v in values.items()]
    conn = _connect()
    try:
        conn.executemany(
            """
            INSERT INTO settings(key, value_json, created_at, updated_at)
            VALUES (?,?,?,?)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def count_users() -> int:
    conn = _connect()
    try:
        row = conn.execute("SELECT COUNT(1) AS c FROM users").fetchone()
        return int(row["c"] or 0) if row else 0
    finally:
        conn.close()


def create_user(
    *,
    username: str,
    password_hash: str,
    role: str,
    email: str | None = None,
    greeting: str | None = None,
    disabled_at: str | None = None,
) -> dict[str, Any]:
    user_id = str(uuid.uuid4())
    now = _utc_now()
    email = str(email or "").strip().lower() or None
    disabled_at = str(disabled_at or "").strip() or None
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO users(user_id, username, email, password_hash, role, greeting, disabled_at, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (user_id, username, email, password_hash, role, greeting, disabled_at, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "user_id": user_id,
        "username": username,
        "email": email,
        "role": role,
        "greeting": greeting,
        "disabled_at": disabled_at,
        "created_at": now,
        "updated_at": now,
    }


def list_users(limit: int = 200) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT user_id, username, email, role, greeting, disabled_at, created_at, updated_at
            FROM users
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_user_by_username(username: str) -> dict[str, Any] | None:
    username = str(username or "").strip()
    if not username:
        return None
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT user_id, username, email, password_hash, role, greeting, disabled_at, created_at, updated_at
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_email(email: str) -> dict[str, Any] | None:
    email = str(email or "").strip().lower()
    if not email:
        return None
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT user_id, username, email, password_hash, role, greeting, disabled_at, created_at, updated_at
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user(user_id: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT user_id, username, email, password_hash, role, greeting, disabled_at, created_at, updated_at
            FROM users
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_user(
    *,
    user_id: str,
    email: str | None | object = _UNSET,
    role: str | None = None,
    greeting: str | None = None,
    password_hash: str | None = None,
    disabled_at: str | None | object = _UNSET,
) -> dict[str, Any] | None:
    updates: list[str] = []
    params: list[Any] = []
    if email is not _UNSET:
        normalized_email = None
        if email is not None:
            normalized_email = str(email).strip().lower() or None
        updates.append("email = ?")
        params.append(normalized_email)
    if role is not None:
        updates.append("role = ?")
        params.append(role)
    if greeting is not None:
        updates.append("greeting = ?")
        params.append(greeting)
    if password_hash is not None:
        updates.append("password_hash = ?")
        params.append(password_hash)
    if disabled_at is not _UNSET:
        normalized_disabled_at = None
        if disabled_at is not None:
            normalized_disabled_at = str(disabled_at).strip() or None
        updates.append("disabled_at = ?")
        params.append(normalized_disabled_at)
    if not updates:
        return get_user(user_id)

    now = _utc_now()
    updates.append("updated_at = ?")
    params.append(now)
    params.append(user_id)

    conn = _connect()
    try:
        conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE user_id = ?", params)
        conn.commit()
    finally:
        conn.close()
    return get_user(user_id)


def create_auth_session(*, token_hash: str, user_id: str, expires_at: str) -> None:
    now = _utc_now()
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO auth_sessions(token_hash, user_id, created_at, expires_at, last_seen_at)
            VALUES (?,?,?,?,?)
            """,
            (token_hash, user_id, now, expires_at, now),
        )
        conn.commit()
    finally:
        conn.close()


def get_auth_session(token_hash: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT s.token_hash, s.user_id, s.created_at, s.expires_at, s.last_seen_at,
                   u.username, u.email, u.role, u.greeting, u.disabled_at
            FROM auth_sessions s
            JOIN users u ON u.user_id = s.user_id
            WHERE s.token_hash = ?
            """,
            (token_hash,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def count_active_admins() -> int:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT COUNT(1) AS c FROM users WHERE role = 'admin' AND (disabled_at IS NULL OR disabled_at = '')"
        ).fetchone()
        return int(row["c"] or 0) if row else 0
    finally:
        conn.close()


def touch_auth_session(token_hash: str) -> None:
    now = _utc_now()
    conn = _connect()
    try:
        conn.execute("UPDATE auth_sessions SET last_seen_at = ? WHERE token_hash = ?", (now, token_hash))
        conn.commit()
    finally:
        conn.close()


def delete_auth_session(token_hash: str) -> None:
    conn = _connect()
    try:
        conn.execute("DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,))
        conn.commit()
    finally:
        conn.close()


def get_doc_edit(*, version: str, doc_id: str, page_id: str, status: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT edit_id, version, doc_id, page_id, status, content_md, author_id, created_at, updated_at
            FROM doc_edits
            WHERE version = ? AND doc_id = ? AND page_id = ? AND status = ?
            """,
            (str(version), str(doc_id), str(page_id), str(status)),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def upsert_doc_edit(
    *,
    version: str,
    doc_id: str,
    page_id: str,
    status: str,
    content_md: str,
    author_id: str,
) -> dict[str, Any]:
    edit_id = str(uuid.uuid4())
    now = _utc_now()
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO doc_edits(edit_id, version, doc_id, page_id, status, content_md, author_id, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(version, doc_id, page_id, status)
            DO UPDATE SET content_md=excluded.content_md,
                          author_id=excluded.author_id,
                          updated_at=excluded.updated_at
            """,
            (edit_id, version, doc_id, page_id, status, content_md, author_id, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return get_doc_edit(version=version, doc_id=doc_id, page_id=page_id, status=status) or {
        "edit_id": edit_id,
        "version": version,
        "doc_id": doc_id,
        "page_id": page_id,
        "status": status,
        "content_md": content_md,
        "author_id": author_id,
        "created_at": now,
        "updated_at": now,
    }


def delete_doc_edit(*, version: str, doc_id: str, page_id: str, status: str) -> bool:
    conn = _connect()
    try:
        cur = conn.execute(
            "DELETE FROM doc_edits WHERE version = ? AND doc_id = ? AND page_id = ? AND status = ?",
            (str(version), str(doc_id), str(page_id), str(status)),
        )
        conn.commit()
        return bool(cur.rowcount)
    finally:
        conn.close()


def _decode_heading_path(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [str(x) for x in data if str(x).strip()]


def _row_to_doc_page(row: sqlite3.Row) -> dict[str, Any]:
    rec = dict(row)
    rec["heading_path"] = _decode_heading_path(rec.get("heading_path_json"))
    return rec


def list_doc_pages(*, version: str, doc_id: str | None = None) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        if doc_id:
            rows = conn.execute(
                """
                SELECT version, doc_id, page_id, doc_title, page_title, heading_path_json,
                       source_path, anchor, base_markdown, author_id, created_at, updated_at
                FROM doc_pages
                WHERE version = ? AND doc_id = ?
                ORDER BY page_title
                """,
                (str(version), str(doc_id)),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT version, doc_id, page_id, doc_title, page_title, heading_path_json,
                       source_path, anchor, base_markdown, author_id, created_at, updated_at
                FROM doc_pages
                WHERE version = ?
                ORDER BY doc_id, page_title
                """,
                (str(version),),
            ).fetchall()
        return [_row_to_doc_page(r) for r in rows]
    finally:
        conn.close()


def get_doc_page(*, version: str, doc_id: str, page_id: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT version, doc_id, page_id, doc_title, page_title, heading_path_json,
                   source_path, anchor, base_markdown, author_id, created_at, updated_at
            FROM doc_pages
            WHERE version = ? AND doc_id = ? AND page_id = ?
            """,
            (str(version), str(doc_id), str(page_id)),
        ).fetchone()
        return _row_to_doc_page(row) if row else None
    finally:
        conn.close()


def create_doc_page(
    *,
    version: str,
    doc_id: str,
    page_id: str,
    doc_title: str,
    page_title: str,
    heading_path: list[str],
    source_path: str,
    base_markdown: str,
    author_id: str,
    anchor: str | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO doc_pages(
              version, doc_id, page_id, doc_title, page_title, heading_path_json,
              source_path, anchor, base_markdown, author_id, created_at, updated_at
            )
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                str(version),
                str(doc_id),
                str(page_id),
                str(doc_title),
                str(page_title),
                json.dumps(heading_path or []),
                str(source_path),
                str(anchor) if anchor else None,
                str(base_markdown),
                str(author_id),
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return get_doc_page(version=version, doc_id=doc_id, page_id=page_id) or {
        "version": str(version),
        "doc_id": str(doc_id),
        "page_id": str(page_id),
        "doc_title": str(doc_title),
        "page_title": str(page_title),
        "heading_path": list(heading_path or []),
        "source_path": str(source_path),
        "anchor": str(anchor) if anchor else None,
        "base_markdown": str(base_markdown),
        "author_id": str(author_id),
        "created_at": now,
        "updated_at": now,
    }


def delete_expired_auth_sessions(now_iso: str) -> int:
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM auth_sessions WHERE expires_at < ?", (now_iso,))
        conn.commit()
        return int(cur.rowcount or 0)
    finally:
        conn.close()
