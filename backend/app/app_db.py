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
              password_hash TEXT NOT NULL,
              role TEXT NOT NULL CHECK(role IN ('admin','redactor','client')),
              greeting TEXT,
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

            CREATE INDEX IF NOT EXISTS idx_messages_session_created
              ON messages(session_id, created_at);

            CREATE INDEX IF NOT EXISTS idx_users_role
              ON users(role);
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


def create_user(*, username: str, password_hash: str, role: str, greeting: str | None = None) -> dict[str, Any]:
    user_id = str(uuid.uuid4())
    now = _utc_now()
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO users(user_id, username, password_hash, role, greeting, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            (user_id, username, password_hash, role, greeting, now, now),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "user_id": user_id,
        "username": username,
        "role": role,
        "greeting": greeting,
        "created_at": now,
        "updated_at": now,
    }


def list_users(limit: int = 200) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT user_id, username, role, greeting, created_at, updated_at
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
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT user_id, username, password_hash, role, greeting, created_at, updated_at
            FROM users
            WHERE username = ?
            """,
            (username,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user(user_id: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute(
            """
            SELECT user_id, username, password_hash, role, greeting, created_at, updated_at
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
    role: str | None = None,
    greeting: str | None = None,
    password_hash: str | None = None,
) -> dict[str, Any] | None:
    updates: list[str] = []
    params: list[Any] = []
    if role is not None:
        updates.append("role = ?")
        params.append(role)
    if greeting is not None:
        updates.append("greeting = ?")
        params.append(greeting)
    if password_hash is not None:
        updates.append("password_hash = ?")
        params.append(password_hash)
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
                   u.username, u.role, u.greeting
            FROM auth_sessions s
            JOIN users u ON u.user_id = s.user_id
            WHERE s.token_hash = ?
            """,
            (token_hash,),
        ).fetchone()
        return dict(row) if row else None
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


def delete_expired_auth_sessions(now_iso: str) -> int:
    conn = _connect()
    try:
        cur = conn.execute("DELETE FROM auth_sessions WHERE expires_at < ?", (now_iso,))
        conn.commit()
        return int(cur.rowcount or 0)
    finally:
        conn.close()
