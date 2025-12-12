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

            CREATE INDEX IF NOT EXISTS idx_messages_session_created
              ON messages(session_id, created_at);
            """
        )
        conn.commit()
    finally:
        conn.close()


def create_session(title: str | None = None) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    created_at = _utc_now()

    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO sessions(session_id, title, created_at) VALUES (?,?,?)",
            (session_id, title, created_at),
        )
        conn.commit()
    finally:
        conn.close()

    return {"session_id": session_id, "title": title, "created_at": created_at}


def get_session(session_id: str) -> dict[str, Any] | None:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT session_id, title, created_at FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def list_sessions(limit: int = 50) -> list[dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT s.session_id, s.title, s.created_at,
                   MAX(m.created_at) AS last_message_at
            FROM sessions s
            LEFT JOIN messages m ON m.session_id = s.session_id
            GROUP BY s.session_id
            ORDER BY COALESCE(last_message_at, s.created_at) DESC
            LIMIT ?
            """,
            (limit,),
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
