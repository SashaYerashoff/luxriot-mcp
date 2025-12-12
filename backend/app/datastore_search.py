from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DATASTORE_DIR, DEFAULT_VERSION
from .logging_utils import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: str
    doc_id: str
    page_id: str
    heading_path: list[str]
    text: str
    source_path: str
    anchor: str | None
    images: list[str]
    length: int


class SearchEngine:
    def __init__(self, version: str = DEFAULT_VERSION, datastore_dir: Path = DATASTORE_DIR) -> None:
        self.version = version
        self.datastore_dir = datastore_dir
        self.index_path = datastore_dir / version / "index.sqlite"
        self._conn: sqlite3.Connection | None = None
        self._meta: dict[str, Any] | None = None

    def is_ready(self) -> bool:
        return self.index_path.exists()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self.index_path.exists():
                raise FileNotFoundError(f"Index not found: {self.index_path}")
            conn = sqlite3.connect(str(self.index_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON;")
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _load_meta(self) -> dict[str, Any]:
        if self._meta is not None:
            return self._meta
        conn = self._connect()
        rows = conn.execute("SELECT key, value FROM meta").fetchall()
        meta = {r["key"]: r["value"] for r in rows}
        self._meta = meta
        return meta

    def _get_stat_floats(self) -> tuple[int, float]:
        meta = self._load_meta()
        try:
            n_chunks = int(meta["n_chunks"])
            avgdl = float(meta["avgdl"])
        except Exception as e:
            raise RuntimeError(f"Invalid meta in index: {e}") from e
        return n_chunks, avgdl

    def _fetch_chunk_rows(self, chunk_ids: list[str]) -> dict[str, ChunkRow]:
        if not chunk_ids:
            return {}
        conn = self._connect()
        qmarks = ",".join(["?"] * len(chunk_ids))
        rows = conn.execute(
            f"""
            SELECT chunk_id, doc_id, page_id, heading_path_json, text, source_path, anchor, images_json, length
            FROM chunks
            WHERE chunk_id IN ({qmarks})
            """,
            chunk_ids,
        ).fetchall()
        out: dict[str, ChunkRow] = {}
        for r in rows:
            out[r["chunk_id"]] = ChunkRow(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                page_id=r["page_id"],
                heading_path=json.loads(r["heading_path_json"]) if r["heading_path_json"] else [],
                text=r["text"],
                source_path=r["source_path"],
                anchor=r["anchor"],
                images=json.loads(r["images_json"]) if r["images_json"] else [],
                length=int(r["length"] or 0),
            )
        return out

    def search(self, query: str, k: int = 8) -> list[dict[str, Any]]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        conn = self._connect()
        n_chunks, avgdl = self._get_stat_floats()

        # BM25 parameters (reasonable defaults).
        k1 = 1.5
        b = 0.75

        # Collect postings for each unique term.
        unique_terms = list(dict.fromkeys(query_terms))
        scores: dict[str, float] = {}
        candidate_chunk_ids: set[str] = set()

        for term in unique_terms:
            df_row = conn.execute("SELECT df FROM terms WHERE term = ?", (term,)).fetchone()
            if not df_row:
                continue
            df = int(df_row["df"])

            # idf variant (BM25+ style shift to avoid negative idf)
            idf = math.log((n_chunks - df + 0.5) / (df + 0.5) + 1.0)

            for post in conn.execute("SELECT chunk_id, tf FROM postings WHERE term = ?", (term,)).fetchall():
                chunk_id = post["chunk_id"]
                tf = int(post["tf"])
                candidate_chunk_ids.add(chunk_id)
                scores.setdefault(chunk_id, 0.0)
                scores[chunk_id] += idf * tf  # length-normalization applied later

        if not candidate_chunk_ids:
            return []

        # Fetch lengths for normalization.
        chunk_rows = self._fetch_chunk_rows(list(candidate_chunk_ids))

        final_scores: dict[str, float] = {}
        for chunk_id, base in scores.items():
            row = chunk_rows.get(chunk_id)
            if not row:
                continue
            dl = max(row.length, 1)
            norm = (k1 * (1 - b + b * (dl / max(avgdl, 1e-9))))
            # Apply a cheap approximation of BM25 (term-weighted tf sum already in base)
            final_scores[chunk_id] = base * ((k1 + 1) / (1 + norm))

        top = sorted(final_scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        results: list[dict[str, Any]] = []
        for chunk_id, score in top:
            row = chunk_rows.get(chunk_id)
            if not row:
                continue
            results.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": row.doc_id,
                    "page_id": row.page_id,
                    "heading_path": row.heading_path,
                    "text": row.text,
                    "score": float(score),
                    "source_path": row.source_path,
                    "anchor": row.anchor,
                    "images": row.images,
                }
            )
        return results

