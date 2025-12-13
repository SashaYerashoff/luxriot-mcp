from __future__ import annotations

import json
import math
import re
import sqlite3
from array import array
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DATASTORE_DIR, DEFAULT_VERSION
from .logging_utils import get_logger
from .lmstudio import LMStudioError, embeddings as lm_embeddings

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "new",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
    "you",
    # Product boilerplate terms (present in most headings).
    "luxriot",
    "evo",
}


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
        self._embedding_vectors: dict[str, array] | None = None
        self._embedding_dim: int | None = None
        self._embedding_model_id: str | None = None

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
        self._meta = None
        self._embedding_vectors = None
        self._embedding_dim = None
        self._embedding_model_id = None

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

    def embeddings_ready(self) -> bool:
        try:
            meta = self._load_meta()
            enabled = str(meta.get("embeddings_enabled", "0"))
            dim = int(meta.get("embedding_dim", "0") or 0)
            model_id = str(meta.get("embedding_model_id", "") or "")
        except Exception:
            return False

        if enabled != "1" or dim <= 0 or not model_id:
            return False

        conn = self._connect()
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='embeddings' LIMIT 1"
        ).fetchone()
        return bool(row)

    def _load_embeddings(self) -> None:
        if self._embedding_vectors is not None:
            return
        if not self.embeddings_ready():
            raise RuntimeError(
                "Embeddings are not available for this datastore. Re-run ingestion without --no-embeddings."
            )

        meta = self._load_meta()
        model_id = str(meta.get("embedding_model_id", "") or "")
        dim = int(meta.get("embedding_dim", "0") or 0)
        if not model_id or dim <= 0:
            raise RuntimeError("Embedding metadata missing in index; re-run ingestion.")

        conn = self._connect()
        rows = conn.execute("SELECT chunk_id, dim, vector FROM embeddings").fetchall()
        if not rows:
            raise RuntimeError("Embeddings table is empty; re-run ingestion.")

        vectors: dict[str, array] = {}
        for r in rows:
            cid = str(r["chunk_id"])
            rdim = int(r["dim"])
            if rdim != dim:
                raise RuntimeError("Embedding dimension mismatch in DB.")
            buf = r["vector"]
            if not isinstance(buf, (bytes, bytearray, memoryview)):
                raise RuntimeError("Invalid embedding vector type in DB.")
            a = array("f")
            a.frombytes(bytes(buf))
            if len(a) != dim:
                raise RuntimeError("Embedding vector length mismatch in DB.")
            vectors[cid] = a

        self._embedding_vectors = vectors
        self._embedding_dim = dim
        self._embedding_model_id = model_id

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

    def _bm25_rank(self, query: str, k: int) -> tuple[list[tuple[str, float]], dict[str, ChunkRow]]:
        query_terms = tokenize(query)
        if not query_terms:
            return ([], {})

        conn = self._connect()
        n_chunks, avgdl = self._get_stat_floats()

        # BM25 parameters (reasonable defaults).
        k1 = 1.5
        b = 0.75

        unique_terms = list(dict.fromkeys(query_terms))

        # Gather postings for each term so we only hit SQLite once per term.
        postings_by_term: dict[str, list[tuple[str, int]]] = {}
        candidate_chunk_ids: set[str] = set()
        idf_by_term: dict[str, float] = {}

        for term in unique_terms:
            df_row = conn.execute("SELECT df FROM terms WHERE term = ?", (term,)).fetchone()
            if not df_row:
                continue
            df = int(df_row["df"])
            idf_by_term[term] = math.log((n_chunks - df + 0.5) / (df + 0.5) + 1.0)

            posts = conn.execute("SELECT chunk_id, tf FROM postings WHERE term = ?", (term,)).fetchall()
            if not posts:
                continue
            lst: list[tuple[str, int]] = []
            for post in posts:
                chunk_id = str(post["chunk_id"])
                tf = int(post["tf"])
                candidate_chunk_ids.add(chunk_id)
                lst.append((chunk_id, tf))
            postings_by_term[term] = lst

        if not candidate_chunk_ids:
            return ([], {})

        chunk_rows = self._fetch_chunk_rows(list(candidate_chunk_ids))
        scores: dict[str, float] = {}
        for term, posts in postings_by_term.items():
            idf = idf_by_term.get(term)
            if idf is None:
                continue
            for chunk_id, tf in posts:
                row = chunk_rows.get(chunk_id)
                if not row:
                    continue
                dl = max(row.length, 1)
                denom = tf + k1 * (1.0 - b + b * (dl / max(avgdl, 1e-9)))
                score = idf * (tf * (k1 + 1.0) / denom)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        top = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return top, chunk_rows

    async def _embedding_rank(self, query: str, k: int) -> list[tuple[str, float]]:
        self._load_embeddings()
        assert self._embedding_vectors is not None
        assert self._embedding_dim is not None
        assert self._embedding_model_id is not None

        try:
            q_vec = (await lm_embeddings([query], model=self._embedding_model_id))[0]
        except LMStudioError as e:
            raise RuntimeError(str(e)) from e

        norm = math.sqrt(sum(x * x for x in q_vec)) or 1.0
        q = array("f", (float(x / norm) for x in q_vec))
        if len(q) != self._embedding_dim:
            raise RuntimeError("Query embedding dimension mismatch; check embedding model and datastore.")

        scores: list[tuple[str, float]] = []
        dim = self._embedding_dim
        for cid, v in self._embedding_vectors.items():
            s = 0.0
            for i in range(dim):
                s += q[i] * v[i]
            scores.append((cid, float(s)))

        scores.sort(key=lambda kv: (-kv[1], kv[0]))
        return scores[:k]

    def _heading_match_multiplier(self, heading_path: list[str], query_terms: set[str], boost: float) -> float:
        if boost <= 0.0 or not query_terms:
            return 1.0
        hay = " ".join(heading_path or [])
        heading_terms = set(tokenize(hay))
        if not heading_terms:
            return 1.0
        overlap = query_terms.intersection(heading_terms)
        if not overlap:
            return 1.0
        frac = float(len(overlap)) / float(len(query_terms) or 1)
        return 1.0 + float(boost) * frac

    def _doc_priority_multiplier(self, doc_id: str, priority: list[str], boost: float, prio_map: dict[str, int]) -> float:
        if boost <= 0.0:
            return 1.0
        idx = prio_map.get(doc_id)
        if idx is None:
            return 1.0
        if len(priority) <= 1:
            rank_factor = 1.0
        else:
            rank_factor = 1.0 - (idx / (len(priority) - 1))
        return 1.0 + boost * rank_factor

    def _apply_dedupe(
        self,
        ranked: list[tuple[str, float]],
        chunk_rows: dict[str, ChunkRow],
        k: int,
        max_per_page: int,
        max_per_doc: int,
    ) -> list[tuple[str, float]]:
        if max_per_page <= 0 and max_per_doc <= 0:
            return ranked[:k]
        page_counts: Counter[tuple[str, str]] = Counter()
        doc_counts: Counter[str] = Counter()
        out: list[tuple[str, float]] = []
        for cid, score in ranked:
            row = chunk_rows.get(cid)
            if not row:
                continue
            page_key = (row.doc_id, row.page_id)
            if max_per_page > 0 and page_counts[page_key] >= max_per_page:
                continue
            if max_per_doc > 0 and doc_counts[row.doc_id] >= max_per_doc:
                continue
            page_counts[page_key] += 1
            doc_counts[row.doc_id] += 1
            out.append((cid, score))
            if len(out) >= k:
                break
        return out

    def _dot(self, a: array, b: array) -> float:
        if len(a) != len(b):
            return 0.0
        s = 0.0
        for i in range(len(a)):
            s += float(a[i]) * float(b[i])
        return float(s)

    def _jaccard_similarity(self, a: set[str], b: set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = a.intersection(b)
        if not inter:
            return 0.0
        union = a.union(b)
        return float(len(inter)) / float(len(union) or 1)

    def _mmr_select(
        self,
        ranked: list[tuple[str, float]],
        chunk_rows: dict[str, ChunkRow],
        *,
        k: int,
        mmr_lambda: float,
        use_embeddings: bool,
        max_per_page: int,
        max_per_doc: int,
        trace_out: list[dict[str, Any]] | None = None,
    ) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        if not ranked:
            return []

        # Clamp lambda to [0, 1]. Higher values favor relevance; lower values favor diversity.
        lam = float(mmr_lambda)
        if lam < 0.0:
            lam = 0.0
        if lam > 1.0:
            lam = 1.0

        # Pre-filter to candidates we can actually return.
        candidates: list[tuple[str, float]] = [(cid, float(score)) for cid, score in ranked if cid in chunk_rows]
        if not candidates:
            return []

        # Normalize relevance to [0, 1] using min-max scores for stable MMR arithmetic.
        scores = [score for _, score in candidates]
        min_score = min(scores)
        max_score = max(scores)
        if max_score > min_score:
            rel = {cid: (score - min_score) / (max_score - min_score) for cid, score in candidates}
        else:
            rel = {cid: 1.0 for cid, _ in candidates}

        use_emb = bool(use_embeddings) and self.embeddings_ready()
        cand_vec: dict[str, array] = {}
        cand_tokens: dict[str, set[str]] = {}
        if use_emb:
            self._load_embeddings()
            assert self._embedding_vectors is not None
            for cid, _ in candidates:
                v = self._embedding_vectors.get(cid)
                if v is not None:
                    cand_vec[cid] = v

        if not use_emb or len(cand_vec) < len(candidates):
            for cid, _ in candidates:
                if cid in cand_tokens:
                    continue
                row = chunk_rows.get(cid)
                if not row:
                    continue
                toks = tokenize(row.text)
                # Bound token-set size to keep it cheap and stable.
                cand_tokens[cid] = set(toks[:256])

        selected: list[tuple[str, float]] = []
        selected_ids: list[str] = []
        page_counts: Counter[tuple[str, str]] = Counter()
        doc_counts: Counter[str] = Counter()

        def allowed(cid: str) -> bool:
            row = chunk_rows.get(cid)
            if not row:
                return False
            page_key = (row.doc_id, row.page_id)
            if max_per_page > 0 and page_counts[page_key] >= max_per_page:
                return False
            if max_per_doc > 0 and doc_counts[row.doc_id] >= max_per_doc:
                return False
            return True

        def update_counts(cid: str) -> None:
            row = chunk_rows.get(cid)
            if not row:
                return
            page_counts[(row.doc_id, row.page_id)] += 1
            doc_counts[row.doc_id] += 1

        def similarity(a_id: str, b_id: str) -> float:
            if a_id == b_id:
                return 1.0
            if use_emb and a_id in cand_vec and b_id in cand_vec:
                s = self._dot(cand_vec[a_id], cand_vec[b_id])
                if s < 0.0:
                    s = 0.0
                if s > 1.0:
                    s = 1.0
                return float(s)
            a_toks = cand_tokens.get(a_id) or set()
            b_toks = cand_tokens.get(b_id) or set()
            return self._jaccard_similarity(a_toks, b_toks)

        # Deterministic MMR selection.
        while len(selected) < k:
            best_cid: str | None = None
            best_mmr: float = -1e18

            for cid, orig_score in candidates:
                if cid in selected_ids:
                    continue
                if not allowed(cid):
                    continue

                relevance = float(rel.get(cid, 0.0))
                if not selected_ids:
                    mmr_score = relevance
                else:
                    max_sim = 0.0
                    for sid in selected_ids:
                        sim = similarity(cid, sid)
                        if sim > max_sim:
                            max_sim = sim
                    mmr_score = lam * relevance - (1.0 - lam) * max_sim

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_cid = cid
                    best_orig_score = orig_score
                    best_relevance = relevance
                    best_max_sim = float(max_sim) if selected_ids else 0.0

            if best_cid is None:
                break
            selected.append((best_cid, float(best_orig_score)))
            selected_ids.append(best_cid)
            update_counts(best_cid)
            if trace_out is not None:
                trace_out.append(
                    {
                        "step": len(selected_ids),
                        "chunk_id": best_cid,
                        "relevance": float(best_relevance),
                        "max_similarity": float(best_max_sim),
                        "mmr_score": float(best_mmr),
                    }
                )

        return selected

    def _neighbor_chunk_ids(self, chunk_id: str, neighbors: int) -> list[str]:
        if neighbors <= 0:
            return [chunk_id]
        try:
            doc_part, page_part, idx_part = chunk_id.rsplit(":", 2)
            idx = int(idx_part)
        except Exception:
            return [chunk_id]
        start = max(0, idx - neighbors)
        end = idx + neighbors
        return [f"{doc_part}:{page_part}:{i:03d}" for i in range(start, end + 1)]

    def _expand_chunk_text(
        self,
        chunk_id: str,
        chunk_rows: dict[str, ChunkRow],
        *,
        neighbors: int,
        max_chars: int,
    ) -> tuple[str, list[str]]:
        if neighbors <= 0:
            row = chunk_rows.get(chunk_id)
            return (row.text, row.images) if row else ("", [])

        wanted = self._neighbor_chunk_ids(chunk_id, neighbors)
        missing = [cid for cid in wanted if cid not in chunk_rows]
        if missing:
            fetched = self._fetch_chunk_rows(missing)
            chunk_rows.update(fetched)

        segments: list[tuple[str, str, list[str]]] = []
        for cid in wanted:
            row = chunk_rows.get(cid)
            if not row:
                continue
            segments.append((cid, row.text, row.images))

        if not segments:
            return ("", [])

        center_idx = 0
        for i, (cid, _, _) in enumerate(segments):
            if cid == chunk_id:
                center_idx = i
                break

        # Always include the center segment; expand outward until we hit max_chars.
        include: set[int] = {center_idx}
        text_total = len(segments[center_idx][1])
        if max_chars <= 0:
            include = set(range(len(segments)))
        else:
            remaining = max_chars - text_total
            if remaining <= 0:
                text = segments[center_idx][1][:max_chars]
                images = segments[center_idx][2]
                return text, images

            left = center_idx - 1
            right = center_idx + 1
            while remaining > 0 and (left >= 0 or right < len(segments)):
                progressed = False
                if left >= 0:
                    seg_len = len(segments[left][1]) + 2
                    if seg_len <= remaining:
                        include.add(left)
                        remaining -= seg_len
                        progressed = True
                    left -= 1
                if right < len(segments):
                    seg_len = len(segments[right][1]) + 2
                    if seg_len <= remaining:
                        include.add(right)
                        remaining -= seg_len
                        progressed = True
                    right += 1
                if not progressed:
                    break

        texts: list[str] = []
        images_out: list[str] = []
        seen_img: set[str] = set()
        for i in sorted(include):
            t = segments[i][1].strip()
            if t:
                texts.append(t)
            for url in segments[i][2]:
                if url in seen_img:
                    continue
                seen_img.add(url)
                images_out.append(url)

        return "\n\n".join(texts).strip(), images_out

    def _rows_to_results(
        self,
        selected: list[tuple[str, float]],
        chunk_rows: dict[str, ChunkRow],
        *,
        expand_neighbors: int,
        expand_max_chars: int,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for cid, score in selected:
            row = chunk_rows.get(cid)
            if not row:
                continue
            if expand_neighbors > 0:
                text, images = self._expand_chunk_text(
                    cid,
                    chunk_rows,
                    neighbors=expand_neighbors,
                    max_chars=expand_max_chars,
                )
            else:
                text, images = row.text, row.images
            out.append(
                {
                    "chunk_id": cid,
                    "doc_id": row.doc_id,
                    "page_id": row.page_id,
                    "heading_path": row.heading_path,
                    "text": text,
                    "score": float(score),
                    "source_path": row.source_path,
                    "anchor": row.anchor,
                    "images": images,
                }
            )
        return out

    async def search(
        self,
        query: str,
        k: int = 8,
        mode: str = "bm25",
        *,
        mmr_enabled: bool = False,
        mmr_lambda: float = 0.7,
        mmr_candidates: int | None = None,
        mmr_use_embeddings: bool = True,
        expand_neighbors: int = 0,
        expand_max_chars: int = 0,
        heading_boost: float = 0.0,
        bm25_candidates: int | None = None,
        embedding_candidates: int | None = None,
        rrf_k: int = 60,
        bm25_weight: float = 1.0,
        embedding_weight: float = 1.0,
        doc_priority: list[str] | None = None,
        doc_priority_boost: float = 0.0,
        max_per_page: int = 0,
        max_per_doc: int = 0,
        debug_out: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        mode = (mode or "bm25").lower().strip()
        if mode not in ("bm25", "embedding", "hybrid"):
            raise ValueError(f"Unknown retrieval mode: {mode}")

        doc_priority = doc_priority or []
        prio_map = {d: i for i, d in enumerate(doc_priority)}
        mmr_enabled = bool(mmr_enabled)
        use_mmr = mmr_enabled and k > 1
        cand_limit = int(mmr_candidates or 0) or 0
        if cand_limit > 0:
            cand_limit = max(cand_limit, k)
        heading_boost = float(heading_boost or 0.0)
        query_terms_for_heading = {t for t in tokenize(query) if t not in _STOPWORDS}
        want_debug = debug_out is not None
        if want_debug:
            debug_out.clear()
            debug_out.update(
                {
                    "query": query,
                    "mode": mode,
                    "k": int(k),
                    "doc_priority_boost": float(doc_priority_boost or 0.0),
                    "heading_boost": float(heading_boost or 0.0),
                    "doc_priority": list(doc_priority),
                    "dedupe": {"max_per_page": int(max_per_page or 0), "max_per_doc": int(max_per_doc or 0)},
                    "mmr": {
                        "enabled": bool(mmr_enabled),
                        "lambda": float(mmr_lambda),
                        "candidates": int(cand_limit or 0),
                        "use_embeddings": bool(mmr_use_embeddings),
                    },
                    "expand": {"neighbors": int(expand_neighbors or 0), "max_chars": int(expand_max_chars or 0)},
                }
            )

        if mode == "bm25":
            top, chunk_rows = self._bm25_rank(query, cand_limit if use_mmr and cand_limit else k)
            adjusted: list[tuple[str, float]] = []
            cand_debug: dict[str, dict[str, Any]] = {}
            for rank, (cid, bm25_score) in enumerate(top, start=1):
                row = chunk_rows.get(cid)
                if not row:
                    continue
                doc_mult = self._doc_priority_multiplier(row.doc_id, doc_priority, doc_priority_boost, prio_map)
                heading_mult = self._heading_match_multiplier(row.heading_path, query_terms_for_heading, heading_boost)
                score = float(bm25_score) * float(doc_mult) * float(heading_mult)
                adjusted.append((cid, float(score)))
                if want_debug:
                    cand_debug[cid] = {
                        "chunk_id": cid,
                        "doc_id": row.doc_id,
                        "page_id": row.page_id,
                        "heading_path": row.heading_path,
                        "score": float(score),
                        "bm25": {"rank": int(rank), "score": float(bm25_score)},
                        "doc_priority_mult": float(doc_mult),
                        "heading_mult": float(heading_mult),
                    }
            adjusted.sort(key=lambda kv: (-kv[1], kv[0]))
            mmr_trace: list[dict[str, Any]] = []
            if use_mmr:
                selected = self._mmr_select(
                    adjusted,
                    chunk_rows,
                    k=k,
                    mmr_lambda=mmr_lambda,
                    use_embeddings=mmr_use_embeddings,
                    max_per_page=max_per_page,
                    max_per_doc=max_per_doc,
                    trace_out=mmr_trace if want_debug else None,
                )
            else:
                selected = self._apply_dedupe(adjusted, chunk_rows, k, max_per_page, max_per_doc)
            results = self._rows_to_results(
                selected,
                chunk_rows,
                expand_neighbors=int(expand_neighbors or 0),
                expand_max_chars=int(expand_max_chars or 0),
            )
            if want_debug:
                debug_out["candidates_count"] = int(len(adjusted))
                debug_out["candidates_top"] = [cand_debug[cid] for cid, _ in adjusted[: min(50, len(adjusted))] if cid in cand_debug]
                debug_out["mmr_trace"] = mmr_trace
                debug_out["selected"] = []
                for i, r in enumerate(results, start=1):
                    info = dict(cand_debug.get(r["chunk_id"], {}))
                    info.update(
                        {
                            "rank": int(i),
                            "returned_score": float(r.get("score", 0.0)),
                            "text_chars": int(len(r.get("text") or "")),
                            "images_count": int(len(r.get("images") or [])),
                        }
                    )
                    debug_out["selected"].append(info)
            return results

        if mode == "embedding":
            if not self.embeddings_ready():
                raise RuntimeError(
                    "Embeddings mode requested but embeddings are not available. Re-run ingestion to build embeddings."
                )
            top = await self._embedding_rank(query, k=max((cand_limit if use_mmr and cand_limit else k), 1))
            chunk_ids = [cid for cid, _ in top]
            chunk_rows = self._fetch_chunk_rows(chunk_ids)
            adjusted: list[tuple[str, float]] = []
            cand_debug: dict[str, dict[str, Any]] = {}
            for rank, (cid, emb_score) in enumerate(top, start=1):
                row = chunk_rows.get(cid)
                if not row:
                    continue
                doc_mult = self._doc_priority_multiplier(row.doc_id, doc_priority, doc_priority_boost, prio_map)
                heading_mult = self._heading_match_multiplier(row.heading_path, query_terms_for_heading, heading_boost)
                score = float(emb_score) * float(doc_mult) * float(heading_mult)
                adjusted.append((cid, float(score)))
                if want_debug:
                    cand_debug[cid] = {
                        "chunk_id": cid,
                        "doc_id": row.doc_id,
                        "page_id": row.page_id,
                        "heading_path": row.heading_path,
                        "score": float(score),
                        "embedding": {"rank": int(rank), "score": float(emb_score)},
                        "doc_priority_mult": float(doc_mult),
                        "heading_mult": float(heading_mult),
                    }
            adjusted.sort(key=lambda kv: (-kv[1], kv[0]))
            mmr_trace: list[dict[str, Any]] = []
            if use_mmr:
                selected = self._mmr_select(
                    adjusted,
                    chunk_rows,
                    k=k,
                    mmr_lambda=mmr_lambda,
                    use_embeddings=mmr_use_embeddings,
                    max_per_page=max_per_page,
                    max_per_doc=max_per_doc,
                    trace_out=mmr_trace if want_debug else None,
                )
            else:
                selected = self._apply_dedupe(adjusted, chunk_rows, k, max_per_page, max_per_doc)
            results = self._rows_to_results(
                selected,
                chunk_rows,
                expand_neighbors=int(expand_neighbors or 0),
                expand_max_chars=int(expand_max_chars or 0),
            )
            if want_debug:
                debug_out["candidates_count"] = int(len(adjusted))
                debug_out["candidates_top"] = [cand_debug[cid] for cid, _ in adjusted[: min(50, len(adjusted))] if cid in cand_debug]
                debug_out["mmr_trace"] = mmr_trace
                debug_out["selected"] = []
                for i, r in enumerate(results, start=1):
                    info = dict(cand_debug.get(r["chunk_id"], {}))
                    info.update(
                        {
                            "rank": int(i),
                            "returned_score": float(r.get("score", 0.0)),
                            "text_chars": int(len(r.get("text") or "")),
                            "images_count": int(len(r.get("images") or [])),
                        }
                    )
                    debug_out["selected"].append(info)
            return results

        # hybrid
        if not self.embeddings_ready():
            raise RuntimeError(
                "Hybrid mode requested but embeddings are not available. Re-run ingestion to build embeddings."
            )

        bm25_candidates = int(bm25_candidates or max(50, k))
        embedding_candidates = int(embedding_candidates or max(50, k))
        if want_debug:
            debug_out["hybrid"] = {
                "rrf_k": int(rrf_k),
                "bm25_weight": float(bm25_weight),
                "embedding_weight": float(embedding_weight),
                "bm25_candidates": int(bm25_candidates),
                "embedding_candidates": int(embedding_candidates),
            }

        bm25_top, _bm25_rows = self._bm25_rank(query, bm25_candidates)
        emb_top = await self._embedding_rank(query, embedding_candidates)

        bm25_rank = {cid: i + 1 for i, (cid, _) in enumerate(bm25_top)}
        emb_rank = {cid: i + 1 for i, (cid, _) in enumerate(emb_top)}
        bm25_score = {cid: float(score) for cid, score in bm25_top}
        emb_score = {cid: float(score) for cid, score in emb_top}

        candidate_ids = set(bm25_rank.keys()) | set(emb_rank.keys())
        chunk_rows = self._fetch_chunk_rows(list(candidate_ids))

        combined: list[tuple[str, float]] = []
        cand_debug: dict[str, dict[str, Any]] = {}
        for cid in candidate_ids:
            base = 0.0
            r_b = bm25_rank.get(cid)
            r_e = emb_rank.get(cid)
            if r_b is not None:
                base += float(bm25_weight) / float(rrf_k + r_b)
            if r_e is not None:
                base += float(embedding_weight) / float(rrf_k + r_e)
            row = chunk_rows.get(cid)
            doc_mult = 1.0
            heading_mult = 1.0
            if row:
                doc_mult = self._doc_priority_multiplier(row.doc_id, doc_priority, doc_priority_boost, prio_map)
                heading_mult = self._heading_match_multiplier(row.heading_path, query_terms_for_heading, heading_boost)
            score = float(base) * float(doc_mult) * float(heading_mult)
            combined.append((cid, float(score)))
            if want_debug and row:
                cand_debug[cid] = {
                    "chunk_id": cid,
                    "doc_id": row.doc_id,
                    "page_id": row.page_id,
                    "heading_path": row.heading_path,
                    "score": float(score),
                    "rrf_base": float(base),
                    "bm25": {"rank": int(r_b) if r_b is not None else None, "score": bm25_score.get(cid)},
                    "embedding": {"rank": int(r_e) if r_e is not None else None, "score": emb_score.get(cid)},
                    "doc_priority_mult": float(doc_mult),
                    "heading_mult": float(heading_mult),
                }

        combined.sort(key=lambda kv: (-kv[1], kv[0]))
        mmr_trace: list[dict[str, Any]] = []
        if use_mmr:
            trimmed = combined[: (cand_limit if cand_limit else len(combined))]
            selected = self._mmr_select(
                trimmed,
                chunk_rows,
                k=k,
                mmr_lambda=mmr_lambda,
                use_embeddings=mmr_use_embeddings,
                max_per_page=max_per_page,
                max_per_doc=max_per_doc,
                trace_out=mmr_trace if want_debug else None,
            )
        else:
            selected = self._apply_dedupe(combined, chunk_rows, k, max_per_page, max_per_doc)
        results = self._rows_to_results(
            selected,
            chunk_rows,
            expand_neighbors=int(expand_neighbors or 0),
            expand_max_chars=int(expand_max_chars or 0),
        )
        if want_debug:
            debug_out["candidates_count"] = int(len(combined))
            debug_out["candidates_top"] = [cand_debug[cid] for cid, _ in combined[: min(50, len(combined))] if cid in cand_debug]
            debug_out["mmr_trace"] = mmr_trace
            debug_out["selected"] = []
            for i, r in enumerate(results, start=1):
                info = dict(cand_debug.get(r["chunk_id"], {}))
                info.update(
                    {
                        "rank": int(i),
                        "returned_score": float(r.get("score", 0.0)),
                        "text_chars": int(len(r.get("text") or "")),
                        "images_count": int(len(r.get("images") or [])),
                    }
                )
                debug_out["selected"].append(info)
        return results
