from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sqlite3
import sys
from collections import Counter
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
import httpx
from markdownify import markdownify as md


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "untitled"


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig", errors="ignore")


def parse_toc(doc_dir: Path, doc_title: str) -> dict[str, list[str]]:
    toc = doc_dir / "__tableofcontents.html"
    if not toc.exists():
        return {}

    soup = BeautifulSoup(read_text(toc), "html.parser")
    mapping: dict[str, list[str]] = {}
    stack: list[str] = []

    for table in soup.select("table.TableOfContents"):
        a = table.select_one("a.__tocentry")
        if not a:
            continue
        href = a.get("href")
        if not href or not isinstance(href, str):
            continue
        href = href.split("#", 1)[0]
        title = a.get_text(" ", strip=True)
        if not title:
            continue

        # Level is encoded by a run of empty spacer <td width="10"> cells.
        tds = table.find_all("td", recursive=True)
        spacer_count = 0
        for td in tds:
            w = td.get("width")
            if w == "10" and not td.get_text(strip=True):
                spacer_count += 1
                continue
            break
        level = spacer_count + 1

        if level <= 0:
            level = 1
        if level > len(stack) + 1:
            level = len(stack) + 1

        stack = stack[: level - 1]
        stack.append(title)
        mapping[href] = [doc_title] + stack.copy()

    return mapping


def classify_hs_box(icon_src: str) -> str:
    s = icon_src.lower()
    if "caution" in s:
        return "Caution"
    if "warning" in s:
        return "Warning"
    if "tip" in s:
        return "Tip"
    if "note" in s:
        return "Note"
    return "Note"


def html_to_markdown(
    html_text: str,
    page_title: str,
    version: str,
    doc_id: str,
    doc_dir: Path,
    assets_out_dir: Path,
) -> tuple[str, list[dict[str, str]]]:
    soup = BeautifulSoup(html_text, "html.parser")
    main = soup.select_one("#mainbody") or soup.body
    if main is None:
        return f"# {page_title}\n", []

    for el in main.select("script,style"):
        el.decompose()

    images: list[dict[str, str]] = []

    # Convert Help+Manual box tables into explicit paragraphs.
    for box in main.select("table.hs-box"):
        icon = box.select_one("td.hs-box-icon img")
        icon_src = icon.get("src", "") if icon else ""
        kind = classify_hs_box(str(icon_src))

        content_td = box.select_one("td.hs-box-content")
        content_text = content_td.get_text(" ", strip=True) if content_td else box.get_text(" ", strip=True)

        p = soup.new_tag("p")
        strong = soup.new_tag("strong")
        strong.string = f"{kind}:"
        p.append(strong)
        p.append(" " + content_text)
        box.replace_with(p)

    # Copy and remap images.
    for img in main.select("img[src]"):
        src = str(img.get("src") or "").strip()
        if not src or src.startswith("http") or src.startswith("data:"):
            continue
        # normalize ./ prefix
        if src.startswith("./"):
            src = src[2:]
        src_path = (doc_dir / src).resolve()
        try:
            src_path.relative_to(doc_dir.resolve())
        except Exception:
            continue
        if not src_path.exists() or not src_path.is_file():
            continue

        out_path = (assets_out_dir / doc_id / src).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, out_path)

        url = f"/assets/{version}/{doc_id}/{src}"
        images.append({"original": src, "url": url, "alt": str(img.get("alt") or "").strip()})
        img["src"] = url

    md_body = md(str(main), heading_style="ATX", bullets="*")
    md_body = md_body.replace("\u00a0", " ")
    md_page = f"# {page_title}\n\n{md_body}".strip() + "\n"
    md_page = re.sub(r"\n{3,}", "\n\n", md_page)
    return md_page, images


def chunk_markdown(md_text: str, max_chars: int = 1400) -> list[dict[str, Any]]:
    parts = [p.strip() for p in md_text.split("\n\n") if p.strip()]
    chunks: list[dict[str, Any]] = []
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        text = "\n\n".join(buf).strip()
        imgs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
        chunks.append({"text": text, "images": imgs})
        buf.clear()

    for p in parts:
        if not buf:
            buf.append(p)
            continue
        if len("\n\n".join(buf)) + 2 + len(p) <= max_chars:
            buf.append(p)
            continue
        flush()
        buf.append(p)
    flush()
    return chunks


def init_index(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
          chunk_id TEXT PRIMARY KEY,
          doc_id TEXT NOT NULL,
          page_id TEXT NOT NULL,
          heading_path_json TEXT NOT NULL,
          text TEXT NOT NULL,
          source_path TEXT NOT NULL,
          anchor TEXT,
          images_json TEXT NOT NULL,
          length INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS terms (
          term TEXT PRIMARY KEY,
          df INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS postings (
          term TEXT NOT NULL,
          chunk_id TEXT NOT NULL,
          tf INTEGER NOT NULL,
          PRIMARY KEY (term, chunk_id)
        );

        CREATE INDEX IF NOT EXISTS idx_postings_term ON postings(term);

        CREATE TABLE IF NOT EXISTS embeddings (
          chunk_id TEXT PRIMARY KEY,
          dim INTEGER NOT NULL,
          vector BLOB NOT NULL
        );
        """
    )
    conn.commit()
    return conn


def _detect_embedding_model_id(base_url: str) -> str:
    resp = httpx.get(f"{base_url}/v1/models", timeout=10.0)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data") or []
    for m in models:
        mid = str(m.get("id") or "")
        if not mid:
            continue
        low = mid.lower()
        if "embedding" in low or low.startswith("text-embedding"):
            return mid
    raise RuntimeError("No embedding model found in LM Studio /v1/models. Load an embedding model in LM Studio.")


def _embed_texts(base_url: str, model_id: str, texts: list[str]) -> list[list[float]]:
    payload = {"model": model_id, "input": texts}
    resp = httpx.post(f"{base_url}/v1/embeddings", json=payload, timeout=60.0)
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Embeddings HTTP {resp.status_code}: {resp.text[:1000]}") from e
    data = resp.json()
    items = data.get("data") or []
    if len(items) != len(texts):
        raise RuntimeError(f"Embedding response size mismatch: got {len(items)} embeddings for {len(texts)} inputs.")
    # items are typically sorted by index
    items = sorted(items, key=lambda x: int(x.get("index", 0)))
    out: list[list[float]] = []
    for it in items:
        emb = it.get("embedding")
        if not isinstance(emb, list) or not emb:
            raise RuntimeError("Embedding response missing 'embedding' list.")
        out.append([float(x) for x in emb])
    return out


def _normalize(vec: list[float]) -> array:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    a = array("f", (float(x / norm) for x in vec))
    return a


def _prepare_embedding_text(text: str, max_chars: int) -> str:
    t = (text or "").replace("\x00", " ").strip()
    t = (
        t.replace("\u00a0", " ")
        .replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2026", "...")
    )
    # LM Studio embedding models can crash on some unicode punctuation; keep embeddings ASCII-safe.
    t = t.encode("ascii", "ignore").decode("ascii")
    if max_chars > 0 and len(t) > max_chars:
        t = t[:max_chars]
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest Help+Manual HTML export into BM25 datastore (Evo 1.32).")
    ap.add_argument("--docs-dir", type=Path, default=Path("docs"), help="Input docs directory (HTML export root)")
    ap.add_argument("--out-dir", type=Path, default=Path("datastore/evo_1_32"), help="Output datastore directory")
    ap.add_argument("--version", type=str, default="evo_1_32", help="Version id used in /assets/{version}/ URLs")
    ap.add_argument("--lmstudio-base-url", type=str, default="http://localhost:1234", help="LM Studio base URL")
    ap.add_argument("--embedding-model", type=str, default="", help="Embedding model id (defaults to first embedding model from /v1/models)")
    ap.add_argument("--embedding-max-chars", type=int, default=4000, help="Max characters per chunk sent for embedding")
    ap.add_argument("--no-embeddings", action="store_true", help="Skip computing embeddings")
    ap.add_argument("--clean", action="store_true", help="Delete existing out-dir before ingesting")
    args = ap.parse_args()

    docs_dir: Path = args.docs_dir
    out_dir: Path = args.out_dir
    version: str = args.version
    lmstudio_base_url: str = str(args.lmstudio_base_url).rstrip("/")
    embedding_model: str = str(args.embedding_model).strip()
    embedding_max_chars: int = int(args.embedding_max_chars)
    compute_embeddings: bool = not bool(args.no_embeddings)

    if not docs_dir.exists():
        log(f"ERROR: docs dir not found: {docs_dir}")
        return 2

    if out_dir.exists():
        if args.clean:
            log(f"Cleaning {out_dir} ...")
            shutil.rmtree(out_dir)
        else:
            # avoid accidental clobber
            if any(out_dir.iterdir()):
                log(f"ERROR: out dir not empty: {out_dir} (use --clean to overwrite)")
                return 2

    pages_out = out_dir / "pages"
    assets_out = out_dir / "assets"
    pages_out.mkdir(parents=True, exist_ok=True)
    assets_out.mkdir(parents=True, exist_ok=True)

    index_path = out_dir / "index.sqlite"
    conn = init_index(index_path)

    pages_jsonl_path = out_dir / "pages.jsonl"
    pages_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    pages_jsonl = pages_jsonl_path.open("w", encoding="utf-8")

    doc_dirs = sorted([p for p in docs_dir.iterdir() if p.is_dir()])
    if not doc_dirs:
        log(f"ERROR: no doc directories found under {docs_dir}")
        return 2

    df_counter: Counter[str] = Counter()
    postings_rows: list[tuple[str, str, int]] = []
    chunk_rows: list[tuple[str, str, str, str, str, str, str | None, str, int]] = []
    chunk_id_text: list[tuple[str, str]] = []

    total_tokens = 0
    n_chunks = 0

    for doc_dir in doc_dirs:
        doc_title = doc_dir.name
        doc_id = slugify(doc_title)
        toc_map = parse_toc(doc_dir, doc_title)

        log(f"Ingesting doc: {doc_title} -> {doc_id} ({len(toc_map)} toc entries)")

        # Determine which HTML pages to ingest.
        if toc_map:
            page_files = [doc_dir / href for href in toc_map.keys() if href.lower().endswith(".html")]
        else:
            page_files = sorted([p for p in doc_dir.glob("*.html") if not p.name.startswith("__")])

        for html_path in page_files:
            if not html_path.exists():
                continue
            html_rel = str(html_path.relative_to(docs_dir))
            html_text = read_text(html_path)
            soup = BeautifulSoup(html_text, "html.parser")
            title_el = soup.select_one("#pagetitle")
            page_title = title_el.get_text(" ", strip=True) if title_el else html_path.stem

            page_id = slugify(html_path.stem)
            heading_path = toc_map.get(html_path.name) or [doc_title, page_title]

            md_text, images = html_to_markdown(
                html_text=html_text,
                page_title=page_title,
                version=version,
                doc_id=doc_id,
                doc_dir=doc_dir,
                assets_out_dir=assets_out,
            )

            md_path = pages_out / doc_id / f"{page_id}.md"
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(md_text, encoding="utf-8")

            page_record = {
                "version": version,
                "doc_id": doc_id,
                "doc_title": doc_title,
                "page_id": page_id,
                "page_title": page_title,
                "heading_path": heading_path,
                "source_path": html_rel,
                "anchor": "pagetitle",
                "images": images,
                "markdown_path": str(md_path.relative_to(out_dir)),
            }
            pages_jsonl.write(json.dumps(page_record, ensure_ascii=False) + "\n")

            chunks = chunk_markdown(md_text)
            for i, ch in enumerate(chunks):
                chunk_id = f"{doc_id}:{page_id}:{i:03d}"
                text = ch["text"]
                images_urls = [u for u in ch.get("images") or [] if u.startswith("/assets/")]
                tokens = tokenize(text)
                dl = len(tokens)
                if dl == 0:
                    continue
                chunk_id_text.append((chunk_id, text))
                n_chunks += 1
                total_tokens += dl

                tf = Counter(tokens)
                for term, term_tf in tf.items():
                    postings_rows.append((term, chunk_id, int(term_tf)))
                for term in tf.keys():
                    df_counter[term] += 1

                chunk_rows.append(
                    (
                        chunk_id,
                        doc_id,
                        page_id,
                        json.dumps(heading_path, ensure_ascii=False),
                        text,
                        html_rel,
                        "pagetitle",
                        json.dumps(images_urls, ensure_ascii=False),
                        dl,
                    )
                )

        log(f"  Done: {doc_title}")

    pages_jsonl.close()

    if n_chunks == 0:
        log("ERROR: no chunks were produced; check docs input.")
        return 2

    avgdl = total_tokens / n_chunks
    log(f"Indexing {n_chunks} chunks (avgdl={avgdl:.2f}) ...")

    embedding_model_id = None
    embedding_dim = None

    try:
        with conn:
            conn.execute("DELETE FROM meta;")
            conn.execute("DELETE FROM chunks;")
            conn.execute("DELETE FROM terms;")
            conn.execute("DELETE FROM postings;")
            conn.execute("DELETE FROM embeddings;")

            conn.executemany(
                """
                INSERT INTO chunks(chunk_id, doc_id, page_id, heading_path_json, text, source_path, anchor, images_json, length)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                chunk_rows,
            )

            # Insert terms.
            conn.executemany(
                "INSERT INTO terms(term, df) VALUES (?,?)",
                [(t, int(df)) for t, df in df_counter.items()],
            )

            # Insert postings in batches.
            batch_size = 5000
            for i in range(0, len(postings_rows), batch_size):
                conn.executemany(
                    "INSERT INTO postings(term, chunk_id, tf) VALUES (?,?,?)",
                    postings_rows[i : i + batch_size],
                )

            if compute_embeddings:
                embedding_model_id = embedding_model or _detect_embedding_model_id(lmstudio_base_url)
                log(f"Computing embeddings via {lmstudio_base_url} model={embedding_model_id} ...")

                emb_batch = 32
                rows: list[tuple[str, int, bytes]] = []
                for i in range(0, len(chunk_id_text), emb_batch):
                    batch = chunk_id_text[i : i + emb_batch]
                    ids = [x[0] for x in batch]
                    texts = [_prepare_embedding_text(x[1], embedding_max_chars) for x in batch]
                    try:
                        vectors = _embed_texts(lmstudio_base_url, embedding_model_id, texts)
                    except Exception as e:
                        max_len = max((len(t) for t in texts), default=0)
                        raise RuntimeError(
                            f"Embedding request failed at batch {i//emb_batch + 1} (max_chars={max_len}): {e}"
                        ) from e

                    if embedding_dim is None:
                        embedding_dim = len(vectors[0])
                    for v in vectors:
                        if embedding_dim != len(v):
                            raise RuntimeError("Embedding dimension mismatch across batches")

                    for cid, vec in zip(ids, vectors):
                        a = _normalize(vec)
                        rows.append((cid, int(embedding_dim), a.tobytes()))

                    if len(rows) >= 2000:
                        conn.executemany("INSERT INTO embeddings(chunk_id, dim, vector) VALUES (?,?,?)", rows)
                        rows.clear()

                if rows:
                    conn.executemany("INSERT INTO embeddings(chunk_id, dim, vector) VALUES (?,?,?)", rows)

            conn.executemany(
                "INSERT INTO meta(key, value) VALUES (?,?)",
                [
                    ("version", version),
                    ("created_at", utc_now()),
                    ("n_chunks", str(n_chunks)),
                    ("avgdl", f"{avgdl:.6f}"),
                    ("embeddings_enabled", "1" if compute_embeddings else "0"),
                    ("embedding_model_id", str(embedding_model_id or "")),
                    ("embedding_dim", str(int(embedding_dim or 0))),
                    ("embedding_max_chars", str(int(embedding_max_chars))),
                ],
            )
    except Exception as e:
        log(f"ERROR: ingestion failed: {e}")
        conn.close()
        return 2

    conn.close()
    log(f"Done. Output written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
