from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
import httpx
from markdownify import markdownify as md


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_EMBED_H3_RE = re.compile(r"(?m)^([ \t]*)###(\s+)")
_CONTROL_RE = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


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
        if not src:
            img.decompose()
            continue
        if src.startswith("data:"):
            # Help+Manual sometimes embeds large base64 images. Keep output deterministic and avoid
            # polluting markdown/chunks with base64 blobs that break embedding backends.
            alt = str(img.get("alt") or "").strip()
            if alt:
                span = soup.new_tag("span")
                span.string = f"[Image: {alt}]"
                img.replace_with(span)
            else:
                img.decompose()
            continue
        if src.startswith("http"):
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


def _split_long_block(text: str, max_chars: int) -> list[str]:
    s = (text or "").strip()
    if not s:
        return []
    if max_chars <= 0 or len(s) <= max_chars:
        return [s]

    out: list[str] = []
    while s:
        if len(s) <= max_chars:
            out.append(s)
            break

        cut = max_chars
        # Prefer splitting at a newline near the end (keeps lists/images intact),
        # then whitespace; fall back to a hard cut.
        nl = s.rfind("\n", int(max_chars * 0.6), max_chars)
        if nl != -1:
            cut = nl
        else:
            sp = s.rfind(" ", int(max_chars * 0.6), max_chars)
            if sp != -1:
                cut = sp

        part = s[:cut].rstrip()
        if part:
            out.append(part)
        s = s[cut:].lstrip()

    return out


def chunk_markdown(md_text: str, max_chars: int = 1400) -> list[dict[str, Any]]:
    raw_parts = [p.strip() for p in md_text.split("\n\n") if p.strip()]
    parts: list[str] = []
    for p in raw_parts:
        parts.extend(_split_long_block(p, max_chars=max_chars))
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


def _select_semantic_levels(headings: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    if not headings:
        return (None, None)
    levels = sorted({int(h.get("level") or 0) for h in headings if int(h.get("level") or 0) > 0})
    if not levels:
        return (None, None)
    levels_gt1 = [l for l in levels if l > 1]
    if levels_gt1:
        topic_level = 2 if 2 in levels_gt1 else min(levels_gt1)
    else:
        topic_level = min(levels)
    section_level = None
    for lvl in levels_gt1:
        if lvl > topic_level:
            section_level = lvl
            break
    return (topic_level, section_level)


def _sections_for_level_or_page(
    lines: list[str],
    headings: list[dict[str, Any]],
    level: int | None,
    *,
    doc_title: str,
    page_title: str,
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    if level is not None:
        sections = _sections_for_level(lines, headings, level)
    if sections:
        return sections
    text = "\n".join(lines).strip()
    if not text:
        return []
    return [{"heading_path": [doc_title, page_title], "text": text}]


def _chunks_from_sections(
    sections: list[dict[str, Any]],
    *,
    max_chars: int,
    granularity: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for sec in sections:
        raw_text = str(sec.get("text") or "").strip()
        if not raw_text:
            continue
        heading_path = sec.get("heading_path") or []
        for chunk in chunk_markdown(raw_text, max_chars=max_chars):
            out.append(
                {
                    "granularity": granularity,
                    "heading_path": heading_path,
                    "text": chunk.get("text") or "",
                    "images": chunk.get("images") or [],
                }
            )
    return out


_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def _split_text_on_images(text: str, heading_path: list[str]) -> list[dict[str, Any]]:
    lines = (text or "").splitlines()
    blocks: list[dict[str, Any]] = []
    buf: list[str] = []

    def flush(images: list[str]) -> None:
        t = "\n".join(buf).strip()
        if not t:
            return
        blocks.append({"text": t, "images": list(images)})
        buf.clear()

    for line in lines:
        matches = _IMAGE_RE.findall(line)
        if matches:
            stripped = _IMAGE_RE.sub("", line).strip()
            if stripped:
                buf.append(stripped)
            if buf:
                flush(matches)
            else:
                if blocks:
                    blocks[-1]["images"].extend(matches)
                else:
                    placeholder = " > ".join(heading_path or [])
                    if placeholder:
                        blocks.append({"text": placeholder, "images": list(matches)})
            continue
        buf.append(line)

    if buf:
        blocks.append({"text": "\n".join(buf).strip(), "images": []})

    return [b for b in blocks if b.get("text") or b.get("images")]


def _split_chunks_by_images(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ch in chunks:
        heading_path = ch.get("heading_path") or []
        parts = _split_text_on_images(str(ch.get("text") or ""), heading_path)
        if not parts:
            continue
        for part in parts:
            out.append(
                {
                    "granularity": ch.get("granularity"),
                    "heading_path": heading_path,
                    "text": part.get("text") or "",
                    "images": part.get("images") or [],
                }
            )
    return out


def semantic_chunk_markdown(
    md_text: str,
    *,
    doc_title: str,
    page_title: str,
    max_chars_part: int = 900,
    max_chars_section: int = 2600,
    max_chars_topic: int = 5200,
) -> list[dict[str, Any]]:
    lines = (md_text or "").splitlines()
    headings = _extract_headings(lines, doc_title=doc_title)
    topic_level, section_level = _select_semantic_levels(headings)

    topic_sections = _sections_for_level_or_page(
        lines,
        headings,
        topic_level,
        doc_title=doc_title,
        page_title=page_title,
    )
    if not topic_sections:
        return []

    chunks: list[dict[str, Any]] = []
    chunks.extend(
        _chunks_from_sections(
            topic_sections,
            max_chars=max_chars_topic,
            granularity="topic",
        )
    )

    section_sections: list[dict[str, Any]] = []
    if section_level is not None:
        section_sections = _sections_for_level(lines, headings, section_level)
        if not section_sections:
            section_sections = []
    if section_sections:
        chunks.extend(
            _chunks_from_sections(
                section_sections,
                max_chars=max_chars_section,
                granularity="section",
            )
        )
        part_sections = section_sections
    else:
        part_sections = topic_sections

    chunks.extend(
        _chunks_from_sections(
            part_sections,
            max_chars=max_chars_part,
            granularity="part",
        )
    )
    return _split_chunks_by_images(chunks)


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

        CREATE TABLE IF NOT EXISTS summary_chunks (
          summary_id TEXT PRIMARY KEY,
          doc_id TEXT NOT NULL,
          page_id TEXT NOT NULL,
          heading_path_json TEXT NOT NULL,
          text TEXT NOT NULL,
          source_path TEXT NOT NULL,
          anchor TEXT,
          length INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS summary_terms (
          term TEXT PRIMARY KEY,
          df INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS summary_postings (
          term TEXT NOT NULL,
          summary_id TEXT NOT NULL,
          tf INTEGER NOT NULL,
          PRIMARY KEY (term, summary_id)
        );

        CREATE INDEX IF NOT EXISTS idx_summary_postings_term ON summary_postings(term);
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

    for attempt in range(1, 4):
        try:
            resp = httpx.post(f"{base_url}/v1/embeddings", json=payload, timeout=60.0)
            break
        except httpx.RequestError as e:
            if attempt >= 3:
                raise RuntimeError(f"Embeddings request failed: {e}") from e
            time.sleep(0.3 * attempt)
    else:
        raise RuntimeError("Embeddings request failed after retries")

    if resp.status_code >= 400:
        body = ""
        try:
            body = resp.text
        except Exception:
            body = ""
        raise RuntimeError(f"Embeddings HTTP {resp.status_code}: {body[:1000]}")

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
    t = (text or "").replace("\x00", " ")
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\u00a0", " ")
    t = t.replace("\\*", "*")
    t = (
        t.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2026", "...")
    )
    # Work around an LM Studio embeddings crash triggered by some markdown H3 headings ("### ").
    # Downgrade to H2 markers instead of stripping (keeps structure stable).
    t = _EMBED_H3_RE.sub(r"\1##\2", t)
    t = _markdown_tables_to_text(t)
    t = t.replace("**", "")
    t = t.replace("*", "")
    t = _CONTROL_RE.sub(" ", t)
    t = t.strip()
    if max_chars > 0 and len(t) > max_chars:
        t = t[:max_chars]
    return t


def _summary_prompt() -> str:
    return (
        "You are summarizing Luxriot EVO documentation for retrieval routing.\n"
        "Rules:\n"
        "- Only use information from the provided text.\n"
        "- Preserve exact UI/menu/button names as written.\n"
        "- Keep it short and factual (1-3 sentences).\n"
        "- Do not invent steps, settings, or requirements.\n"
        "- Output plain text, no markdown lists."
    )


def _chat_completion(base_url: str, model_id: str, text: str, max_tokens: int, timeout_s: float = 120.0) -> str:
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": _summary_prompt()},
            {"role": "user", "content": text},
        ],
        "temperature": 0.1,
        "max_tokens": int(max_tokens),
    }
    resp = httpx.post(f"{base_url}/v1/chat/completions", json=payload, timeout=timeout_s)
    if resp.status_code >= 400:
        raise RuntimeError(f"Summary HTTP {resp.status_code}: {resp.text[:1000]}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Summary response missing choices")
    msg = choices[0].get("message") or {}
    content = str(msg.get("content") or "").strip()
    if not content:
        raise RuntimeError("Summary response empty")
    return content


def _token_count(text: str) -> int:
    return len(tokenize(text))


def _strip_markdown_images(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if "![" in line and "](" in line:
            continue
        out.append(line)
    return out


def _extract_headings(lines: list[str], doc_title: str) -> list[dict[str, Any]]:
    stack: list[tuple[int, str]] = []
    headings: list[dict[str, Any]] = []
    for idx, line in enumerate(lines):
        m = _HEADING_RE.match(line.strip())
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        if not title:
            continue
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = [doc_title] + [t for _, t in stack]
        headings.append({"line": idx, "level": level, "title": title, "path": path})
    return headings


def _sections_for_level(
    lines: list[str], headings: list[dict[str, Any]], level: int
) -> list[dict[str, Any]]:
    items = [h for h in headings if int(h.get("level") or 0) == level]
    if not items:
        return []
    sections: list[dict[str, Any]] = []
    for h in items:
        start = int(h["line"])
        end = len(lines)
        for nxt in headings:
            if int(nxt.get("line")) <= start:
                continue
            if int(nxt.get("level") or 0) <= level:
                end = int(nxt.get("line"))
                break
        text = "\n".join(lines[start:end]).strip()
        if not text:
            continue
        sections.append({"heading_path": h["path"], "text": text})
    return sections


def split_markdown_for_summary(
    md_text: str, *, doc_title: str, page_title: str, unit_max_tokens: int
) -> list[dict[str, Any]]:
    lines = md_text.splitlines()
    lines = _strip_markdown_images(lines)
    headings = _extract_headings(lines, doc_title=doc_title)
    if not headings:
        text = "\n".join(lines).strip()
        return [{"heading_path": [doc_title, page_title], "text": text}] if text else []

    levels = sorted({int(h["level"]) for h in headings})
    chosen: list[dict[str, Any]] = []
    for level in levels:
        sections = _sections_for_level(lines, headings, level)
        if not sections:
            continue
        max_tokens = max((_token_count(s["text"]) for s in sections), default=0)
        if max_tokens <= unit_max_tokens:
            chosen = sections
            break
    if not chosen:
        chosen = _sections_for_level(lines, headings, max(levels))
    return chosen


def _load_published_edits(app_db: Path, version: str) -> dict[tuple[str, str], str]:
    if not app_db.exists():
        log(f"WARNING: app db not found at {app_db}; skipping published edits.")
        return {}
    conn = sqlite3.connect(str(app_db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT doc_id, page_id, content_md
            FROM doc_edits
            WHERE version = ? AND status = 'published'
            """,
            (version,),
        ).fetchall()
    except sqlite3.Error as e:
        log(f"WARNING: failed to read doc_edits: {e}")
        return {}
    finally:
        conn.close()

    out: dict[tuple[str, str], str] = {}
    for r in rows:
        doc_id = str(r["doc_id"] or "").strip()
        page_id = str(r["page_id"] or "").strip()
        if not doc_id or not page_id:
            continue
        content = str(r["content_md"] or "").strip()
        if not content:
            continue
        out[(doc_id, page_id)] = content
    if out:
        log(f"Loaded {len(out)} published edits from app db.")
    return out


def _load_custom_pages(app_db: Path, version: str) -> list[dict[str, Any]]:
    if not app_db.exists():
        return []
    conn = sqlite3.connect(str(app_db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT doc_id, page_id, doc_title, page_title, heading_path_json, source_path, anchor, base_markdown
            FROM doc_pages
            WHERE version = ?
            """,
            (version,),
        ).fetchall()
    except sqlite3.Error as e:
        log(f"WARNING: failed to read doc_pages: {e}")
        return []
    finally:
        conn.close()

    out: list[dict[str, Any]] = []
    for r in rows:
        doc_id = str(r["doc_id"] or "").strip()
        page_id = str(r["page_id"] or "").strip()
        if not doc_id or not page_id:
            continue
        doc_title = str(r["doc_title"] or doc_id)
        page_title = str(r["page_title"] or page_id)
        heading_path: list[str] = []
        raw_hp = str(r["heading_path_json"] or "").strip()
        if raw_hp:
            try:
                parsed = json.loads(raw_hp)
                if isinstance(parsed, list):
                    heading_path = [str(x) for x in parsed if str(x).strip()]
            except json.JSONDecodeError:
                heading_path = []
        if not heading_path:
            heading_path = [doc_title, page_title]
        out.append(
            {
                "doc_id": doc_id,
                "page_id": page_id,
                "doc_title": doc_title,
                "page_title": page_title,
                "heading_path": heading_path,
                "source_path": str(r["source_path"] or f"custom/{doc_id}/{page_id}.md"),
                "anchor": str(r["anchor"] or "pagetitle"),
                "base_markdown": str(r["base_markdown"] or ""),
            }
        )
    if out:
        log(f"Loaded {len(out)} custom pages from app db.")
    return out


def _markdown_tables_to_text(text: str) -> str:
    lines: list[str] = []
    for line in (text or "").splitlines():
        if "|" not in line:
            lines.append(line)
            continue
        stripped = line.strip().strip("|").strip()
        if not stripped:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # Table separator row: only dashes/colons.
        if cells and all((not c) or (set(c) <= set("-:")) for c in cells):
            continue
        cleaned = []
        for c in cells:
            c = c.strip()
            if not c:
                continue
            c = c.replace("**", "")
            c = re.sub(r"\s+", " ", c).strip()
            if c:
                cleaned.append(c)
        if cleaned:
            lines.append(" - ".join(cleaned))
    return "\n".join(lines)


def _prepare_embedding_variants(text: str, max_chars: int) -> list[tuple[str, str]]:
    base = _prepare_embedding_text(text, max_chars)
    # Keep variants deterministic and only used on failures.
    variants: list[tuple[str, str]] = [("base", base)]

    ascii_safe = base.encode("ascii", "ignore").decode("ascii")
    if ascii_safe != base:
        variants.append(("ascii_safe", ascii_safe))

    collapsed_ws = " ".join(base.split())
    if collapsed_ws and collapsed_ws != base:
        variants.append(("collapsed_ws", collapsed_ws))

    # Some local embedding backends are brittle for certain input lengths.
    # Prefer trying a few deterministic truncation sizes that preserve context.
    for n in (1280, 1200, 1100, 1024, 896, 832, 768, 704, 640, 576, 512, 448, 384, 320, 256):
        if len(base) > n:
            variants.append((f"truncate_{n}", base[:n]))

    # Drop very short standalone rows (common table artifacts).
    non_short_lines = [ln for ln in base.splitlines() if len(ln.strip()) >= 12]
    if non_short_lines:
        variants.append(("drop_short_lines", "\n".join(non_short_lines)))

    # Last resort: bag-of-words string (stable, punctuation-free).
    words = re.findall(r"[A-Za-z0-9]+", base)
    seen: set[str] = set()
    bag: list[str] = []
    for w in words:
        lw = w.lower()
        if lw in seen:
            continue
        seen.add(lw)
        bag.append(w)
        if len(" ".join(bag)) >= 420:
            break
    if bag:
        variants.append(("bag_of_words", " ".join(bag)))

    return variants


def _embed_texts_resilient(
    base_url: str,
    model_id: str,
    chunk_ids: list[str],
    raw_texts: list[str],
    max_chars: int,
) -> list[list[float]]:
    prepared = [_prepare_embedding_text(t, max_chars) for t in raw_texts]
    try:
        return _embed_texts(base_url, model_id, prepared)
    except Exception as e:
        log(f"Embedding batch failed ({len(prepared)} items); trying fallbacks: {e}")

    # Try a few batch-wide fallbacks first to avoid per-chunk slowdown.
    def bag_of_words(s: str) -> str:
        words = re.findall(r"[A-Za-z0-9]+", s)
        seen: set[str] = set()
        bag: list[str] = []
        for w in words:
            lw = w.lower()
            if lw in seen:
                continue
            seen.add(lw)
            bag.append(w)
            if len(" ".join(bag)) >= 420:
                break
        return " ".join(bag)

    batch_fallbacks: list[tuple[str, list[str]]] = []
    for n in (1280, 1024, 896, 768, 640, 576, 512, 448, 384, 320, 256):
        batch_fallbacks.append((f"truncate_{n}", [t[:n] for t in prepared]))
    batch_fallbacks.append(("collapsed_ws", [" ".join(t.split()) for t in prepared]))
    batch_fallbacks.append(("bag_of_words", [bag_of_words(t) for t in prepared]))

    for name, texts in batch_fallbacks:
        try:
            vectors = _embed_texts(base_url, model_id, texts)
            log(f"Embedding batch recovered using variant={name}")
            return vectors
        except Exception:
            continue

    out: list[list[float]] = []
    for cid, raw, prepared_text in zip(chunk_ids, raw_texts, prepared):
        try:
            out.append(_embed_texts(base_url, model_id, [prepared_text])[0])
            continue
        except Exception as e:
            last_err: Exception = e

        for variant_name, variant_text in _prepare_embedding_variants(raw, max_chars):
            try:
                out.append(_embed_texts(base_url, model_id, [variant_text])[0])
                log(f"Embedding recovered for {cid} using variant={variant_name}")
                break
            except Exception as e:
                last_err = e
        else:
            snippet = prepared_text[:260].replace("\n", " ")
            raise RuntimeError(
                f"Embedding failed for chunk_id={cid} after retries. "
                f"Last error: {last_err}. Prepared snippet: {snippet}"
            ) from last_err

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest Help+Manual HTML export into BM25 datastore (Evo 1.32).")
    ap.add_argument("--docs-dir", type=Path, default=Path("docs"), help="Input docs directory (HTML export root)")
    ap.add_argument("--out-dir", type=Path, default=Path("datastore/evo_1_32"), help="Output datastore directory")
    ap.add_argument("--version", type=str, default="evo_1_32", help="Version id used in /assets/{version}/ URLs")
    ap.add_argument(
        "--app-db",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "backend" / "data" / "app.sqlite",
        help="App SQLite db path (for published edits/custom pages)",
    )
    ap.add_argument(
        "--include-edits",
        action="store_true",
        help="Include published edits + custom pages from the app database",
    )
    ap.add_argument("--summary-enabled", action="store_true", help="Build summary index for two-pass retrieval")
    ap.add_argument("--summary-model", type=str, default="", help="LLM model id used to summarize sections")
    ap.add_argument("--summary-max-input-chars", type=int, default=6000, help="Max chars per summary input")
    ap.add_argument("--summary-max-output-tokens", type=int, default=280, help="Max tokens per summary output")
    ap.add_argument("--summary-unit-max-tokens", type=int, default=900, help="Max tokens per summary unit (controls heading level)")
    ap.add_argument("--chunk-max-chars-part", type=int, default=900, help="Max chars per fine-grained chunk (subsection)")
    ap.add_argument("--chunk-max-chars-section", type=int, default=2600, help="Max chars per section chunk")
    ap.add_argument("--chunk-max-chars-topic", type=int, default=5200, help="Max chars per topic chunk")
    ap.add_argument("--lmstudio-base-url", type=str, default="http://localhost:1234", help="LM Studio base URL")
    ap.add_argument("--embedding-model", type=str, default="", help="Embedding model id (defaults to first embedding model from /v1/models)")
    ap.add_argument("--embedding-max-chars", type=int, default=448, help="Max characters per chunk sent for embedding")
    ap.add_argument("--embedding-batch-size", type=int, default=8, help="How many chunks to embed per request (lower = more stable)")
    ap.add_argument("--no-embeddings", action="store_true", help="Skip computing embeddings")
    ap.add_argument("--clean", action="store_true", help="Delete existing out-dir before ingesting")
    args = ap.parse_args()

    docs_dir: Path = args.docs_dir
    out_dir: Path = args.out_dir
    version: str = args.version
    lmstudio_base_url: str = str(args.lmstudio_base_url).rstrip("/")
    embedding_model: str = str(args.embedding_model).strip()
    embedding_max_chars: int = int(args.embedding_max_chars)
    embedding_batch_size: int = max(1, int(args.embedding_batch_size))
    compute_embeddings: bool = not bool(args.no_embeddings)
    app_db_path: Path = Path(args.app_db)
    include_edits: bool = bool(args.include_edits)
    summary_enabled: bool = bool(args.summary_enabled)
    summary_model: str = str(args.summary_model or "").strip()
    summary_max_input_chars: int = int(args.summary_max_input_chars)
    summary_max_output_tokens: int = int(args.summary_max_output_tokens)
    summary_unit_max_tokens: int = int(args.summary_unit_max_tokens)
    chunk_max_chars_part: int = int(args.chunk_max_chars_part)
    chunk_max_chars_section: int = int(args.chunk_max_chars_section)
    chunk_max_chars_topic: int = int(args.chunk_max_chars_topic)

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

    summary_df_counter: Counter[str] = Counter()
    summary_postings_rows: list[tuple[str, str, int]] = []
    summary_rows: list[tuple[str, str, str, str, str, str, str | None, int]] = []
    summary_units = 0
    summary_total_tokens = 0
    summary_active = summary_enabled and bool(summary_model)
    summary_failed = False
    if summary_enabled and not summary_model:
        log("WARNING: summary enabled but --summary-model is empty; skipping summary index.")
        summary_active = False

    published_edits: dict[tuple[str, str], str] = {}
    custom_pages: list[dict[str, Any]] = []
    if include_edits:
        published_edits = _load_published_edits(app_db_path, version)
        custom_pages = _load_custom_pages(app_db_path, version)

    seen_pages: set[tuple[str, str]] = set()

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
            edit_text = published_edits.get((doc_id, page_id))
            if edit_text:
                md_text = edit_text

            if summary_active:
                sections = split_markdown_for_summary(
                    md_text,
                    doc_title=doc_title,
                    page_title=page_title,
                    unit_max_tokens=summary_unit_max_tokens,
                )
                if sections:
                    log(f"  Summarizing {doc_id}/{page_id} ({len(sections)} sections)...")
                for s_idx, sec in enumerate(sections):
                    raw_text = str(sec.get("text") or "").strip()
                    if not raw_text:
                        continue
                    summary_input = raw_text[:summary_max_input_chars] if summary_max_input_chars > 0 else raw_text
                    try:
                        summary_text = _chat_completion(
                            lmstudio_base_url,
                            summary_model,
                            summary_input,
                            max_tokens=summary_max_output_tokens,
                        ).strip()
                    except Exception as e:
                        log(f"WARNING: summary failed for {doc_id}/{page_id}: {e}")
                        summary_active = False
                        summary_failed = True
                        break
                    if not summary_text:
                        continue
                    summary_id = f"{doc_id}:{page_id}:s{s_idx:03d}"
                    heading_path = sec.get("heading_path") or [doc_title, page_title]
                    tokens = tokenize(summary_text)
                    dl = len(tokens)
                    if dl == 0:
                        continue
                    summary_units += 1
                    summary_total_tokens += dl
                    tf = Counter(tokens)
                    for term, term_tf in tf.items():
                        summary_postings_rows.append((term, summary_id, int(term_tf)))
                    for term in tf.keys():
                        summary_df_counter[term] += 1
                    summary_rows.append(
                        (
                            summary_id,
                            doc_id,
                            page_id,
                            json.dumps(heading_path, ensure_ascii=False),
                            summary_text,
                            html_rel,
                            "pagetitle",
                            dl,
                        )
                    )
                if not summary_active:
                    log("WARNING: summary disabled after failure; continuing without summary index.")

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

            chunks = semantic_chunk_markdown(
                md_text,
                doc_title=doc_title,
                page_title=page_title,
                max_chars_part=chunk_max_chars_part,
                max_chars_section=chunk_max_chars_section,
                max_chars_topic=chunk_max_chars_topic,
            )
            granularity_counts: dict[str, int] = {"topic": 0, "section": 0, "part": 0}
            granularity_prefix = {"topic": "t", "section": "s", "part": "p"}
            for ch in chunks:
                granularity = str(ch.get("granularity") or "part")
                prefix = granularity_prefix.get(granularity, "p")
                idx = granularity_counts.get(granularity, 0)
                granularity_counts[granularity] = idx + 1
                chunk_id = f"{doc_id}:{page_id}:{prefix}{idx:03d}"
                text = str(ch.get("text") or "")
                images_urls = [u for u in ch.get("images") or [] if str(u).startswith("/assets/")]
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
                        json.dumps(ch.get("heading_path") or heading_path, ensure_ascii=False),
                        text,
                        html_rel,
                        "pagetitle",
                        json.dumps(images_urls, ensure_ascii=False),
                        dl,
                    )
                )
            seen_pages.add((doc_id, page_id))

        log(f"  Done: {doc_title}")

    for custom in custom_pages:
        doc_id = custom["doc_id"]
        page_id = custom["page_id"]
        if (doc_id, page_id) in seen_pages:
            continue
        doc_title = custom["doc_title"]
        page_title = custom["page_title"]
        heading_path = custom["heading_path"]
        source_path = custom["source_path"]
        anchor = custom["anchor"]
        md_text = published_edits.get((doc_id, page_id)) or custom["base_markdown"]
        if not md_text.strip():
            md_text = f"# {page_title}\n\n"

        if summary_active:
            sections = split_markdown_for_summary(
                md_text,
                doc_title=doc_title,
                page_title=page_title,
                unit_max_tokens=summary_unit_max_tokens,
            )
            if sections:
                log(f"  Summarizing {doc_id}/{page_id} ({len(sections)} sections)...")
            for s_idx, sec in enumerate(sections):
                raw_text = str(sec.get("text") or "").strip()
                if not raw_text:
                    continue
                summary_input = raw_text[:summary_max_input_chars] if summary_max_input_chars > 0 else raw_text
                try:
                    summary_text = _chat_completion(
                        lmstudio_base_url,
                        summary_model,
                        summary_input,
                        max_tokens=summary_max_output_tokens,
                    ).strip()
                except Exception as e:
                    log(f"WARNING: summary failed for {doc_id}/{page_id}: {e}")
                    summary_active = False
                    summary_failed = True
                    break
                if not summary_text:
                    continue
                summary_id = f"{doc_id}:{page_id}:s{s_idx:03d}"
                sec_heading_path = sec.get("heading_path") or heading_path
                tokens = tokenize(summary_text)
                dl = len(tokens)
                if dl == 0:
                    continue
                summary_units += 1
                summary_total_tokens += dl
                tf = Counter(tokens)
                for term, term_tf in tf.items():
                    summary_postings_rows.append((term, summary_id, int(term_tf)))
                for term in tf.keys():
                    summary_df_counter[term] += 1
                summary_rows.append(
                    (
                        summary_id,
                        doc_id,
                        page_id,
                        json.dumps(sec_heading_path, ensure_ascii=False),
                        summary_text,
                        source_path,
                        anchor,
                        dl,
                    )
                )
            if not summary_active:
                log("WARNING: summary disabled after failure; continuing without summary index.")

        chunks = semantic_chunk_markdown(
            md_text,
            doc_title=doc_title,
            page_title=page_title,
            max_chars_part=chunk_max_chars_part,
            max_chars_section=chunk_max_chars_section,
            max_chars_topic=chunk_max_chars_topic,
        )
        granularity_counts: dict[str, int] = {"topic": 0, "section": 0, "part": 0}
        granularity_prefix = {"topic": "t", "section": "s", "part": "p"}
        for ch in chunks:
            granularity = str(ch.get("granularity") or "part")
            prefix = granularity_prefix.get(granularity, "p")
            idx = granularity_counts.get(granularity, 0)
            granularity_counts[granularity] = idx + 1
            chunk_id = f"{doc_id}:{page_id}:{prefix}{idx:03d}"
            text = str(ch.get("text") or "")
            images_urls = [u for u in ch.get("images") or [] if str(u).startswith("/assets/")]
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
                    json.dumps(ch.get("heading_path") or heading_path, ensure_ascii=False),
                    text,
                    source_path,
                    anchor,
                    json.dumps(images_urls, ensure_ascii=False),
                    dl,
                )
            )
        seen_pages.add((doc_id, page_id))

    if summary_failed:
        summary_rows = []
        summary_postings_rows = []
        summary_df_counter = Counter()
        summary_units = 0
        summary_total_tokens = 0

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
            conn.execute("DELETE FROM summary_chunks;")
            conn.execute("DELETE FROM summary_terms;")
            conn.execute("DELETE FROM summary_postings;")

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

            if summary_rows:
                conn.executemany(
                    """
                    INSERT INTO summary_chunks(
                      summary_id, doc_id, page_id, heading_path_json, text, source_path, anchor, length
                    )
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    summary_rows,
                )
                conn.executemany(
                    "INSERT INTO summary_terms(term, df) VALUES (?,?)",
                    [(t, int(df)) for t, df in summary_df_counter.items()],
                )
                for i in range(0, len(summary_postings_rows), batch_size):
                    conn.executemany(
                        "INSERT INTO summary_postings(term, summary_id, tf) VALUES (?,?,?)",
                        summary_postings_rows[i : i + batch_size],
                    )

            if compute_embeddings:
                embedding_model_id = embedding_model or _detect_embedding_model_id(lmstudio_base_url)
                log(
                    f"Computing embeddings via {lmstudio_base_url} model={embedding_model_id} "
                    f"(batch={embedding_batch_size}, max_chars={embedding_max_chars}) ..."
                )

                rows: list[tuple[str, int, bytes]] = []
                for i in range(0, len(chunk_id_text), embedding_batch_size):
                    batch = chunk_id_text[i : i + embedding_batch_size]
                    ids = [x[0] for x in batch]
                    raw_texts = [x[1] for x in batch]
                    try:
                        vectors = _embed_texts_resilient(
                            lmstudio_base_url,
                            embedding_model_id,
                            chunk_ids=ids,
                            raw_texts=raw_texts,
                            max_chars=embedding_max_chars,
                        )
                    except Exception as e:
                        max_len = max((len(t) for t in raw_texts), default=0)
                        raise RuntimeError(
                            f"Embedding request failed at batch {i//embedding_batch_size + 1} (max_chars={max_len}): {e}"
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
                    ("embedding_batch_size", str(int(embedding_batch_size))),
                    ("summary_enabled", "1" if summary_rows else "0"),
                    ("summary_model_id", str(summary_model or "")),
                    ("summary_units", str(int(summary_units or 0))),
                    ("summary_avgdl", f"{(summary_total_tokens / summary_units):.6f}" if summary_units else "0"),
                    ("summary_max_input_chars", str(int(summary_max_input_chars))),
                    ("summary_max_output_tokens", str(int(summary_max_output_tokens))),
                    ("summary_unit_max_tokens", str(int(summary_unit_max_tokens))),
                    ("chunk_max_chars_part", str(int(chunk_max_chars_part))),
                    ("chunk_max_chars_section", str(int(chunk_max_chars_section))),
                    ("chunk_max_chars_topic", str(int(chunk_max_chars_topic))),
                    ("chunk_granularity_scheme", "topic/section/part"),
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
