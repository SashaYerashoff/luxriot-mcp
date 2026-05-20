from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup, NavigableString, Tag
import httpx
from markdownify import markdownify as md


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_EMBED_H3_RE = re.compile(r"(?m)^([ \t]*)###(\s+)")
_CONTROL_RE = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_LIST_BULLET_RE = re.compile(r"^(\s*)[+*]\s+")
_LIST_NUM_RE = re.compile(r"^(\s*)(\d+)\)\s+")
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".svg"}


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


def _normal_path(text: str) -> str:
    return unquote(str(text or "")).replace("\\", "/").strip()


def _strip_windows_drive(path_text: str) -> str:
    out = _normal_path(path_text)
    if re.match(r"^/[A-Za-z]:/", out):
        out = out[1:]
    return out


def _image_src_local_path(src: str) -> str:
    raw = str(src or "").strip()
    if not raw:
        return ""
    if raw.startswith("file:"):
        parsed = urlparse(raw)
        return _strip_windows_drive(parsed.path or "")
    parsed = urlparse(raw)
    path = parsed.path or raw
    if path.startswith("./"):
        path = path[2:]
    return _normal_path(path).lstrip("/")


def _image_src_basename(src: str) -> str:
    local_path = _image_src_local_path(src)
    return local_path.rsplit("/", 1)[-1] if local_path else ""


def _doc_image_rel_candidates(src: str) -> list[str]:
    local_path = _image_src_local_path(src)
    if not local_path:
        return []

    candidates: list[str] = []
    if str(src or "").strip().startswith("file:"):
        lower = local_path.lower()
        images_idx = lower.rfind("/images/")
        if images_idx >= 0:
            candidates.append(local_path[images_idx + 1 :])
        basename = local_path.rsplit("/", 1)[-1]
        if basename:
            candidates.append(f"images/{basename}")
    else:
        candidates.append(local_path)

    out: list[str] = []
    seen: set[str] = set()
    for rel in candidates:
        rel = rel.strip().lstrip("/")
        if not rel or rel in seen:
            continue
        seen.add(rel)
        out.append(rel)
    return out


def resolve_doc_image_src(src: str, doc_dir: Path) -> tuple[Path, str] | None:
    doc_root = doc_dir.resolve()

    for rel in _doc_image_rel_candidates(src):
        src_path = (doc_dir / rel).resolve()
        try:
            src_path.relative_to(doc_root)
        except Exception:
            continue
        if src_path.exists() and src_path.is_file():
            return src_path, rel
    return None


def audit_docs_dir(docs_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "docs_dir": str(docs_dir),
        "doc_dirs": 0,
        "html_pages": 0,
        "image_refs": 0,
        "file_uri_refs": 0,
        "file_uri_image_refs": 0,
        "recoverable_file_uri_images": 0,
        "missing_file_uri_images": 0,
        "missing_relative_images": 0,
        "data_uri_images": 0,
        "http_images": 0,
        "empty_pages": 0,
        "broken_images": [],
        "warnings": [],
    }
    warnings: list[dict[str, Any]] = []
    broken_images: list[dict[str, Any]] = []

    doc_dirs = sorted([p for p in docs_dir.iterdir() if p.is_dir()]) if docs_dir.exists() else []
    report["doc_dirs"] = len(doc_dirs)

    def add_warning(
        kind: str,
        html_path: Path,
        detail: str,
        src: str | None = None,
        page_title: str | None = None,
    ) -> None:
        item: dict[str, Any] = {
            "kind": kind,
            "page": str(html_path.relative_to(docs_dir)),
            "detail": detail,
        }
        if page_title:
            item["page_title"] = page_title
        if src:
            item["src"] = src
            item["basename"] = _image_src_basename(src)
            item["candidate_paths"] = _doc_image_rel_candidates(src)
        warnings.append(item)
        if kind in {"missing_file_uri_image", "missing_relative_image"}:
            broken_images.append(item)

    for doc_dir in doc_dirs:
        html_files = sorted([p for p in doc_dir.glob("*.html") if not p.name.startswith("__")])
        for html_path in html_files:
            report["html_pages"] += 1
            html_text = read_text(html_path)
            file_refs = re.findall(r"file:///[^\s\"'<>)]*", html_text)
            report["file_uri_refs"] += len(file_refs)

            soup = BeautifulSoup(html_text, "html.parser")
            main = soup.select_one("#mainbody") or soup.body
            title_el = soup.select_one("#pagetitle")
            page_title = title_el.get_text(" ", strip=True) if title_el else html_path.stem
            text = main.get_text(" ", strip=True) if main is not None else ""
            if not text:
                report["empty_pages"] += 1
                add_warning("empty_page", html_path, "No visible text found in page body.", page_title=page_title)

            for img in soup.select("img[src]"):
                src = str(img.get("src") or "").strip()
                if not src:
                    continue
                report["image_refs"] += 1
                if src.startswith("data:"):
                    report["data_uri_images"] += 1
                    continue
                if src.startswith("http"):
                    report["http_images"] += 1
                    continue

                resolved = resolve_doc_image_src(src, doc_dir)
                if src.startswith("file:"):
                    report["file_uri_image_refs"] += 1
                    if resolved is None:
                        report["missing_file_uri_images"] += 1
                        add_warning(
                            "missing_file_uri_image",
                            html_path,
                            "File URI image is not present in local export.",
                            src,
                            page_title=page_title,
                        )
                    else:
                        report["recoverable_file_uri_images"] += 1
                    continue

                if resolved is None:
                    report["missing_relative_images"] += 1
                    add_warning(
                        "missing_relative_image",
                        html_path,
                        "Relative image is not present in local export.",
                        src,
                        page_title=page_title,
                    )

    by_basename: dict[str, dict[str, Any]] = {}
    for item in broken_images:
        basename = str(item.get("basename") or "").strip() or "(unknown)"
        bucket = by_basename.setdefault(basename, {"basename": basename, "count": 0, "pages": [], "kinds": []})
        bucket["count"] += 1
        page = str(item.get("page") or "")
        if page and page not in bucket["pages"]:
            bucket["pages"].append(page)
        kind = str(item.get("kind") or "")
        if kind and kind not in bucket["kinds"]:
            bucket["kinds"].append(kind)

    report["broken_images"] = broken_images
    report["broken_images_total"] = len(broken_images)
    report["broken_images_by_basename"] = sorted(
        by_basename.values(),
        key=lambda x: (-int(x.get("count") or 0), str(x.get("basename") or "").lower()),
    )
    report["warnings"] = warnings
    report["warnings_total"] = len(warnings)
    report["ok"] = (
        report["doc_dirs"] > 0
        and report["html_pages"] > 0
        and report["missing_file_uri_images"] == 0
        and report["missing_relative_images"] == 0
        and report["file_uri_refs"] == report["file_uri_image_refs"]
    )
    return report


def print_audit_report(report: dict[str, Any]) -> None:
    log("Ingest preflight audit:")
    for key in (
        "docs_dir",
        "doc_dirs",
        "html_pages",
        "image_refs",
        "file_uri_refs",
        "file_uri_image_refs",
        "recoverable_file_uri_images",
        "missing_file_uri_images",
        "missing_relative_images",
        "data_uri_images",
        "http_images",
        "empty_pages",
        "broken_images_total",
        "warnings_total",
    ):
        log(f"  {key}: {report.get(key)}")
    warnings = report.get("warnings") if isinstance(report.get("warnings"), list) else []
    for item in warnings[:20]:
        src = f" src={item.get('src')}" if item.get("src") else ""
        log(f"  WARNING {item.get('kind')}: {item.get('page')}: {item.get('detail')}{src}")
    if int(report.get("warnings_total") or 0) > 20:
        log(f"  ... {int(report.get('warnings_total') or 0) - 20} more warning(s)")


def _md_cell(value: Any, *, max_len: int = 180) -> str:
    text = str(value if value is not None else "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        text = text[: max(0, max_len - 1)].rstrip() + "..."
    return text.replace("|", "\\|")


def build_broken_images_markdown_report(report: dict[str, Any]) -> str:
    broken = report.get("broken_images") if isinstance(report.get("broken_images"), list) else []
    by_basename = (
        report.get("broken_images_by_basename")
        if isinstance(report.get("broken_images_by_basename"), list)
        else []
    )
    lines: list[str] = [
        "# Broken Image Report",
        "",
        f"- Docs dir: `{report.get('docs_dir')}`",
        f"- HTML pages: {int(report.get('html_pages') or 0)}",
        f"- Image refs: {int(report.get('image_refs') or 0)}",
        f"- Missing file URI images: {int(report.get('missing_file_uri_images') or 0)}",
        f"- Missing relative images: {int(report.get('missing_relative_images') or 0)}",
        f"- Broken images total: {int(report.get('broken_images_total') or 0)}",
        "",
    ]

    if not broken:
        lines.extend(["No broken images found.", ""])
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(
        [
            "## By Basename",
            "",
            "| Basename | Count | Pages |",
            "| --- | ---: | --- |",
        ]
    )
    for item in by_basename:
        pages = item.get("pages") if isinstance(item.get("pages"), list) else []
        page_text = "; ".join(str(p) for p in pages[:8])
        if len(pages) > 8:
            page_text += f"; +{len(pages) - 8} more"
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(item.get("basename"), max_len=80),
                    str(int(item.get("count") or 0)),
                    _md_cell(page_text, max_len=220),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Details",
            "",
            "| Kind | Basename | Page | Title | Original src | Candidate local paths |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for item in broken:
        candidates = item.get("candidate_paths") if isinstance(item.get("candidate_paths"), list) else []
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_cell(item.get("kind"), max_len=40),
                    _md_cell(item.get("basename"), max_len=80),
                    _md_cell(item.get("page"), max_len=120),
                    _md_cell(item.get("page_title"), max_len=120),
                    _md_cell(item.get("src"), max_len=240),
                    _md_cell("; ".join(str(c) for c in candidates), max_len=180),
                ]
            )
            + " |"
        )

    return "\n".join(lines).rstrip() + "\n"


def write_broken_images_markdown_report(report: dict[str, Any], path: Path) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_broken_images_markdown_report(report), encoding="utf-8")
    log(f"Broken image report written to {path}")


def _image_file_paths(source_dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for raw in source_dirs:
        root = raw.expanduser()
        if not root.exists():
            log(f"WARNING: repair image source not found: {root}")
            continue
        if root.is_file():
            if root.suffix.lower() in _IMAGE_EXTS:
                out.append(root.resolve())
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in _IMAGE_EXTS:
                out.append(path.resolve())
    return sorted(set(out), key=lambda p: str(p).lower())


def _build_image_source_index(source_dirs: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in _image_file_paths(source_dirs):
        index.setdefault(path.name.lower(), []).append(path)
    return index


def _source_match_score(src: str, candidate: Path, page: str) -> int:
    hint = _image_src_local_path(src).lower()
    hint_parts = [p for p in hint.split("/") if p]
    candidate_norm = _normal_path(str(candidate)).lower()
    score = 0
    max_suffix = min(5, len(hint_parts))
    for n in range(max_suffix, 1, -1):
        suffix = "/".join(hint_parts[-n:])
        if suffix and candidate_norm.endswith(suffix):
            score = max(score, 100 * n)
            break

    page_stem = Path(page).stem.lower().strip()
    if page_stem and page_stem in candidate_norm:
        score += 20
    return score


def _choose_repair_source(
    item: dict[str, Any],
    source_index: dict[str, list[Path]],
    *,
    allow_ambiguous: bool,
) -> tuple[str, Path | None, list[Path]]:
    basename = str(item.get("basename") or "").strip().lower()
    candidates = source_index.get(basename, [])
    if not candidates:
        return "not_found", None, []
    if len(candidates) == 1:
        return "matched", candidates[0], candidates

    scored = sorted(
        ((_source_match_score(str(item.get("src") or ""), c, str(item.get("page") or "")), c) for c in candidates),
        key=lambda x: (-x[0], str(x[1]).lower()),
    )
    best_score, best_path = scored[0]
    same_score = [p for score, p in scored if score == best_score]
    if best_score > 0 and len(same_score) == 1:
        return "matched", best_path, candidates
    if allow_ambiguous:
        return "matched_ambiguous", best_path, candidates
    return "ambiguous", None, candidates


def _safe_image_basename(name: str, src: str) -> str:
    basename = Path(str(name or "")).name.strip()
    if not basename:
        digest = hashlib.sha1(str(src or "").encode("utf-8")).hexdigest()[:10]
        basename = f"image-{digest}.png"
    return basename.replace("/", "_").replace("\\", "_")


def _repair_target_rel(item: dict[str, Any]) -> str:
    basename = _safe_image_basename(str(item.get("basename") or ""), str(item.get("src") or ""))
    kind = str(item.get("kind") or "")
    candidates = item.get("candidate_paths") if isinstance(item.get("candidate_paths"), list) else []
    if kind == "missing_relative_image" and candidates:
        rel = str(candidates[0] or "").strip().lstrip("/")
        if rel and ".." not in Path(rel).parts:
            return rel
    page_stem = slugify(Path(str(item.get("page") or "page")).stem)
    return f"images/_recovered/{page_stem}/{basename}"


def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _target_path_for_copy(doc_dir: Path, target_rel: str, source_path: Path) -> tuple[str, Path]:
    doc_root = doc_dir.resolve()
    target_rel = target_rel.strip().lstrip("/")
    target_path = (doc_dir / target_rel).resolve()
    try:
        target_path.relative_to(doc_root)
    except Exception as e:
        raise RuntimeError(f"Repair target escapes doc dir: {target_rel}") from e

    if not target_path.exists():
        return target_rel, target_path
    if target_path.is_file() and target_path.stat().st_size == source_path.stat().st_size:
        if _file_sha1(target_path) == _file_sha1(source_path):
            return target_rel, target_path

    suffix = source_path.suffix
    stem = Path(target_rel).with_suffix("").as_posix()
    digest = _file_sha1(source_path)[:10]
    unique_rel = f"{stem}-{digest}{suffix}"
    unique_path = (doc_dir / unique_rel).resolve()
    try:
        unique_path.relative_to(doc_root)
    except Exception as e:
        raise RuntimeError(f"Repair target escapes doc dir: {unique_rel}") from e
    return unique_rel, unique_path


def _replace_img_src(html_text: str, old_src: str, new_src: str) -> tuple[str, int]:
    variants = [old_src]
    amp = old_src.replace("&", "&amp;")
    if amp != old_src:
        variants.append(amp)
    for src in variants:
        pattern = re.compile(r"(\bsrc\s*=\s*)([\"'])" + re.escape(src) + r"\2")
        html_text, count = pattern.subn(lambda m: f"{m.group(1)}{m.group(2)}{new_src}{m.group(2)}", html_text)
        if count:
            return html_text, count
    return html_text, 0


def repair_missing_images(
    docs_dir: Path,
    source_dirs: list[Path],
    *,
    dry_run: bool,
    allow_ambiguous: bool,
) -> dict[str, Any]:
    source_index = _build_image_source_index(source_dirs)
    audit = audit_docs_dir(docs_dir)
    broken = audit.get("broken_images") if isinstance(audit.get("broken_images"), list) else []
    report: dict[str, Any] = {
        "docs_dir": str(docs_dir),
        "source_dirs": [str(p) for p in source_dirs],
        "source_images": sum(len(v) for v in source_index.values()),
        "dry_run": bool(dry_run),
        "allow_ambiguous": bool(allow_ambiguous),
        "total_broken": len(broken),
        "repaired": 0,
        "would_repair": 0,
        "not_found": 0,
        "ambiguous": 0,
        "errors": 0,
        "skipped_duplicates": 0,
        "actions": [],
    }
    actions: list[dict[str, Any]] = []
    seen_page_src: set[tuple[str, str]] = set()

    for item in broken:
        page_key = str(item.get("page") or "")
        src_key = str(item.get("src") or "")
        action: dict[str, Any] = {
            "page": page_key,
            "page_title": item.get("page_title"),
            "kind": item.get("kind"),
            "src": src_key,
            "basename": item.get("basename"),
        }
        duplicate_key = (page_key, src_key)
        if duplicate_key in seen_page_src:
            report["skipped_duplicates"] += 1
            action["status"] = "duplicate_ref"
            actions.append(action)
            continue
        seen_page_src.add(duplicate_key)

        status, source_path, candidates = _choose_repair_source(
            item,
            source_index,
            allow_ambiguous=allow_ambiguous,
        )
        action["status"] = status
        action["candidate_count"] = len(candidates)
        action["candidates"] = [str(p) for p in candidates[:20]]
        if source_path is not None:
            action["source"] = str(source_path)

        if status == "not_found":
            report["not_found"] += 1
            actions.append(action)
            continue
        if status == "ambiguous":
            report["ambiguous"] += 1
            actions.append(action)
            continue
        if source_path is None:
            report["errors"] += 1
            action["status"] = "error"
            action["error"] = "No source path selected"
            actions.append(action)
            continue

        try:
            page_rel = Path(str(item.get("page") or ""))
            html_path = (docs_dir / page_rel).resolve()
            doc_dir = html_path.parent
            target_rel = _repair_target_rel(item)
            target_rel, target_path = _target_path_for_copy(doc_dir, target_rel, source_path)
            action["target_rel"] = target_rel
            action["target"] = str(target_path)

            if dry_run:
                report["would_repair"] += 1
                action["status"] = "would_repair"
                actions.append(action)
                continue

            html_text = read_text(html_path)
            patched, count = _replace_img_src(html_text, str(item.get("src") or ""), target_rel)
            if count <= 0:
                raise RuntimeError("Could not replace img src in HTML")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if not target_path.exists():
                shutil.copy2(source_path, target_path)
            html_path.write_text(patched, encoding="utf-8")
            report["repaired"] += 1
            action["status"] = "repaired"
            action["replacements"] = count
        except Exception as e:
            report["errors"] += 1
            action["status"] = "error"
            action["error"] = str(e)
        actions.append(action)

    report["actions"] = actions
    if not dry_run:
        final_audit = audit_docs_dir(docs_dir)
        report["remaining_broken"] = int(final_audit.get("broken_images_total") or 0)
    report["ok"] = (
        int(report.get("errors") or 0) == 0
        and int(report.get("not_found") or 0) == 0
        and int(report.get("ambiguous") or 0) == 0
    )
    return report


def print_repair_report(report: dict[str, Any]) -> None:
    log("Image repair report:")
    for key in (
        "docs_dir",
        "source_images",
        "total_broken",
        "would_repair",
        "repaired",
        "remaining_broken",
        "not_found",
        "ambiguous",
        "errors",
        "skipped_duplicates",
    ):
        if key in report:
            log(f"  {key}: {report.get(key)}")
    actions = report.get("actions") if isinstance(report.get("actions"), list) else []
    for action in actions[:20]:
        status = action.get("status")
        page = action.get("page")
        src = action.get("src")
        source = action.get("source")
        extra = f" source={source}" if source else ""
        log(f"  {status}: {page}: {src}{extra}")
    if len(actions) > 20:
        log(f"  ... {len(actions) - 20} more action(s)")


def _pdf_title(pdf_path: Path) -> str:
    name = pdf_path.stem.replace("_", " ").replace("-", " ")
    name = re.sub(r"\s+", " ", name).strip()
    return name or "PDF export"


def pdf_text_to_markdown(text: str, title: str) -> str:
    src = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    src = src.replace("\f", "\n\n").replace("\u00a0", " ")
    src = src.replace("\u2022", "-")
    src = re.sub(r"[ \t]+\n", "\n", src)
    src = re.sub(r"\n{3,}", "\n\n", src)

    paragraphs = [p.strip() for p in src.split("\n\n") if p.strip()]
    out: list[str] = [f"# {title}"]
    last_heading = title.lower()

    for para in paragraphs:
        lines = [re.sub(r"\s+", " ", ln).strip() for ln in para.splitlines() if ln.strip()]
        if not lines:
            continue
        joined = " ".join(lines).strip()
        if not joined:
            continue

        words = joined.split()
        looks_like_heading = (
            len(joined) <= 90
            and len(words) <= 12
            and not joined.endswith((".", ",", ";"))
            and not joined.startswith(("-", "http://", "https://"))
        )
        if looks_like_heading and joined.lower() != last_heading:
            out.append(f"## {joined.rstrip(':')}")
            last_heading = joined.lower()
            continue

        normalized_lines: list[str] = []
        for line in lines:
            if line.startswith("- "):
                normalized_lines.append(line)
            elif line.startswith("-"):
                normalized_lines.append("- " + line.lstrip("- ").strip())
            else:
                normalized_lines.append(line)
        out.append("\n".join(normalized_lines))

    return "\n\n".join(out).strip() + "\n"


def extract_pdf_markdown(pdf_path: Path) -> str:
    exe = shutil.which("pdftotext")
    if not exe:
        raise RuntimeError("pdftotext is not installed")
    proc = subprocess.run(
        [exe, "-layout", "-enc", "UTF-8", "-nopgbrk", str(pdf_path), "-"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        msg = (proc.stderr or "").strip()
        raise RuntimeError(msg or f"pdftotext exited with code {proc.returncode}")
    return pdf_text_to_markdown(proc.stdout, _pdf_title(pdf_path))


def write_pdf_markdown_sidecars(docs_dir: Path, out_dir: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "docs_dir": str(docs_dir),
        "out_dir": str(out_dir),
        "pdf_files": 0,
        "written": 0,
        "errors": [],
    }
    errors: list[dict[str, str]] = []
    pdf_files = sorted([p for p in docs_dir.rglob("*.pdf") if p.is_file()]) if docs_dir.exists() else []
    report["pdf_files"] = len(pdf_files)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_files:
        try:
            rel = pdf_path.relative_to(docs_dir)
        except ValueError:
            rel = Path(pdf_path.name)
        try:
            md_text = extract_pdf_markdown(pdf_path)
            md_rel = rel.with_suffix(rel.suffix + ".md")
            md_path = out_dir / md_rel
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(md_text, encoding="utf-8")
            report["written"] += 1
            log(f"PDF markdown sidecar: {pdf_path} -> {md_path}")
        except Exception as e:
            errors.append({"pdf": str(pdf_path), "error": str(e)})
            log(f"WARNING: failed to extract PDF markdown from {pdf_path}: {e}")

    report["errors"] = errors
    report["ok"] = len(errors) == 0
    return report


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

    # Convert Help+Manual box tables into admonition-style blockquotes.
    for box in main.select("table.hs-box"):
        icon = box.select_one("td.hs-box-icon img")
        icon_src = icon.get("src", "") if icon else ""
        kind = classify_hs_box(str(icon_src)).upper()

        content_td = box.select_one("td.hs-box-content")
        block = soup.new_tag("blockquote")
        head = soup.new_tag("p")
        head.string = f"[!{kind}]"
        block.append(head)
        if content_td is not None:
            # Keep original inline markup inside the note box. Using get_text() with a newline
            # separator would split inline spans (e.g., bold words) into separate lines, which
            # then breaks chunking and hurts retrieval quality.
            block_tags = {
                "p",
                "div",
                "ul",
                "ol",
                "table",
                "pre",
                "blockquote",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "dl",
            }
            has_block = any(isinstance(c, Tag) and c.name in block_tags for c in content_td.contents)
            moved = False
            if has_block:
                for child in list(content_td.contents):
                    if isinstance(child, NavigableString):
                        text = str(child).strip()
                        if not text:
                            continue
                        p = soup.new_tag("p")
                        p.string = text
                        block.append(p)
                        moved = True
                        continue
                    block.append(child.extract())
                    moved = True
            else:
                p = soup.new_tag("p")
                for child in list(content_td.contents):
                    if isinstance(child, NavigableString):
                        p.append(str(child))
                        child.extract()
                        continue
                    p.append(child.extract())
                if p.get_text(" ", strip=True):
                    block.append(p)
                    moved = True
            if not moved:
                text = content_td.get_text(" ", strip=True)
                if text:
                    p = soup.new_tag("p")
                    p.string = text
                    block.append(p)
        else:
            text = box.get_text(" ", strip=True)
            if text:
                p = soup.new_tag("p")
                p.string = text
                block.append(p)
        box.replace_with(block)

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
        resolved = resolve_doc_image_src(src, doc_dir)
        if resolved is None:
            if src.startswith("file:"):
                log(f"WARNING: missing file URI image in {doc_id}/{page_title}: {src}")
            img.decompose()
            continue
        src_path, rel_src = resolved

        out_path = (assets_out_dir / doc_id / rel_src).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, out_path)

        url = f"/assets/{version}/{doc_id}/{rel_src}"
        images.append({"original": rel_src, "url": url, "alt": str(img.get("alt") or "").strip()})
        img["src"] = url

    md_body = md(str(main), heading_style="ATX", bullets="*")
    md_body = md_body.replace("\u00a0", " ")
    md_page = f"# {page_title}\n\n{md_body}".strip() + "\n"
    md_page = re.sub(r"\n{3,}", "\n\n", md_page)
    return md_page, images


def markdown_images(md_text: str) -> list[dict[str, str]]:
    images: list[dict[str, str]] = []
    for alt, url in _MD_IMAGE_RE.findall(md_text or ""):
        clean = str(url or "").strip()
        if not clean:
            continue
        images.append({"original": clean, "url": clean, "alt": str(alt or "").strip()})
    return images


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_markdown(md_text: str) -> str:
    src = (md_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = src.split("\n")
    out: list[str] = []
    in_code = False
    seen_h1 = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out.append(line.rstrip())
            continue
        if in_code:
            out.append(line.rstrip())
            continue

        match = _HEADING_RE.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            if level == 1:
                if not seen_h1:
                    seen_h1 = True
                else:
                    level = 2
            if level > 5:
                level = 5
            out.append("#" * level + " " + title)
            continue

        line = _LIST_BULLET_RE.sub(r"\1- ", line)
        line = _LIST_NUM_RE.sub(r"\1\2. ", line)
        out.append(line.rstrip())

    collapsed: list[str] = []
    blank_run = 0
    for line in out:
        if not line.strip():
            blank_run += 1
            if blank_run > 1:
                continue
        else:
            blank_run = 0
        collapsed.append(line)

    normalized = "\n".join(collapsed).strip()
    return normalized + "\n" if normalized else ""


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


class EmbeddingModelLoadError(RuntimeError):
    pass


def _is_model_load_error(message: str) -> bool:
    low = str(message or "").lower()
    return (
        "failed to load model" in low
        or "model has unloaded" in low
        or "unloaded or crashed" in low
        or "model has unloaded or crashed" in low
    )


def _list_embedding_models(base_url: str) -> list[str]:
    resp = httpx.get(f"{base_url}/v1/models", timeout=10.0)
    resp.raise_for_status()
    data = resp.json()
    models = data.get("data") or []
    out: list[str] = []
    for m in models:
        mid = str(m.get("id") or "").strip()
        if not mid:
            continue
        low = mid.lower()
        if "embedding" in low or low.startswith("text-embedding"):
            out.append(mid)
    return out


def _detect_embedding_model_id(base_url: str) -> str:
    models = _list_embedding_models(base_url)
    if models:
        return models[0]
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
        msg = f"Embeddings HTTP {resp.status_code}: {body[:1000]}"
        if _is_model_load_error(body):
            raise EmbeddingModelLoadError(msg)
        raise RuntimeError(msg)

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


def _select_embedding_model(base_url: str, preferred: str | None) -> str:
    preferred = str(preferred or "").strip() or None
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    try:
        for mid in _list_embedding_models(base_url):
            if mid not in candidates:
                candidates.append(mid)
    except Exception as e:
        if preferred:
            raise RuntimeError(f"Failed to list embedding models from LM Studio: {e}") from e
        raise

    if not candidates:
        raise RuntimeError("No embedding models available in LM Studio. Load one and retry.")

    probe_text = "Embedding probe."
    last_error: Exception | None = None
    for mid in candidates:
        try:
            _embed_texts(base_url, mid, [probe_text])
            if mid != preferred:
                log(f"Embedding model fallback selected: {mid}")
            return mid
        except EmbeddingModelLoadError as e:
            last_error = e
            log(f"Embedding model '{mid}' failed to load; trying next. {e}")
            continue
        except Exception as e:
            last_error = e
            raise RuntimeError(f"Embedding model '{mid}' failed during probe: {e}") from e

    if last_error:
        raise RuntimeError(f"No embedding model could be loaded. Last error: {last_error}") from last_error
    raise RuntimeError("No embedding model could be loaded.")


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


SUMMARY_MAX_INPUT_CHARS_MIN = 500
SUMMARY_MAX_INPUT_CHARS_MAX = 8000
SUMMARY_MAX_OUTPUT_TOKENS_MIN = 32
SUMMARY_MAX_OUTPUT_TOKENS_MAX = 512
SUMMARY_UNIT_MAX_TOKENS_MIN = 200
SUMMARY_UNIT_MAX_TOKENS_MAX = 1600


def _clamp_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = int(default)
    return max(min_value, min(max_value, n))


def _clamp_summary_arg(name: str, value: int, default: int, min_value: int, max_value: int) -> int:
    clamped = _clamp_int(value, default, min_value, max_value)
    if clamped != int(value):
        log(f"WARNING: {name} clamped from {value} to {clamped}")
    return clamped


def _extractive_summary_text(text: str, heading_path: list[str], max_chars: int) -> str:
    """Deterministic page-router summary used when the LLM summarizer is unavailable."""
    max_chars = max(200, int(max_chars or 1200))
    lines = _strip_markdown_images(str(text or "").splitlines())
    out: list[str] = []
    heading = " > ".join(str(x).strip() for x in heading_path if str(x).strip())
    if heading:
        out.append(heading)

    for raw in lines:
        line = str(raw or "").strip()
        if not line:
            continue
        line = re.sub(r"^#{1,6}\s+", "", line).strip()
        line = re.sub(r"^>\s*", "", line).strip()
        line = line.replace("**", "").replace("*", "")
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if set(line) <= set("-:| "):
            continue
        if line not in out:
            out.append(line)
        if len(" ".join(out)) >= max_chars:
            break

    summary = " ".join(out).strip()
    if len(summary) > max_chars:
        cut = summary.rfind(" ", 0, max_chars)
        summary = summary[: cut if cut > int(max_chars * 0.6) else max_chars].rstrip()
    return summary


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
    except EmbeddingModelLoadError:
        raise
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
    ap.add_argument(
        "--from-datastore",
        type=Path,
        default=None,
        help="Use existing datastore as input (skip HTML conversion)",
    )
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
    ap.add_argument("--summary-failure-limit", type=int, default=3, help="Disable LLM summaries after this many failures (0 = never disable)")
    ap.add_argument("--summary-fallback-max-chars", type=int, default=1200, help="Max chars for extractive summary fallback")
    ap.add_argument("--summary-no-fallback", action="store_true", help="Disable deterministic extractive summaries when LLM summaries fail")
    ap.add_argument("--chunk-max-chars-part", type=int, default=900, help="Max chars per fine-grained chunk (subsection)")
    ap.add_argument("--chunk-max-chars-section", type=int, default=2600, help="Max chars per section chunk")
    ap.add_argument("--chunk-max-chars-topic", type=int, default=5200, help="Max chars per topic chunk")
    ap.add_argument("--lmstudio-base-url", type=str, default="http://localhost:1234", help="LM Studio base URL")
    ap.add_argument(
        "--md-conventions-file",
        type=Path,
        default=None,
        help="Markdown conventions file (used to normalize HTML ingestion and recorded in metadata)",
    )
    ap.add_argument("--embedding-model", type=str, default="", help="Embedding model id (defaults to first embedding model from /v1/models)")
    ap.add_argument("--embedding-max-chars", type=int, default=448, help="Max characters per chunk sent for embedding")
    ap.add_argument("--embedding-batch-size", type=int, default=8, help="How many chunks to embed per request (lower = more stable)")
    ap.add_argument("--no-embeddings", action="store_true", help="Skip computing embeddings")
    ap.add_argument("--audit", action="store_true", help="Run HTML export preflight audit before ingesting")
    ap.add_argument("--audit-only", action="store_true", help="Run HTML export preflight audit and exit")
    ap.add_argument("--audit-json", type=Path, default=None, help="Write full preflight audit report to this JSON file")
    ap.add_argument("--broken-images-report", type=Path, default=None, help="Write broken image report as Markdown")
    ap.add_argument("--strict-assets", action="store_true", help="Abort ingest when preflight audit finds missing/local file assets")
    ap.add_argument(
        "--repair-image-source",
        type=Path,
        action="append",
        default=[],
        help="Directory or image file to search when repairing missing HTML image refs (can be repeated)",
    )
    ap.add_argument("--repair-images", action="store_true", help="Repair missing HTML image refs before ingesting")
    ap.add_argument("--repair-images-only", action="store_true", help="Repair missing HTML image refs and exit")
    ap.add_argument("--repair-images-dry-run", action="store_true", help="Report image repairs without writing files")
    ap.add_argument("--repair-allow-ambiguous", action="store_true", help="Allow basename-only repair when multiple candidates match")
    ap.add_argument("--repair-report-json", type=Path, default=None, help="Write image repair report to this JSON file")
    ap.add_argument(
        "--pdf-md-dir",
        type=Path,
        default=None,
        help="Write PDF->Markdown diagnostic sidecars to this directory (not indexed)",
    )
    ap.add_argument("--pdf-md-only", action="store_true", help="Write PDF->Markdown sidecars and exit")
    ap.add_argument("--clean", action="store_true", help="Delete existing out-dir before ingesting")
    args = ap.parse_args()

    docs_dir: Path = args.docs_dir
    source_store: Path | None = args.from_datastore
    out_dir: Path = args.out_dir
    version: str = args.version
    lmstudio_base_url: str = str(args.lmstudio_base_url).rstrip("/")
    embedding_model: str = str(args.embedding_model).strip()
    embedding_max_chars: int = int(args.embedding_max_chars)
    embedding_batch_size: int = max(1, int(args.embedding_batch_size))
    compute_embeddings: bool = not bool(args.no_embeddings)
    app_db_path: Path = Path(args.app_db)
    include_edits: bool = bool(args.include_edits)
    md_conventions_file: Path | None = args.md_conventions_file
    md_conventions_text = ""
    md_conventions_hash: str | None = None
    summary_enabled: bool = bool(args.summary_enabled)
    summary_model: str = str(args.summary_model or "").strip()
    summary_max_input_chars: int = _clamp_summary_arg(
        "--summary-max-input-chars",
        int(args.summary_max_input_chars),
        6000,
        SUMMARY_MAX_INPUT_CHARS_MIN,
        SUMMARY_MAX_INPUT_CHARS_MAX,
    )
    summary_max_output_tokens: int = _clamp_summary_arg(
        "--summary-max-output-tokens",
        int(args.summary_max_output_tokens),
        280,
        SUMMARY_MAX_OUTPUT_TOKENS_MIN,
        SUMMARY_MAX_OUTPUT_TOKENS_MAX,
    )
    summary_unit_max_tokens: int = _clamp_summary_arg(
        "--summary-unit-max-tokens",
        int(args.summary_unit_max_tokens),
        900,
        SUMMARY_UNIT_MAX_TOKENS_MIN,
        SUMMARY_UNIT_MAX_TOKENS_MAX,
    )
    summary_failure_limit: int = max(0, int(args.summary_failure_limit))
    summary_fallback_max_chars: int = int(args.summary_fallback_max_chars)
    summary_fallback_enabled: bool = bool(summary_enabled and not args.summary_no_fallback)
    chunk_max_chars_part: int = int(args.chunk_max_chars_part)
    chunk_max_chars_section: int = int(args.chunk_max_chars_section)
    chunk_max_chars_topic: int = int(args.chunk_max_chars_topic)
    audit_json_path: Path | None = args.audit_json
    broken_images_report_path: Path | None = args.broken_images_report
    repair_report_json_path: Path | None = args.repair_report_json
    pdf_md_dir: Path | None = args.pdf_md_dir

    if source_store is not None:
        source_store = source_store.expanduser()
        if not source_store.exists():
            log(f"ERROR: source datastore not found: {source_store}")
            return 2
    else:
        if not docs_dir.exists():
            log(f"ERROR: docs dir not found: {docs_dir}")
            return 2

    repair_requested = bool(args.repair_images or args.repair_images_only or repair_report_json_path)
    if repair_requested:
        repair_sources = [Path(p).expanduser() for p in (args.repair_image_source or [])]
        if not repair_sources:
            log("ERROR: --repair-images requires at least one --repair-image-source")
            return 2
        if source_store is not None and not args.repair_images_only:
            log("WARNING: image repair edits --docs-dir while ingest input comes from --from-datastore.")
        repair_report = repair_missing_images(
            docs_dir,
            repair_sources,
            dry_run=bool(args.repair_images_dry_run),
            allow_ambiguous=bool(args.repair_allow_ambiguous),
        )
        print_repair_report(repair_report)
        if repair_report_json_path is not None:
            repair_report_json_path = repair_report_json_path.expanduser()
            repair_report_json_path.parent.mkdir(parents=True, exist_ok=True)
            repair_report_json_path.write_text(json.dumps(repair_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            log(f"Image repair JSON written to {repair_report_json_path}")
        if args.repair_images_only:
            return 0 if bool(repair_report.get("ok")) else 1

    audit_requested = bool(
        args.audit
        or args.audit_only
        or args.strict_assets
        or audit_json_path
        or broken_images_report_path
    )
    if audit_requested:
        if not docs_dir.exists():
            log(f"ERROR: docs dir not found for preflight audit: {docs_dir}")
            return 2
        if source_store is not None and not args.audit_only:
            log("WARNING: HTML preflight audit uses --docs-dir while ingest input comes from --from-datastore.")
        audit_report = audit_docs_dir(docs_dir)
        print_audit_report(audit_report)
        if audit_json_path is not None:
            audit_json_path = audit_json_path.expanduser()
            audit_json_path.parent.mkdir(parents=True, exist_ok=True)
            audit_json_path.write_text(json.dumps(audit_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            log(f"Audit JSON written to {audit_json_path}")
        if broken_images_report_path is not None:
            write_broken_images_markdown_report(audit_report, broken_images_report_path)
        audit_ok = bool(audit_report.get("ok"))
        if args.audit_only:
            return 0 if audit_ok else 1
        if args.strict_assets and not audit_ok:
            log("ERROR: preflight audit failed; fix missing/local file assets or rerun without --strict-assets.")
            return 2

    if args.pdf_md_only:
        if not docs_dir.exists():
            log(f"ERROR: docs dir not found for PDF markdown extraction: {docs_dir}")
            return 2
        pdf_sidecar_dir = pdf_md_dir or (out_dir / "pdf_markdown")
        pdf_report = write_pdf_markdown_sidecars(docs_dir, pdf_sidecar_dir)
        log(
            f"PDF markdown sidecars: {pdf_report.get('written')}/{pdf_report.get('pdf_files')} "
            f"written to {pdf_sidecar_dir}"
        )
        return 0 if bool(pdf_report.get("ok")) else 1

    if md_conventions_file is not None:
        md_conventions_file = md_conventions_file.expanduser()
        if not md_conventions_file.exists():
            log(f"ERROR: markdown conventions file not found: {md_conventions_file}")
            return 2
        md_conventions_text = read_text(md_conventions_file)
        if md_conventions_text.strip():
            md_conventions_hash = _hash_text(md_conventions_text)

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

    if pdf_md_dir is not None:
        if not docs_dir.exists():
            log(f"ERROR: docs dir not found for PDF markdown extraction: {docs_dir}")
            return 2
        pdf_report = write_pdf_markdown_sidecars(docs_dir, pdf_md_dir)
        log(
            f"PDF markdown sidecars: {pdf_report.get('written')}/{pdf_report.get('pdf_files')} "
            f"written to {pdf_md_dir}"
        )

    index_path = out_dir / "index.sqlite"
    conn = init_index(index_path)

    pages_jsonl_path = out_dir / "pages.jsonl"
    pages_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    pages_jsonl = pages_jsonl_path.open("w", encoding="utf-8")

    if source_store is not None:
        source_assets = source_store / "assets"
        if source_assets.exists():
            shutil.copytree(source_assets, assets_out, dirs_exist_ok=True)
        else:
            log(f"WARNING: source assets not found at {source_assets}")
        doc_dirs: list[Path] = []
    else:
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
    summary_active = summary_enabled and (bool(summary_model) or summary_fallback_enabled)
    summary_llm_active = summary_enabled and bool(summary_model)
    summary_failed = False
    summary_llm_units = 0
    summary_fallback_units = 0
    summary_failed_units = 0
    if summary_enabled and not summary_model:
        if summary_fallback_enabled:
            log("WARNING: summary enabled but --summary-model is empty; using extractive summaries.")
        else:
            log("WARNING: summary enabled but --summary-model is empty; skipping summary index.")
            summary_active = False

    published_edits: dict[tuple[str, str], str] = {}
    custom_pages: list[dict[str, Any]] = []
    if include_edits:
        published_edits = _load_published_edits(app_db_path, version)
        custom_pages = _load_custom_pages(app_db_path, version)

    def add_summary_entries(
        *,
        doc_id: str,
        page_id: str,
        doc_title: str,
        page_title: str,
        md_text: str,
        default_heading_path: list[str],
        source_path: str,
        anchor: str | None,
    ) -> None:
        nonlocal summary_active
        nonlocal summary_llm_active
        nonlocal summary_failed
        nonlocal summary_units
        nonlocal summary_total_tokens
        nonlocal summary_llm_units
        nonlocal summary_fallback_units
        nonlocal summary_failed_units

        if not summary_active:
            return

        sections = split_markdown_for_summary(
            md_text,
            doc_title=doc_title,
            page_title=page_title,
            unit_max_tokens=summary_unit_max_tokens,
        )
        if sections:
            mode = "llm" if summary_llm_active else "extractive"
            log(f"  Summarizing {doc_id}/{page_id} ({len(sections)} sections, mode={mode})...")

        for s_idx, sec in enumerate(sections):
            raw_text = str(sec.get("text") or "").strip()
            if not raw_text:
                continue
            sec_heading_path = sec.get("heading_path") or default_heading_path or [doc_title, page_title]
            summary_text = ""
            used_llm = False

            if summary_llm_active:
                summary_input = raw_text[:summary_max_input_chars] if summary_max_input_chars > 0 else raw_text
                try:
                    summary_text = _chat_completion(
                        lmstudio_base_url,
                        summary_model,
                        summary_input,
                        max_tokens=summary_max_output_tokens,
                    ).strip()
                    used_llm = True
                except Exception as e:
                    summary_failed = True
                    summary_failed_units += 1
                    log(f"WARNING: summary failed for {doc_id}/{page_id}: {e}")
                    if summary_failure_limit > 0 and summary_failed_units >= summary_failure_limit:
                        summary_llm_active = False
                        log(
                            "WARNING: LLM summaries disabled after "
                            f"{summary_failed_units} failure(s); using extractive fallback."
                        )

            if not summary_text and summary_fallback_enabled:
                summary_text = _extractive_summary_text(
                    raw_text,
                    [str(x) for x in sec_heading_path],
                    summary_fallback_max_chars,
                )
                if summary_text:
                    summary_fallback_units += 1
            elif used_llm:
                summary_llm_units += 1

            if not summary_text:
                continue

            summary_id = f"{doc_id}:{page_id}:s{s_idx:03d}"
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

    seen_pages: set[tuple[str, str]] = set()

    if source_store is not None:
        source_pages = source_store / "pages"
        source_pages_jsonl = source_store / "pages.jsonl"
        if not source_pages_jsonl.exists():
            log(f"ERROR: source pages.jsonl not found at {source_pages_jsonl}")
            return 2
        log(f"Indexing from datastore: {source_store}")
        with source_pages_jsonl.open("r", encoding="utf-8") as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = str(entry.get("doc_id") or "").strip()
                page_id = str(entry.get("page_id") or "").strip()
                if not doc_id or not page_id:
                    continue
                doc_title = str(entry.get("doc_title") or doc_id)
                page_title = str(entry.get("page_title") or page_id)
                heading_path = entry.get("heading_path") or [doc_title, page_title]
                source_path = str(entry.get("source_path") or f"{doc_id}/{page_id}.md")
                anchor = str(entry.get("anchor") or "pagetitle")
                md_hash = entry.get("md_conventions_hash")
                md_hash = str(md_hash).strip() if isinstance(md_hash, str) and str(md_hash).strip() else None
                md_in_path = source_pages / doc_id / f"{page_id}.md"
                if not md_in_path.exists():
                    log(f"WARNING: missing markdown {md_in_path}")
                    continue
                md_text = read_text(md_in_path)
                edit_text = published_edits.get((doc_id, page_id))
                if edit_text:
                    md_text = edit_text
                images = markdown_images(md_text)

                add_summary_entries(
                    doc_id=doc_id,
                    page_id=page_id,
                    doc_title=doc_title,
                    page_title=page_title,
                    md_text=md_text,
                    default_heading_path=list(heading_path or [doc_title, page_title]),
                    source_path=source_path,
                    anchor=anchor,
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
                    "source_path": source_path,
                    "anchor": anchor,
                    "images": images,
                    "markdown_path": str(md_path.relative_to(out_dir)),
                    "md_conventions_hash": md_hash,
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
                    granularity_counts[granularity] += 1
                    chunk_idx = granularity_counts[granularity]
                    chunk_id = f"{doc_id}:{page_id}:{granularity_prefix.get(granularity, 'p')}{chunk_idx:03d}"
                    text = str(ch.get("text") or "").strip()
                    if not text:
                        continue
                    tokens = tokenize(text)
                    dl = len(tokens)
                    if dl == 0:
                        continue
                    n_chunks += 1
                    total_tokens += dl
                    tf = Counter(tokens)
                    for term, term_tf in tf.items():
                        postings_rows.append((term, chunk_id, int(term_tf)))
                    for term in tf.keys():
                        df_counter[term] += 1

                    images_urls = [u for u in ch.get("images") or [] if str(u).startswith("/assets/")]

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
            md_hash = md_conventions_hash
            if md_conventions_text.strip():
                md_text = normalize_markdown(md_text)
            edit_text = published_edits.get((doc_id, page_id))
            if edit_text:
                md_text = edit_text

            add_summary_entries(
                doc_id=doc_id,
                page_id=page_id,
                doc_title=doc_title,
                page_title=page_title,
                md_text=md_text,
                default_heading_path=list(heading_path or [doc_title, page_title]),
                source_path=html_rel,
                anchor="pagetitle",
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
                "md_conventions_hash": md_hash,
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
        if (doc_id, page_id) not in published_edits:
            log(f"  Skipping unpublished custom page {doc_id}/{page_id}")
            continue
        doc_title = custom["doc_title"]
        page_title = custom["page_title"]
        heading_path = custom["heading_path"]
        source_path = custom["source_path"]
        anchor = custom["anchor"]
        md_text = published_edits.get((doc_id, page_id)) or custom["base_markdown"]
        if not md_text.strip():
            md_text = f"# {page_title}\n\n"

        add_summary_entries(
            doc_id=doc_id,
            page_id=page_id,
            doc_title=doc_title,
            page_title=page_title,
            md_text=md_text,
            default_heading_path=list(heading_path or [doc_title, page_title]),
            source_path=source_path,
            anchor=anchor,
        )

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

    if summary_failed and not summary_fallback_enabled:
        summary_rows = []
        summary_postings_rows = []
        summary_df_counter = Counter()
        summary_units = 0
        summary_total_tokens = 0
        summary_llm_units = 0
        summary_fallback_units = 0
        log("WARNING: summary index discarded after LLM failure because extractive fallback is disabled.")

    pages_jsonl.close()

    if n_chunks == 0:
        log("ERROR: no chunks were produced; check docs input.")
        return 2

    avgdl = total_tokens / n_chunks
    log(f"Indexing {n_chunks} chunks (avgdl={avgdl:.2f}) ...")
    if summary_enabled:
        log(
            "Summary units: "
            f"{summary_units} total, {summary_llm_units} llm, "
            f"{summary_fallback_units} extractive, {summary_failed_units} llm failure(s)."
        )

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
                embedding_model_id = _select_embedding_model(lmstudio_base_url, embedding_model or None)
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
                    ("summary_failure_limit", str(int(summary_failure_limit))),
                    ("summary_llm_failures", str(int(summary_failed_units))),
                    ("summary_llm_units", str(int(summary_llm_units))),
                    ("summary_fallback_enabled", "1" if summary_fallback_enabled else "0"),
                    ("summary_fallback_units", str(int(summary_fallback_units))),
                    ("summary_fallback_max_chars", str(int(summary_fallback_max_chars))),
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
