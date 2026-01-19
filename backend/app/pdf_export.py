from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from fpdf import FPDF
from PIL import Image
from markdown_it import MarkdownIt
from mdit_py_plugins.container import container_plugin

from .config import DATASTORE_DIR, REPO_ROOT

_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_IMG_WIDTH_RE = re.compile(r"^(?P<value>\d+(?:\.\d+)?)(?P<unit>%|px|mm|cm)?$", re.IGNORECASE)
_FRONT_MATTER_KEY_RE = re.compile(r"^([A-Za-z0-9_-]+):\s*(.*)$")
_FRONT_MATTER_CHILD_RE = re.compile(r"^\s+([A-Za-z0-9_-]+):\s*(.*)$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+(.*)$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1")
_ITALIC_RE = re.compile(r"(\*|_)([^*_]+?)\1")
_CODE_RE = re.compile(r"`([^`]+)`")
_ADMONITION_RE = re.compile(r"^\[!(TIP|INFO|WARNING)\]\s*(.*)$", re.IGNORECASE)
_TOKEN_SPLIT_RE = re.compile(r"(\s+)")
_SECTION_MARKER_RE = re.compile(r"\[\[\s*DOC_SECTION\s*:\s*(.*?)\s*\]\]", re.IGNORECASE)
_PDF_FONT_MAP = {
    "sans": "Helvetica",
    "grotesk": "Helvetica",
    "serif": "Times",
    "slab": "Times",
    "mono": "Courier",
}
_CUSTOM_FONTS = {
    "titillium": {
        "family": "TitilliumWeb",
        "files": {
            "": "TitilliumWeb-Regular.ttf",
            "B": "TitilliumWeb-Bold.ttf",
            "I": "TitilliumWeb-Italic.ttf",
            "BI": "TitilliumWeb-BoldItalic.ttf",
        },
    },
    "unicode": {
        "family": "DejaVuSans",
        "files": {
            "": "DejaVuSans.ttf",
            "B": "DejaVuSans-Bold.ttf",
            "I": "DejaVuSans-Oblique.ttf",
            "BI": "DejaVuSans-BoldOblique.ttf",
        },
    },
}
_FONT_DIR = REPO_ROOT / "backend" / "assets" / "fonts"
_ASCII_REPLACEMENTS = {
    "\u00a0": " ",
    "\u2007": " ",
    "\u2009": " ",
    "\u200b": "",
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "--",
    "\u2015": "--",
    "\u2212": "-",
    "\u2026": "...",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": "\"",
    "\u201d": "\"",
    "\u00b7": "*",
    "\u2022": "*",
    "\u2190": "<-",
    "\u2192": "->",
    "\u2194": "<->",
    "\u21d0": "<=",
    "\u21d2": "=>",
    "\u21d4": "<=>",
}
_SPECIAL_SYMBOLS = {
    "\u2190",
    "\u2192",
    "\u2194",
    "\u21d0",
    "\u21d2",
    "\u21d4",
    "\u2022",
}
_SPECIAL_SYMBOL_SEQUENCES = {
    "<=>": "\u21d4",
    "<->": "\u2194",
    "=>": "\u21d2",
    "<=": "\u21d0",
    "->": "\u2192",
    "<-": "\u2190",
}


def _replace_symbol_sequences(text: str) -> str:
    if not text:
        return text
    out = str(text)
    for key in sorted(_SPECIAL_SYMBOL_SEQUENCES.keys(), key=len, reverse=True):
        out = out.replace(key, _SPECIAL_SYMBOL_SEQUENCES[key])
    return out


def _needs_unicode(text: str) -> bool:
    if not text:
        return False
    for ch in text:
        if ord(ch) > 0x00FF or ch in _SPECIAL_SYMBOLS:
            return True
    return False


def _normalize_ascii(text: str) -> str:
    out = text
    for key, val in _ASCII_REPLACEMENTS.items():
        out = out.replace(key, val)
    return out


def _sanitize_pdf_text(text: str, *, allow_unicode: bool) -> str:
    if allow_unicode:
        return text
    cleaned = _normalize_ascii(text)
    return cleaned.encode("latin-1", "replace").decode("latin-1")


def _normalize_admonitions(markdown: str) -> str:
    lines = str(markdown or "").splitlines()
    out: list[str] = []
    in_code = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out.append(line)
            i += 1
            continue
        if in_code:
            out.append(line)
            i += 1
            continue
        match = re.match(r"^\s*>\s*\[!(TIP|INFO|WARNING|NOTE)\]\s*(.*)$", line, re.IGNORECASE)
        if not match:
            out.append(line)
            i += 1
            continue
        kind = match.group(1).lower()
        title = (match.group(2) or "").strip()
        out.append(f"::: {kind}{(' ' + title) if title else ''}")
        i += 1
        while i < len(lines):
            nxt = lines[i]
            if re.match(r"^\s*>\s?", nxt):
                out.append(re.sub(r"^\s*>\s?", "", nxt))
                i += 1
                continue
            break
        out.append(":::")
    return "\n".join(out)


def _build_markdown_parser() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": False, "breaks": True})
    md.enable("table")
    for kind in ("tip", "info", "warning", "note"):
        md.use(container_plugin, kind)
    return md


_MD_PARSER: MarkdownIt | None = None


def _get_markdown_parser() -> MarkdownIt:
    global _MD_PARSER
    if _MD_PARSER is None:
        _MD_PARSER = _build_markdown_parser()
    return _MD_PARSER


def _inline_text(token, *, keep_linebreaks: bool = False, skip_images: bool = False) -> str:
    if not token:
        return ""
    if token.type != "inline":
        return token.content or ""
    if not token.children:
        return token.content or ""
    parts: list[str] = []
    for child in token.children:
        if child.type == "text":
            parts.append(child.content)
        elif child.type == "code_inline":
            parts.append(child.content)
        elif child.type == "softbreak":
            parts.append("\n" if keep_linebreaks else " ")
        elif child.type == "hardbreak":
            parts.append("\n")
        elif child.type == "image":
            if skip_images:
                continue
            parts.append(child.content or "")
    return "".join(parts)


def _attrs_to_dict(attrs: object | None) -> dict[str, str]:
    if not attrs:
        return {}
    if isinstance(attrs, dict):
        return {str(k): "" if v is None else str(v) for k, v in attrs.items()}
    out: dict[str, str] = {}
    try:
        for item in attrs:  # type: ignore[assignment]
            if not item:
                continue
            if isinstance(item, (list, tuple)):
                key = item[0] if len(item) > 0 else None
                val = item[1] if len(item) > 1 else ""
            else:
                continue
            if key is None:
                continue
            out[str(key)] = "" if val is None else str(val)
    except TypeError:
        return {}
    return out


def _text_segments(
    text: str,
    *,
    bold: bool = False,
    italic: bool = False,
    code: bool = False,
    unicode_fallback: bool = False,
    convert_sequences: bool = True,
) -> list[dict[str, object]]:
    if not text:
        return []
    if not unicode_fallback:
        return [
            {
                "text": text,
                "bold": bold,
                "italic": italic,
                "code": code,
                "symbol": False,
                "unicode": False,
            }
        ]
    segments: list[dict[str, object]] = []
    buf: list[str] = []
    i = 0
    while i < len(text):
        if convert_sequences:
            match = None
            for key in sorted(_SPECIAL_SYMBOL_SEQUENCES.keys(), key=len, reverse=True):
                if text.startswith(key, i):
                    match = _SPECIAL_SYMBOL_SEQUENCES[key]
                    i += len(key)
                    break
            if match:
                if buf:
                    segments.append(
                        {
                            "text": "".join(buf),
                            "bold": bold,
                            "italic": italic,
                            "code": code,
                            "symbol": False,
                            "unicode": False,
                        }
                    )
                    buf = []
                segments.append(
                    {
                        "text": match,
                        "bold": False,
                        "italic": False,
                        "code": False,
                        "symbol": True,
                        "unicode": False,
                    }
                )
                continue
        ch = text[i]
        i += 1
        if ch in _SPECIAL_SYMBOLS:
            if buf:
                segments.append(
                    {
                        "text": "".join(buf),
                        "bold": bold,
                        "italic": italic,
                        "code": code,
                        "symbol": False,
                        "unicode": False,
                    }
                )
                buf = []
            segments.append(
                {
                    "text": ch,
                    "bold": False,
                    "italic": False,
                    "code": False,
                    "symbol": True,
                    "unicode": False,
                }
            )
            continue
        if ord(ch) > 0x00FF:
            if buf:
                segments.append(
                    {
                        "text": "".join(buf),
                        "bold": bold,
                        "italic": italic,
                        "code": code,
                        "symbol": False,
                        "unicode": False,
                    }
                )
                buf = []
            segments.append(
                {
                    "text": ch,
                    "bold": bold,
                    "italic": italic,
                    "code": code,
                    "symbol": False,
                    "unicode": True,
                }
            )
            continue
        buf.append(ch)
    if buf:
        segments.append(
            {
                "text": "".join(buf),
                "bold": bold,
                "italic": italic,
                "code": code,
                "symbol": False,
                "unicode": False,
            }
        )
    return segments


def _inline_segments(token, *, unicode_fallback: bool = False) -> list[dict[str, object]]:
    if not token or token.type != "inline":
        return _text_segments(token.content if token else "", unicode_fallback=unicode_fallback)
    segments: list[dict[str, object]] = []
    bold = False
    italic = False
    link_href: str | None = None
    for child in token.children or []:
        if child.type == "text":
            segments.extend(
                _text_segments(
                    child.content,
                    bold=bold,
                    italic=italic,
                    unicode_fallback=unicode_fallback,
                )
            )
        elif child.type == "strong_open":
            bold = True
        elif child.type == "strong_close":
            bold = False
        elif child.type == "em_open":
            italic = True
        elif child.type == "em_close":
            italic = False
        elif child.type == "code_inline":
            segments.extend(
                _text_segments(
                    child.content,
                    code=True,
                    unicode_fallback=unicode_fallback,
                    convert_sequences=False,
                )
            )
        elif child.type == "softbreak":
            segments.append(
                {
                    "text": "\n",
                    "bold": bold,
                    "italic": italic,
                    "code": False,
                    "symbol": False,
                    "unicode": False,
                }
            )
        elif child.type == "hardbreak":
            segments.append(
                {
                    "text": "\n",
                    "bold": bold,
                    "italic": italic,
                    "code": False,
                    "symbol": False,
                    "unicode": False,
                }
            )
        elif child.type == "link_open":
            link_href = ""
            attrs = _attrs_to_dict(child.attrs)
            if "href" in attrs:
                link_href = attrs.get("href") or ""
        elif child.type == "link_close":
            if link_href:
                segments.extend(
                    _text_segments(f" ({link_href})", unicode_fallback=unicode_fallback)
                )
            link_href = None
    return segments


def _list_start(token) -> int:
    if not token or not token.attrs:
        return 1
    attrs = _attrs_to_dict(token.attrs)
    if "start" in attrs:
        try:
            return int(attrs["start"])
        except (TypeError, ValueError):
            return 1
    return 1


def _tokens_to_lines(tokens: list) -> list[str]:
    lines: list[str] = []
    list_stack: list[dict[str, int | str]] = []
    item_stack: list[dict[str, bool]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        t = tok.type
        if t in {"bullet_list_open", "ordered_list_open"}:
            list_stack.append(
                {
                    "type": "ol" if t.startswith("ordered") else "ul",
                    "index": _list_start(tok),
                }
            )
            i += 1
            continue
        if t in {"bullet_list_close", "ordered_list_close"}:
            if list_stack:
                list_stack.pop()
            i += 1
            continue
        if t == "list_item_open":
            item_stack.append({"first": True})
            i += 1
            continue
        if t == "list_item_close":
            if item_stack:
                item_stack.pop()
            i += 1
            continue
        if t == "paragraph_open":
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            text = _inline_text(inline, skip_images=True).strip()
            if text:
                if list_stack and item_stack:
                    top = list_stack[-1]
                    item = item_stack[-1]
                    if top["type"] == "ol":
                        prefix = f"{top['index']}."
                    else:
                        prefix = "-"
                    if item["first"]:
                        item["first"] = False
                        if top["type"] == "ol":
                            top["index"] = int(top["index"]) + 1
                        line = f"{prefix} {text}".strip()
                    else:
                        line = text
                    lines.append(line)
                else:
                    lines.append(text)
            i += 3
            continue
        if t in {"fence", "code_block"}:
            if tok.content:
                lines.extend([ln for ln in tok.content.splitlines() if ln.strip()])
            i += 1
            continue
        i += 1
    return lines


def _normalize_inline(text: str) -> str:
    text = _LINK_RE.sub(lambda m: f"{m.group(1)} ({m.group(2)})", text)
    text = _CODE_RE.sub(lambda m: m.group(1), text)
    text = _BOLD_RE.sub(lambda m: m.group(2), text)
    text = _ITALIC_RE.sub(lambda m: m.group(2), text)
    return text


def _split_long_tokens(text: str, max_len: int = 60) -> str:
    parts = _TOKEN_SPLIT_RE.split(text)
    out: list[str] = []
    for part in parts:
        if not part or part.isspace():
            out.append(part)
            continue
        if len(part) <= max_len:
            out.append(part)
            continue
        chunks = [part[i : i + max_len] for i in range(0, len(part), max_len)]
        out.append(" ".join(chunks))
    return "".join(out)


def _wrap_code_line(line: str, max_len: int = 90) -> list[str]:
    if len(line) <= max_len:
        return [line]
    return [line[i : i + max_len] for i in range(0, len(line), max_len)]


def _resolve_pdf_font(name: str | None, *, default: str = "Helvetica", custom: dict[str, str] | None = None) -> str:
    if not name:
        return default
    key = str(name).strip().lower()
    if custom and key in custom:
        return custom[key]
    if key in _PDF_FONT_MAP:
        return _PDF_FONT_MAP[key]
    if "serif" in key:
        return "Times"
    if "mono" in key or "courier" in key:
        return "Courier"
    return "Helvetica"


def _register_custom_fonts(pdf: FPDF) -> dict[str, str]:
    registered: dict[str, str] = {}
    if not _FONT_DIR.exists():
        return registered
    for key, meta in _CUSTOM_FONTS.items():
        family = meta.get("family")
        files = meta.get("files") or {}
        if not family or "" not in files:
            continue
        regular_path = _FONT_DIR / files[""]
        if not regular_path.exists():
            continue
        for style, filename in files.items():
            path = _FONT_DIR / filename
            if not path.exists():
                continue
            pdf.add_font(family, style=style, fname=str(path))
        registered[key] = family
    return registered


def _normalize_heading_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _first_heading_text(markdown: str) -> str | None:
    for raw in str(markdown or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        match = _HEADING_RE.match(line)
        if match:
            return match.group(2).strip()
        break
    return None


def _headings_match(title: str | None, heading: str | None) -> bool:
    if not title or not heading:
        return False
    return _normalize_heading_text(title) == _normalize_heading_text(heading)


def _is_table_separator(line: str) -> bool:
    raw = line.strip()
    if not raw or "|" not in raw:
        return False
    cells = [c.strip() for c in raw.strip("|").split("|")]
    if not cells:
        return False
    for cell in cells:
        if not cell:
            return False
        if not re.fullmatch(r":?-{3,}:?", cell):
            return False
    return True


def _split_table_row(line: str) -> list[str]:
    raw = line.strip().strip("|")
    if not raw:
        return []
    return [c.strip() for c in raw.split("|")]


def _looks_like_table_row(line: str) -> bool:
    raw = line.strip()
    if "|" not in raw:
        return False
    if raw.startswith("|") or raw.endswith("|"):
        return True
    return raw.count("|") >= 2


def _safe_resolve(base: Path, rel: str) -> Path | None:
    candidate = (base / rel).resolve()
    if not str(candidate).startswith(str(base.resolve())):
        return None
    return candidate


def _resolve_image_path(src: str, *, version: str) -> Path | None:
    raw = str(src or "").strip()
    if not raw:
        return None
    if "|width=" in raw or "|w=" in raw:
        raw = raw.split("|", 1)[0]
    path = raw
    if raw.startswith("http://") or raw.startswith("https://"):
        path = urlparse(raw).path
    if path.startswith("assets/"):
        path = f"/{path}"
    if not path.startswith("/assets/"):
        return None
    parts = path.split("/", 3)
    if len(parts) < 4:
        return None
    ver = parts[2] or version
    rel = parts[3]
    base = DATASTORE_DIR / ver / "assets"
    resolved = _safe_resolve(base, rel)
    if not resolved or not resolved.exists() or not resolved.is_file():
        return None
    return resolved


def _parse_image_width(src: str) -> tuple[str, float | None, str | None]:
    raw = str(src or "").strip()
    if not raw:
        return "", None, None
    width_spec = None
    if "|width=" in raw or "|w=" in raw:
        base, _, tail = raw.partition("|")
        raw = base.strip()
        match = re.match(r"^(?:width|w)=(.+)$", tail.strip(), re.IGNORECASE)
        if match:
            width_spec = match.group(1).strip()
    parsed = urlparse(raw)
    if not width_spec and parsed.query:
        params = parse_qs(parsed.query)
        width_spec = params.get("width", [None])[0] or params.get("w", [None])[0]
    if not width_spec and parsed.fragment:
        fragment = parsed.fragment
        if "=" in fragment:
            params = parse_qs(fragment)
            width_spec = params.get("width", [None])[0] or params.get("w", [None])[0]
    if not width_spec:
        return raw, None, None
    match = _IMG_WIDTH_RE.match(width_spec.strip())
    if not match:
        return raw, None, None
    value = float(match.group("value"))
    unit = (match.group("unit") or "px").lower()
    return raw, value, unit


def _strip_yaml_quotes(value: str) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]
    return text.strip()


def _split_front_matter_block(markdown: str) -> tuple[str | None, str]:
    src = str(markdown or "")
    if src.startswith("\ufeff"):
        src = src.lstrip("\ufeff")
    if not src.startswith("---"):
        return None, markdown
    lines = src.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, markdown
    end_idx = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            end_idx = idx
            break
    if end_idx is None:
        return None, markdown
    front = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1 :]).lstrip("\n")
    return front, body


def _parse_front_matter(front: str) -> dict[str, object]:
    meta: dict[str, object] = {}
    current_key: str | None = None
    for raw in str(front or "").splitlines():
        if not raw.strip():
            continue
        m = _FRONT_MATTER_KEY_RE.match(raw)
        if m:
            key = m.group(1).strip()
            value = m.group(2).strip()
            if value == "":
                meta[key] = {}
                current_key = key
            else:
                meta[key] = _strip_yaml_quotes(value)
                current_key = None
            continue
        if current_key:
            m = _FRONT_MATTER_CHILD_RE.match(raw)
            if not m:
                continue
            child_key = m.group(1).strip()
            child_val = _strip_yaml_quotes(m.group(2))
            node = meta.get(current_key)
            if not isinstance(node, dict):
                node = {}
                meta[current_key] = node
            node[child_key] = child_val
    return meta


def _extract_cover_overrides(meta: dict[str, object]) -> dict[str, str]:
    cover: dict[str, str] = {}
    cover_block = meta.get("cover")
    if isinstance(cover_block, dict):
        for key, val in cover_block.items():
            if isinstance(val, str) and val.strip():
                cover[key] = val.strip()
    mapping = {
        "guide_type": "type",
        "cover_type": "type",
        "cover_title": "title",
        "cover_image": "image",
        "cover_text": "text",
        "cover_version": "version",
        "cover_date": "date",
        "cover_copyright": "copyright",
    }
    for src_key, dst_key in mapping.items():
        val = meta.get(src_key)
        if isinstance(val, str) and val.strip():
            cover[dst_key] = val.strip()
    return cover


def _extract_header_overrides(meta: dict[str, object]) -> dict[str, str]:
    headers: dict[str, str] = {}
    header_block = meta.get("headers")
    if isinstance(header_block, dict):
        for key, val in header_block.items():
            if isinstance(val, str) and val.strip():
                headers[key] = val.strip()
    mapping = {
        "header_left": "left",
        "header_right": "right",
    }
    for src_key, dst_key in mapping.items():
        val = meta.get(src_key)
        if isinstance(val, str) and val.strip():
            headers[dst_key] = val.strip()
    return headers


def _extract_section_marker(raw: str | None) -> str | None:
    if not raw:
        return None
    match = _SECTION_MARKER_RE.search(str(raw))
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def render_markdown_to_pdf(
    markdown: str,
    *,
    title: str,
    version: str,
    heading_font: str | None = None,
    body_font: str | None = None,
    cover: dict[str, object] | None = None,
) -> bytes:
    pdf = FPDF(format="A4", unit="mm")
    pdf.alias_nb_pages()
    pdf.set_margins(15, 24, 15)
    markdown = str(markdown or "")
    markdown = markdown.replace("\\r\\n", "\n").replace("\\n", "\n")
    front_block, markdown_body = _split_front_matter_block(markdown)
    has_front = front_block is not None
    front_meta = _parse_front_matter(front_block) if has_front else {}
    cover_overrides = _extract_cover_overrides(front_meta)
    header_overrides = _extract_header_overrides(front_meta)
    if cover_overrides:
        cover_data = dict(cover or {})
        if not cover_data:
            cover_data = {
                "guide_type": "Guide",
                "title": title,
                "version": version,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
        if "type" in cover_overrides:
            cover_data["guide_type"] = cover_overrides["type"]
        if "title" in cover_overrides:
            cover_data["title"] = cover_overrides["title"]
        if "image" in cover_overrides:
            cover_data["image"] = cover_overrides["image"]
        if "text" in cover_overrides:
            cover_data["text"] = cover_overrides["text"]
        if "version" in cover_overrides:
            cover_data["version"] = cover_overrides["version"]
        if "date" in cover_overrides:
            cover_data["generated_at"] = cover_overrides["date"]
        if "copyright" in cover_overrides:
            cover_data["copyright"] = cover_overrides["copyright"]
        cover = cover_data
    if cover and "guide_type" not in cover and cover_overrides.get("type"):
        cover["guide_type"] = cover_overrides["type"]
    markdown = markdown_body if has_front else markdown

    custom_fonts = _register_custom_fonts(pdf)
    content_width = pdf.w - pdf.l_margin - pdf.r_margin
    half_width = content_width / 2.0
    heading_face = _resolve_pdf_font(heading_font, default="Helvetica", custom=custom_fonts)
    body_face = _resolve_pdf_font(body_font, default="Helvetica", custom=custom_fonts)
    custom_values = set(custom_fonts.values())
    unicode_face = custom_fonts.get("unicode")
    allow_unicode_heading = heading_face in custom_values
    allow_unicode_body = body_face in custom_values
    allow_unicode_unicode = unicode_face in custom_values if unicode_face else False
    use_unicode_fallback = bool(unicode_face)
    pdf.set_title(_sanitize_pdf_text(title, allow_unicode=allow_unicode_heading))

    current_section = ""
    section_marker_seen = False
    first_marker = _extract_section_marker(markdown)
    if first_marker:
        current_section = _replace_symbol_sequences(_normalize_inline(first_marker))
        section_marker_seen = True

    def _fit_header_text(text: str, max_width: float, face: str, style: str, size: int, allow_unicode: bool) -> str:
        if not text:
            return ""
        pdf.set_font(face, style, size)
        safe = _sanitize_pdf_text(text, allow_unicode=allow_unicode)
        if pdf.get_string_width(safe) <= max_width:
            return safe
        ellipsis = "…" if allow_unicode else "..."
        for i in range(len(text), 0, -1):
            candidate = f"{text[:i].rstrip()}{ellipsis}"
            safe_candidate = _sanitize_pdf_text(candidate, allow_unicode=allow_unicode)
            if pdf.get_string_width(safe_candidate) <= max_width:
                return safe_candidate
        return _sanitize_pdf_text(text[:1], allow_unicode=allow_unicode)

    def header(self: FPDF) -> None:
        if self.page_no() <= 1 and cover:
            return
        left_label = ""
        right_label = ""
        if header_overrides.get("left"):
            left_label = header_overrides["left"]
        elif cover and cover.get("guide_type"):
            left_label = str(cover.get("guide_type") or "")
        if header_overrides.get("right"):
            right_label = header_overrides["right"]
        else:
            right_label = current_section
        if not left_label and not right_label:
            return
        self.set_y(8)
        size = 8
        use_unicode_left = use_unicode_fallback and _needs_unicode(left_label)
        use_unicode_right = use_unicode_fallback and _needs_unicode(right_label)
        left_face = unicode_face if use_unicode_left and unicode_face else body_face
        right_face = unicode_face if use_unicode_right and unicode_face else body_face
        left_allow_unicode = allow_unicode_unicode if use_unicode_left and unicode_face else allow_unicode_body
        right_allow_unicode = allow_unicode_unicode if use_unicode_right and unicode_face else allow_unicode_body
        left_text = _fit_header_text(left_label, half_width - 2, left_face, "", size, left_allow_unicode)
        right_text = _fit_header_text(right_label, half_width - 2, right_face, "", size, right_allow_unicode)
        self.set_text_color(110, 110, 110)
        self.set_font(left_face, "", size)
        self.set_x(self.l_margin)
        self.cell(half_width, 5, left_text, align="L")
        self.set_font(right_face, "", size)
        self.cell(half_width, 5, right_text, align="R")
        self.set_text_color(0, 0, 0)
        self.set_y(self.t_margin)

    def footer(self: FPDF) -> None:
        if self.page_no() <= 1 and cover:
            return
        self.set_y(-12)
        size = 8
        if use_unicode_fallback and unicode_face:
            self.set_font(unicode_face, "", size)
            allow_unicode = allow_unicode_unicode
        else:
            self.set_font(body_face, "", size)
            allow_unicode = allow_unicode_body
        page_num = self.page_no()
        if cover:
            page_num -= 1
        label = _sanitize_pdf_text(f"{page_num}", allow_unicode=allow_unicode)
        self.cell(0, 8, label, align="C")

    pdf.header = header.__get__(pdf, FPDF)
    pdf.footer = footer.__get__(pdf, FPDF)
    pdf.set_auto_page_break(auto=True, margin=15)

    def resolve_cover_image(src: str) -> tuple[Path | None, float | None, str | None]:
        if not src:
            return None, None, None
        clean_src, width_val, width_unit = _parse_image_width(src)
        if clean_src.startswith("http://") or clean_src.startswith("https://"):
            return None, width_val, width_unit
        if clean_src.startswith("/assets/") or clean_src.startswith("assets/"):
            return _resolve_image_path(clean_src, version=version), width_val, width_unit
        rel = clean_src.lstrip("/")
        candidate = _safe_resolve(REPO_ROOT, rel)
        if candidate and candidate.exists() and candidate.is_file():
            return candidate, width_val, width_unit
        return None, width_val, width_unit

    def write_cover_page() -> None:
        guide_type = str((cover or {}).get("guide_type") or "Guide").strip()
        cover_title = str((cover or {}).get("title") or title).strip() or title
        cover_text = str((cover or {}).get("text") or "").strip()
        cover_image = str((cover or {}).get("image") or "").strip()
        cover_version = str((cover or {}).get("version") or version).strip()
        if cover_version:
            cover_version = cover_version.replace("_", " ").strip()
        cover_date = str((cover or {}).get("generated_at") or "").strip()
        cover_copyright = str((cover or {}).get("copyright") or "").strip()

        pdf.add_page()
        pdf.set_text_color(90, 90, 90)
        pdf.set_y(26)
        type_text = guide_type.upper() if guide_type else ""
        if type_text:
            use_unicode = use_unicode_fallback and _needs_unicode(type_text)
            face = unicode_face if use_unicode and unicode_face else heading_face
            allow_unicode = allow_unicode_unicode if use_unicode and unicode_face else allow_unicode_heading
            pdf.set_font(face, "B", 12)
            safe_type = _sanitize_pdf_text(_split_long_tokens(type_text), allow_unicode=allow_unicode)
            pdf.multi_cell(0, 6, safe_type, align="C")
            pdf.ln(2)

        pdf.set_text_color(20, 20, 20)
        use_unicode = use_unicode_fallback and _needs_unicode(cover_title)
        face = unicode_face if use_unicode and unicode_face else heading_face
        allow_unicode = allow_unicode_unicode if use_unicode and unicode_face else allow_unicode_heading
        pdf.set_font(face, "B", 28)
        safe_title = _sanitize_pdf_text(_split_long_tokens(cover_title), allow_unicode=allow_unicode)
        pdf.multi_cell(0, 12, safe_title, align="C")
        pdf.ln(4)

        if cover_image:
            path, width_val, width_unit = resolve_cover_image(cover_image)
            if path:
                write_image(
                    path,
                    width_override=width_val,
                    width_unit=width_unit,
                    allow_page_break=False,
                )

        if cover_text:
            use_unicode = use_unicode_fallback and _needs_unicode(cover_text)
            face = unicode_face if use_unicode and unicode_face else body_face
            allow_unicode = allow_unicode_unicode if use_unicode and unicode_face else allow_unicode_body
            pdf.set_font(face, "", 12)
            safe_text = _sanitize_pdf_text(_split_long_tokens(cover_text, max_len=80), allow_unicode=allow_unicode)
            pdf.multi_cell(0, 6, safe_text, align="C")

        footer_lines: list[str] = []
        if cover_version and cover_date:
            footer_lines.append(f"{cover_version} • {cover_date}")
        elif cover_version:
            footer_lines.append(cover_version)
        elif cover_date:
            footer_lines.append(cover_date)
        if cover_copyright:
            footer_lines.append(cover_copyright)
        if footer_lines:
            pdf.set_y(pdf.h - pdf.b_margin - 22)
            use_unicode = use_unicode_fallback and _needs_unicode(" ".join(footer_lines))
            face = unicode_face if use_unicode and unicode_face else body_face
            allow_unicode = allow_unicode_unicode if use_unicode and unicode_face else allow_unicode_body
            pdf.set_font(face, "", 9)
            pdf.set_text_color(110, 110, 110)
            pdf.multi_cell(
                0,
                5,
                _sanitize_pdf_text("\n".join(footer_lines), allow_unicode=allow_unicode),
                align="C",
            )
            pdf.set_text_color(0, 0, 0)

    def write_heading(text: str, level: int) -> None:
        size = {1: 18, 2: 16, 3: 14, 4: 12, 5: 11, 6: 11}.get(level, 12)
        normalized = _replace_symbol_sequences(_normalize_inline(text))
        nonlocal current_section, section_marker_seen
        if not section_marker_seen:
            if level == 2:
                current_section = normalized
            elif level == 1 and not current_section:
                current_section = normalized
        use_unicode = use_unicode_fallback and _needs_unicode(normalized)
        head_face = unicode_face if use_unicode and unicode_face else heading_face
        allow_unicode = allow_unicode_unicode if use_unicode and unicode_face else allow_unicode_heading
        pdf.set_font(head_face, "B", size)
        pdf.set_x(pdf.l_margin)
        safe = _sanitize_pdf_text(_split_long_tokens(normalized), allow_unicode=allow_unicode)
        pdf.multi_cell(0, 8, safe)
        pdf.ln(1)
        pdf.set_font(body_face, "", 11)

    def write_rich_text(
        segments: list[dict[str, object]],
        *,
        prefix: str = "",
        indent: float = 0.0,
        line_height: float = 6.0,
        gap: float = 0.0,
    ) -> None:
        if not segments and not prefix:
            return
        if prefix:
            prefix_segments = _text_segments(prefix, unicode_fallback=use_unicode_fallback)
            segments = prefix_segments + _text_segments(" ", unicode_fallback=use_unicode_fallback) + segments
        prev_left = pdf.l_margin
        pdf.set_left_margin(prev_left + indent)
        pdf.set_x(pdf.l_margin)
        for seg in segments:
            text = str(seg.get("text") or "")
            if not text:
                continue
            if text == "\n":
                pdf.ln(line_height)
                pdf.set_x(pdf.l_margin)
                continue
            if len(text) > 60 and not seg.get("code"):
                text = _split_long_tokens(text)
            if (seg.get("symbol") or seg.get("unicode")) and unicode_face:
                style = ""
                if seg.get("bold"):
                    style += "B"
                if seg.get("italic"):
                    style += "I"
                pdf.set_font(unicode_face, style, 11)
                safe = _sanitize_pdf_text(text, allow_unicode=allow_unicode_unicode)
                pdf.write(line_height, safe)
                continue
            if seg.get("code"):
                pdf.set_font("Courier", "", 10)
                safe = _sanitize_pdf_text(text, allow_unicode=False)
                pdf.write(line_height, safe)
                continue
            style = ""
            if seg.get("bold"):
                style += "B"
            if seg.get("italic"):
                style += "I"
            pdf.set_font(body_face, style, 11)
            safe = _sanitize_pdf_text(text, allow_unicode=allow_unicode_body)
            pdf.write(line_height, safe)
        pdf.ln(line_height)
        if gap:
            pdf.ln(gap)
        pdf.set_left_margin(prev_left)
        pdf.set_x(prev_left)

    def write_paragraph(text: str) -> None:
        if not text:
            return
        segments = _text_segments(_normalize_inline(text), unicode_fallback=use_unicode_fallback)
        write_rich_text(segments, line_height=6, gap=1)

    def write_list_item(text: str, prefix: str, *, indent: float = 0.0, continuation: bool = False) -> None:
        bullet = f"{prefix}" if prefix and not continuation else ""
        segments = _text_segments(_normalize_inline(text), unicode_fallback=use_unicode_fallback)
        write_rich_text(segments, prefix=bullet, indent=indent, line_height=6, gap=0)

    def write_code_block(lines: list[str]) -> None:
        pdf.set_font("Courier", "", 9)
        for ln in lines:
            for chunk in _wrap_code_line(ln):
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 4, _sanitize_pdf_text(chunk, allow_unicode=False))
        pdf.ln(1)
        pdf.set_font(body_face, "", 11)

    def write_blockquote(lines: list[str]) -> None:
        if not lines:
            return
        text = "\n".join(s.strip() for s in lines if s.strip())
        if not text:
            return
        text = _replace_symbol_sequences(_normalize_inline(text))
        segments = _text_segments(text, italic=True, unicode_fallback=use_unicode_fallback)
        pdf.set_text_color(120, 120, 120)
        write_rich_text(segments, line_height=5, gap=1)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font(body_face, "", 11)

    def write_admonition(kind: str, title_text: str, lines: list[str]) -> None:
        body = "\n".join(s.strip() for s in lines if s.strip())
        if kind == "warning":
            fill = (240, 220, 220)
            stripe = (216, 92, 92)
        elif kind == "info":
            fill = (220, 230, 245)
            stripe = (88, 155, 219)
        else:
            fill = (220, 240, 230)
            stripe = (63, 163, 124)
        padding_x = 6
        padding_y = 3
        start_x = pdf.l_margin
        start_y = pdf.get_y()
        text_width = content_width - padding_x * 2
        line_height = 6
        title_text = _replace_symbol_sequences(_normalize_inline(title_text))
        body_text = _replace_symbol_sequences(_normalize_inline(body))
        use_unicode = use_unicode_fallback and _needs_unicode(f"{title_text} {body_text}")
        admon_face = unicode_face if use_unicode and unicode_face else body_face
        allow_unicode_admon = allow_unicode_unicode if use_unicode and unicode_face else allow_unicode_body
        pdf.set_font(admon_face, "B", 11)
        safe_title = _sanitize_pdf_text(_split_long_tokens(title_text), allow_unicode=allow_unicode_admon)
        title_lines = pdf.multi_cell(text_width, line_height, safe_title, split_only=True)
        body_lines = []
        if body_text:
            pdf.set_font(admon_face, "", 11)
            safe_body = _sanitize_pdf_text(_split_long_tokens(body_text), allow_unicode=allow_unicode_admon)
            body_lines = pdf.multi_cell(text_width, line_height, safe_body, split_only=True)
        total_lines = max(1, len(title_lines)) + (len(body_lines) if body_lines else 0)
        total_height = total_lines * line_height + padding_y * 2
        if start_y + total_height > pdf.h - pdf.b_margin:
            pdf.add_page()
            start_y = pdf.get_y()
        pdf.set_fill_color(*fill)
        pdf.rect(start_x, start_y, content_width, total_height, "F")
        pdf.set_fill_color(*stripe)
        pdf.rect(start_x, start_y, 3, total_height, "F")
        prev_left = pdf.l_margin
        pdf.set_left_margin(start_x + padding_x)
        pdf.set_xy(start_x + padding_x, start_y + padding_y)
        pdf.set_font(admon_face, "B", 11)
        pdf.multi_cell(text_width, line_height, safe_title)
        pdf.set_font(admon_face, "", 11)
        if body_lines:
            pdf.set_x(start_x + padding_x)
            pdf.multi_cell(text_width, line_height, "\n".join(body_lines))
        pdf.set_left_margin(prev_left)
        pdf.set_xy(start_x, start_y + total_height + 2)

    def write_table(rows: list[list[str]]) -> None:
        if not rows:
            return
        cols = max(len(r) for r in rows)
        if cols == 0:
            return
        table_font_size = 10
        min_font_size = 7
        padding = 2.0

        def cell_text(cell: str) -> str:
            normalized = _replace_symbol_sequences(_normalize_inline(cell))
            return _split_long_tokens(normalized, max_len=32)

        def measure_width(text: str, face: str, style: str, size: int) -> float:
            pdf.set_font(face, style, size)
            return pdf.get_string_width(text)

        def compute_widths(size: int) -> list[float] | None:
            widths = [0.0] * cols
            for r_index, row in enumerate(rows):
                cells = list(row) + [""] * (cols - len(row))
                for c_index, cell in enumerate(cells):
                    text = cell_text(cell)
                    use_unicode = use_unicode_fallback and _needs_unicode(text)
                    face = unicode_face if use_unicode and unicode_face else body_face
                    style = "B" if r_index == 0 else ""
                    width = measure_width(text, face, style, size) + padding * 2
                    widths[c_index] = max(widths[c_index], width)
            total = sum(widths)
            if total <= 0:
                return [content_width / cols] * cols
            if total > content_width:
                scale = content_width / total
                widths = [w * scale for w in widths]
            pdf.set_font(body_face, "", size)
            min_char = max(4.0, pdf.get_string_width("W") + padding)
            if min(widths) < min_char:
                return None
            return widths

        col_widths: list[float] | None = None
        size = table_font_size
        while size >= min_font_size:
            col_widths = compute_widths(size)
            if col_widths:
                table_font_size = size
                break
            size -= 1
        if not col_widths:
            table_font_size = min_font_size
            col_widths = [content_width / cols] * cols

        line_height = max(4.0, table_font_size * 0.6)
        pdf.set_draw_color(200, 200, 200)
        for r_index, row in enumerate(rows):
            cells = list(row) + [""] * (cols - len(row))
            split_cells = []
            max_lines = 1
            for c_index, cell in enumerate(cells):
                normalized = cell_text(cell)
                use_unicode = use_unicode_fallback and _needs_unicode(normalized)
                face = unicode_face if use_unicode and unicode_face else body_face
                style = "B" if r_index == 0 else ""
                allow_unicode_cell = allow_unicode_unicode if use_unicode and unicode_face else allow_unicode_body
                text = _sanitize_pdf_text(normalized, allow_unicode=allow_unicode_cell)
                pdf.set_font(face, style, table_font_size)
                lines = pdf.multi_cell(col_widths[c_index], line_height, text, split_only=True)
                if not lines:
                    lines = [""]
                split_cells.append((lines, face, style))
                max_lines = max(max_lines, len(lines))
            row_height = line_height * max_lines
            if pdf.get_y() + row_height > pdf.h - pdf.b_margin:
                pdf.add_page()
            x = pdf.l_margin
            y = pdf.get_y()
            for c_index, cell_info in enumerate(split_cells):
                lines, face, style = cell_info
                pdf.set_xy(x, y)
                text = "\n".join(lines)
                pdf.set_font(face, style, table_font_size)
                if r_index == 0:
                    pdf.set_fill_color(235, 235, 235)
                    pdf.multi_cell(col_widths[c_index], line_height, text, border=1, fill=True)
                else:
                    pdf.multi_cell(col_widths[c_index], line_height, text, border=1)
                x += col_widths[c_index]
            pdf.set_xy(pdf.l_margin, y + row_height)
        pdf.ln(1)

    def write_image(
        path: Path,
        *,
        width_override: float | None = None,
        width_unit: str | None = None,
        allow_page_break: bool = True,
    ) -> None:
        try:
            with Image.open(path) as img:
                width_px, height_px = img.size
                dpi = img.info.get("dpi")
        except Exception:
            width_px = 0
            height_px = 0
            dpi = None
        if width_px <= 0 or height_px <= 0:
            pdf.set_font(body_face, "", 10)
            pdf.multi_cell(0, 5, _sanitize_pdf_text(f"[image omitted: {path.name}]", allow_unicode=allow_unicode_body))
            pdf.ln(1)
            return
        dpi_value = 96.0
        if isinstance(dpi, (tuple, list)) and dpi and dpi[0]:
            dpi_value = float(dpi[0])
        elif isinstance(dpi, (int, float)) and dpi:
            dpi_value = float(dpi)
        natural_width = (width_px / dpi_value) * 25.4
        width_mm = 0.0
        if width_override and width_override > 0:
            unit = (width_unit or "px").lower()
            if unit == "%":
                width_mm = content_width * (width_override / 100.0)
            elif unit == "mm":
                width_mm = width_override
            elif unit == "cm":
                width_mm = width_override * 10.0
            else:
                width_mm = (width_override / dpi_value) * 25.4
            width_mm = max(5.0, min(width_mm, content_width))
        else:
            is_large = width_px >= 900 or natural_width >= content_width * 0.85
            if is_large:
                width_mm = content_width
            else:
                max_inline = content_width * 0.6
                width_mm = min(natural_width, max_inline)
                if width_mm <= 0:
                    width_mm = min(natural_width, content_width)
        height_mm = width_mm * (height_px / width_px)
        remaining = pdf.h - pdf.b_margin - pdf.get_y()
        max_height = pdf.h - pdf.t_margin - pdf.b_margin
        if height_mm > remaining:
            if allow_page_break:
                pdf.add_page()
                remaining = pdf.h - pdf.b_margin - pdf.get_y()
            else:
                scale = remaining / height_mm if remaining > 0 else 1.0
                width_mm = width_mm * scale
                height_mm = height_mm * scale
        if height_mm > max_height:
            scale = max_height / height_mm
            width_mm = width_mm * scale
            height_mm = max_height
        try:
            x = pdf.l_margin
            if width_mm < content_width * 0.85:
                x = pdf.l_margin + (content_width - width_mm) / 2
            pdf.image(str(path), x=x, w=width_mm, h=height_mm)
            pdf.ln(2)
        except Exception:
            pdf.set_font(body_face, "", 10)
            pdf.multi_cell(0, 5, _sanitize_pdf_text(f"[image omitted: {path.name}]", allow_unicode=allow_unicode_body))
            pdf.ln(1)

    first_heading = _first_heading_text(markdown)
    if not current_section and first_heading and not section_marker_seen:
        current_section = _replace_symbol_sequences(_normalize_inline(first_heading))

    if cover:
        write_cover_page()
    pdf.add_page()

    parser = _get_markdown_parser()
    normalized = _normalize_admonitions(markdown)
    tokens = parser.parse(normalized)
    list_stack: list[dict[str, int | str]] = []
    item_stack: list[dict[str, bool]] = []
    indent_step = 6.0

    def list_prefix() -> str:
        if not list_stack:
            return "-"
        top = list_stack[-1]
        if top["type"] == "ol":
            return f"{top['index']}."
        return "•"

    def list_indent() -> float:
        return indent_step * max(0, len(list_stack) - 1)

    def inline_images(token) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        if not token or not token.children:
            return out
        for child in token.children:
            if child.type != "image":
                continue
            attrs = _attrs_to_dict(child.attrs)
            src = attrs.get("src") if attrs else None
            if src:
                clean_src, width_val, width_unit = _parse_image_width(src)
                out.append(
                    {
                        "src": clean_src,
                        "width": width_val,
                        "unit": width_unit,
                    }
                )
        return out

    def parse_block(start_idx: int, open_type: str, close_type: str) -> tuple[list, int]:
        depth = 0
        i = start_idx
        if tokens[i].type == open_type:
            depth = 1
            i += 1
        inner_start = i
        while i < len(tokens):
            t = tokens[i].type
            if t == open_type:
                depth += 1
            elif t == close_type:
                depth -= 1
                if depth == 0:
                    break
            i += 1
        inner = tokens[inner_start:i]
        return inner, i + 1

    def parse_table(start_idx: int) -> tuple[list[list[str]], int]:
        rows: list[list[str]] = []
        i = start_idx + 1
        current_row: list[str] | None = None
        cell_text = ""
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == "table_close":
                break
            if tok.type == "tr_open":
                current_row = []
            elif tok.type in {"th_open", "td_open"}:
                cell_text = ""
            elif tok.type == "inline" and current_row is not None:
                cell_text = _inline_text(tok, skip_images=True).strip()
            elif tok.type in {"th_close", "td_close"} and current_row is not None:
                current_row.append(cell_text)
            elif tok.type == "tr_close" and current_row is not None:
                rows.append(current_row)
                current_row = None
            i += 1
        return rows, i + 1

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        t = tok.type

        if t in {"html_block", "html_inline"}:
            marker = _extract_section_marker(tok.content)
            if marker:
                current_section = _replace_symbol_sequences(_normalize_inline(marker))
                section_marker_seen = True
            i += 1
            continue

        if t == "heading_open":
            level = int(tok.tag[1]) if tok.tag and tok.tag.startswith("h") else 2
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            text = _inline_text(inline, skip_images=True).strip()
            if text:
                write_heading(text, level)
            i += 3
            continue

        if t in {"bullet_list_open", "ordered_list_open"}:
            list_stack.append(
                {
                    "type": "ol" if t.startswith("ordered") else "ul",
                    "index": _list_start(tok),
                }
            )
            i += 1
            continue
        if t in {"bullet_list_close", "ordered_list_close"}:
            if list_stack:
                list_stack.pop()
            pdf.ln(1)
            i += 1
            continue
        if t == "list_item_open":
            item_stack.append({"first": True})
            i += 1
            continue
        if t == "list_item_close":
            if item_stack:
                item_stack.pop()
            i += 1
            continue

        if t == "paragraph_open":
            inline = tokens[i + 1] if i + 1 < len(tokens) else None
            inline_text = _inline_text(inline, skip_images=True).strip()
            if inline_text:
                marker = _extract_section_marker(inline_text)
                if marker and _SECTION_MARKER_RE.fullmatch(inline_text):
                    current_section = _replace_symbol_sequences(_normalize_inline(marker))
                    section_marker_seen = True
                    i += 3
                    continue
            for img in inline_images(inline):
                src = str(img.get("src") or "")
                path = _resolve_image_path(src, version=version)
                if path:
                    width_val = img.get("width")
                    width_unit = img.get("unit")
                    width_override = float(width_val) if isinstance(width_val, (int, float)) else None
                    write_image(path, width_override=width_override, width_unit=width_unit if isinstance(width_unit, str) else None)
            segments = _inline_segments(inline, unicode_fallback=use_unicode_fallback)
            if segments:
                if list_stack and item_stack:
                    item = item_stack[-1]
                    prefix = ""
                    if item["first"]:
                        prefix = list_prefix()
                        item["first"] = False
                        if list_stack[-1]["type"] == "ol":
                            list_stack[-1]["index"] = int(list_stack[-1]["index"]) + 1
                    write_rich_text(
                        segments,
                        prefix=prefix,
                        indent=list_indent(),
                        line_height=6,
                        gap=0,
                    )
                else:
                    write_rich_text(segments, line_height=6, gap=1)
            i += 3
            continue

        if t in {"fence", "code_block"}:
            lines = tok.content.splitlines() if tok.content else []
            write_code_block(lines)
            i += 1
            continue

        if t == "table_open":
            rows, next_i = parse_table(i)
            if rows:
                write_table(rows)
            i = next_i
            continue

        if t == "blockquote_open":
            inner, next_i = parse_block(i, "blockquote_open", "blockquote_close")
            lines = _tokens_to_lines(inner)
            write_blockquote(lines)
            i = next_i
            continue

        if t.startswith("container_") and t.endswith("_open"):
            kind = t[len("container_") : -len("_open")]
            info = str(tok.info or "").strip()
            parts = info.split(None, 1)
            title_text = parts[1].strip() if len(parts) > 1 else kind.title()
            inner, next_i = parse_block(i, t, f"container_{kind}_close")
            lines = _tokens_to_lines(inner)
            write_admonition(kind, title_text, lines)
            i = next_i
            continue

        i += 1

    output = pdf.output(dest="S")
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    return str(output).encode("latin-1", "replace")
