from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from fpdf import FPDF
from markdown_it import MarkdownIt
from mdit_py_plugins.container import container_plugin

from .config import DATASTORE_DIR, REPO_ROOT

_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+(.*)$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1")
_ITALIC_RE = re.compile(r"(\*|_)([^*_]+?)\1")
_CODE_RE = re.compile(r"`([^`]+)`")
_ADMONITION_RE = re.compile(r"^\[!(TIP|INFO|WARNING)\]\s*(.*)$", re.IGNORECASE)
_TOKEN_SPLIT_RE = re.compile(r"(\s+)")
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
    }
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


def _list_start(token) -> int:
    if not token or not token.attrs:
        return 1
    for key, val in token.attrs:
        if key == "start":
            try:
                return int(val)
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


def render_markdown_to_pdf(
    markdown: str,
    *,
    title: str,
    version: str,
    heading_font: str | None = None,
    body_font: str | None = None,
) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    markdown = str(markdown or "")
    markdown = markdown.replace("\\r\\n", "\n").replace("\\n", "\n")

    custom_fonts = _register_custom_fonts(pdf)
    content_width = pdf.w - pdf.l_margin - pdf.r_margin
    heading_face = _resolve_pdf_font(heading_font, default="Helvetica", custom=custom_fonts)
    body_face = _resolve_pdf_font(body_font, default="Helvetica", custom=custom_fonts)
    custom_values = set(custom_fonts.values())
    allow_unicode_heading = heading_face in custom_values
    allow_unicode_body = body_face in custom_values
    pdf.set_title(_sanitize_pdf_text(title, allow_unicode=allow_unicode_heading))

    def write_heading(text: str, level: int) -> None:
        size = {1: 18, 2: 16, 3: 14, 4: 12, 5: 11, 6: 11}.get(level, 12)
        pdf.set_font(heading_face, "B", size)
        pdf.set_x(pdf.l_margin)
        safe = _sanitize_pdf_text(_split_long_tokens(_normalize_inline(text)), allow_unicode=allow_unicode_heading)
        pdf.multi_cell(0, 8, safe)
        pdf.ln(1)
        pdf.set_font(body_face, "", 11)

    def write_paragraph(text: str) -> None:
        if not text:
            return
        pdf.set_font(body_face, "", 11)
        pdf.set_x(pdf.l_margin)
        safe = _sanitize_pdf_text(_split_long_tokens(_normalize_inline(text)), allow_unicode=allow_unicode_body)
        pdf.multi_cell(0, 6, safe)
        pdf.ln(1)

    def write_list_item(text: str, prefix: str, *, indent: float = 0.0, continuation: bool = False) -> None:
        pdf.set_font(body_face, "", 11)
        pdf.set_x(pdf.l_margin + indent)
        bullet = f"{prefix} " if prefix and not continuation else ""
        safe = _sanitize_pdf_text(
            _split_long_tokens(_normalize_inline(f"{bullet}{text}".strip())),
            allow_unicode=allow_unicode_body,
        )
        pdf.multi_cell(0, 6, safe)

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
        pdf.set_font(body_face, "I", 10)
        pdf.set_text_color(120, 120, 120)
        pdf.set_x(pdf.l_margin)
        safe = _sanitize_pdf_text(_split_long_tokens(_normalize_inline(text)), allow_unicode=allow_unicode_body)
        pdf.multi_cell(0, 5, safe)
        pdf.ln(1)
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
        safe_title = _sanitize_pdf_text(_split_long_tokens(_normalize_inline(title_text)), allow_unicode=allow_unicode_body)
        title_lines = pdf.multi_cell(text_width, line_height, safe_title, split_only=True)
        body_lines = []
        if body:
            safe_body = _sanitize_pdf_text(_split_long_tokens(_normalize_inline(body)), allow_unicode=allow_unicode_body)
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
        pdf.set_font(body_face, "B", 11)
        pdf.multi_cell(text_width, line_height, safe_title)
        pdf.set_font(body_face, "", 11)
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
        col_width = content_width / cols
        line_height = 6
        pdf.set_font(body_face, "", 10)
        pdf.set_draw_color(200, 200, 200)
        for r_index, row in enumerate(rows):
            cells = list(row) + [""] * (cols - len(row))
            split_cells = []
            max_lines = 1
            for cell in cells:
                text = _sanitize_pdf_text(_split_long_tokens(_normalize_inline(cell)), allow_unicode=allow_unicode_body)
                lines = pdf.multi_cell(col_width, line_height, text, split_only=True)
                if not lines:
                    lines = [""]
                split_cells.append(lines)
                max_lines = max(max_lines, len(lines))
            row_height = line_height * max_lines
            if pdf.get_y() + row_height > pdf.h - pdf.b_margin:
                pdf.add_page()
            x = pdf.l_margin
            y = pdf.get_y()
            for c_index, lines in enumerate(split_cells):
                pdf.set_xy(x, y)
                text = "\n".join(lines)
                if r_index == 0:
                    pdf.set_font(body_face, "B", 10)
                    pdf.set_fill_color(235, 235, 235)
                    pdf.multi_cell(col_width, line_height, text, border=1, fill=True)
                    pdf.set_font(body_face, "", 10)
                else:
                    pdf.multi_cell(col_width, line_height, text, border=1)
                x += col_width
            pdf.set_xy(pdf.l_margin, y + row_height)
        pdf.ln(1)

    def write_image(path: Path) -> None:
        try:
            pdf.image(str(path), w=content_width)
            pdf.ln(1)
        except Exception:
            pdf.set_font(body_face, "", 10)
            pdf.multi_cell(0, 5, _sanitize_pdf_text(f"[image omitted: {path.name}]", allow_unicode=allow_unicode_body))
            pdf.ln(1)

    first_heading = _first_heading_text(markdown)
    if title and not _headings_match(title, first_heading):
        write_heading(title, 1)

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

    def inline_images(token) -> list[str]:
        out: list[str] = []
        if not token or not token.children:
            return out
        for child in token.children:
            if child.type != "image":
                continue
            attrs = dict(child.attrs or [])
            src = attrs.get("src")
            if src:
                out.append(src)
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
            for src in inline_images(inline):
                path = _resolve_image_path(src, version=version)
                if path:
                    write_image(path)
            text = _inline_text(inline, skip_images=True).strip()
            if text:
                if list_stack and item_stack:
                    item = item_stack[-1]
                    prefix = ""
                    if item["first"]:
                        prefix = list_prefix()
                        item["first"] = False
                        if list_stack[-1]["type"] == "ol":
                            list_stack[-1]["index"] = int(list_stack[-1]["index"]) + 1
                    write_list_item(
                        text,
                        prefix,
                        indent=list_indent(),
                        continuation=not prefix,
                    )
                else:
                    write_paragraph(text)
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
