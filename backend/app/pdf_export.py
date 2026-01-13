from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from fpdf import FPDF

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


def _sanitize_pdf_text(text: str) -> str:
    return text.encode("latin-1", "replace").decode("latin-1")


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
    pdf.set_title(_sanitize_pdf_text(title))

    markdown = str(markdown or "")
    markdown = markdown.replace("\\r\\n", "\n").replace("\\n", "\n")

    custom_fonts = _register_custom_fonts(pdf)
    content_width = pdf.w - pdf.l_margin - pdf.r_margin
    heading_face = _resolve_pdf_font(heading_font, default="Helvetica", custom=custom_fonts)
    body_face = _resolve_pdf_font(body_font, default="Helvetica", custom=custom_fonts)

    def write_heading(text: str, level: int) -> None:
        size = {1: 18, 2: 16, 3: 14, 4: 12, 5: 11, 6: 11}.get(level, 12)
        pdf.set_font(heading_face, "B", size)
        pdf.set_x(pdf.l_margin)
        safe = _sanitize_pdf_text(_split_long_tokens(_normalize_inline(text)))
        pdf.multi_cell(0, 8, safe)
        pdf.ln(1)
        pdf.set_font(body_face, "", 11)

    def write_paragraph(text: str) -> None:
        if not text:
            return
        pdf.set_font(body_face, "", 11)
        pdf.set_x(pdf.l_margin)
        safe = _sanitize_pdf_text(_split_long_tokens(_normalize_inline(text)))
        pdf.multi_cell(0, 6, safe)
        pdf.ln(1)

    def write_list_item(text: str, prefix: str) -> None:
        pdf.set_font(body_face, "", 11)
        pdf.set_x(pdf.l_margin)
        safe = _sanitize_pdf_text(_split_long_tokens(_normalize_inline(f"{prefix} {text}".strip())))
        pdf.multi_cell(0, 6, safe)

    def write_code_block(lines: list[str]) -> None:
        pdf.set_font("Courier", "", 9)
        for ln in lines:
            for chunk in _wrap_code_line(ln):
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 4, _sanitize_pdf_text(chunk))
        pdf.ln(1)
        pdf.set_font(body_face, "", 11)

    def write_blockquote(lines: list[str]) -> None:
        if not lines:
            return
        text = " ".join(s.strip() for s in lines if s.strip())
        if not text:
            return
        pdf.set_font(body_face, "I", 10)
        pdf.set_text_color(120, 120, 120)
        pdf.set_x(pdf.l_margin)
        safe = _sanitize_pdf_text(_split_long_tokens(_normalize_inline(text)))
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
        safe_title = _sanitize_pdf_text(_split_long_tokens(_normalize_inline(title_text)))
        title_lines = pdf.multi_cell(text_width, line_height, safe_title, split_only=True)
        body_lines = []
        if body:
            safe_body = _sanitize_pdf_text(_split_long_tokens(_normalize_inline(body)))
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
                text = _sanitize_pdf_text(_split_long_tokens(_normalize_inline(cell)))
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
            pdf.multi_cell(0, 5, _sanitize_pdf_text(f"[image omitted: {path.name}]"))
            pdf.ln(1)

    first_heading = _first_heading_text(markdown)
    if title and not _headings_match(title, first_heading):
        write_heading(title, 1)

    lines = markdown.splitlines()
    in_code = False
    para: list[str] = []
    code: list[str] = []

    def flush_paragraph() -> None:
        nonlocal para
        if not para:
            return
        text = " ".join(s.strip() for s in para if s.strip())
        para = []
        if text:
            write_paragraph(text)

    def flush_code() -> None:
        nonlocal code
        if not code:
            return
        write_code_block(code)
        code = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                flush_paragraph()
                in_code = True
            i += 1
            continue
        if in_code:
            code.append(line)
            i += 1
            continue
        if not line.strip():
            flush_paragraph()
            i += 1
            continue

        if line.lstrip().startswith(">"):
            flush_paragraph()
            quote_lines: list[str] = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                chunk = lines[i].lstrip()[1:]
                if chunk.startswith(" "):
                    chunk = chunk[1:]
                quote_lines.append(chunk)
                i += 1
            if quote_lines:
                first = quote_lines[0].strip()
                admon = _ADMONITION_RE.match(first)
                if admon:
                    kind = admon.group(1).lower()
                    title_text = admon.group(2).strip() or kind.title()
                    write_admonition(kind, title_text, quote_lines[1:])
                else:
                    write_blockquote(quote_lines)
            continue

        if _looks_like_table_row(line):
            j = i
            rows: list[list[str]] = []
            while j < len(lines):
                raw = lines[j]
                if not raw.strip():
                    j += 1
                    continue
                if not _looks_like_table_row(raw):
                    break
                if _is_table_separator(raw):
                    j += 1
                    continue
                cells = _split_table_row(raw)
                if cells and any(c.strip() for c in cells):
                    rows.append(cells)
                j += 1
            if len(rows) >= 2:
                flush_paragraph()
                write_table(rows)
                i = j
                continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            text = heading.group(2).strip()
            write_heading(text, level)
            i += 1
            continue

        img_matches = list(_IMG_RE.finditer(line))
        if img_matches:
            flush_paragraph()
            for match in img_matches:
                path = _resolve_image_path(match.group(1), version=version)
                if path:
                    write_image(path)
            remainder = _IMG_RE.sub("", line).strip()
            if remainder:
                para.append(remainder)
            i += 1
            continue

        list_match = _LIST_RE.match(line)
        if list_match:
            flush_paragraph()
            bullet = list_match.group(1)
            text = list_match.group(2).strip()
            prefix = bullet if bullet.endswith(".") else "-"
            write_list_item(text, prefix)
            i += 1
            continue

        para.append(line)
        i += 1

    if in_code:
        flush_code()
    flush_paragraph()

    output = pdf.output(dest="S")
    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    return str(output).encode("latin-1", "replace")
