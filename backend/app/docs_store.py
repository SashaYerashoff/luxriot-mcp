from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PageRecord:
    version: str
    doc_id: str
    doc_title: str
    page_id: str
    page_title: str
    heading_path: list[str]
    source_path: str
    anchor: str | None
    images: list[dict[str, Any]]
    markdown_path: str


class DocsStore:
    def __init__(self, *, version: str, datastore_dir: Path) -> None:
        self.version = version
        self.datastore_dir = datastore_dir
        self.root = datastore_dir / version
        self.pages_jsonl_path = self.root / "pages.jsonl"

        self._mtime: float | None = None
        self._doc_order: list[str] = []
        self._doc_titles: dict[str, str] = {}
        self._doc_pages: dict[str, list[PageRecord]] = {}
        self._pages: dict[tuple[str, str], PageRecord] = {}

    def is_ready(self) -> bool:
        return self.pages_jsonl_path.exists()

    def _safe_resolve(self, base: Path, rel: str) -> Path:
        candidate = (base / rel).resolve()
        if not str(candidate).startswith(str(base.resolve())):
            raise RuntimeError("Invalid markdown_path in catalog")
        return candidate

    def _load_if_needed(self) -> None:
        if not self.pages_jsonl_path.exists():
            raise FileNotFoundError(f"Missing pages catalog: {self.pages_jsonl_path}")

        mtime = self.pages_jsonl_path.stat().st_mtime
        if self._mtime == mtime and self._doc_order:
            return

        doc_order: list[str] = []
        doc_titles: dict[str, str] = {}
        doc_pages: dict[str, list[PageRecord]] = {}
        pages: dict[tuple[str, str], PageRecord] = {}

        with self.pages_jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RuntimeError(f"Invalid pages.jsonl at line {line_no}: {e}") from e

                doc_id = str(rec.get("doc_id") or "").strip()
                page_id = str(rec.get("page_id") or "").strip()
                if not doc_id or not page_id:
                    continue

                doc_title = str(rec.get("doc_title") or doc_id)
                page_title = str(rec.get("page_title") or page_id)
                heading_path = rec.get("heading_path")
                if not isinstance(heading_path, list):
                    heading_path = [doc_title, page_title]
                heading_path = [str(x) for x in heading_path if str(x).strip()]
                if not heading_path:
                    heading_path = [doc_title, page_title]

                source_path = str(rec.get("source_path") or "").strip()
                anchor = rec.get("anchor")
                anchor_str = str(anchor).strip() if isinstance(anchor, str) and anchor.strip() else None
                images = rec.get("images")
                if not isinstance(images, list):
                    images = []
                markdown_path = str(rec.get("markdown_path") or "").strip()
                if not markdown_path:
                    continue

                if doc_id not in doc_titles:
                    doc_titles[doc_id] = doc_title
                    doc_pages[doc_id] = []
                    doc_order.append(doc_id)

                page = PageRecord(
                    version=str(rec.get("version") or self.version),
                    doc_id=doc_id,
                    doc_title=doc_title,
                    page_id=page_id,
                    page_title=page_title,
                    heading_path=heading_path,
                    source_path=source_path,
                    anchor=anchor_str,
                    images=images,
                    markdown_path=markdown_path,
                )
                doc_pages[doc_id].append(page)
                pages[(doc_id, page_id)] = page

        self._mtime = mtime
        self._doc_order = doc_order
        self._doc_titles = doc_titles
        self._doc_pages = doc_pages
        self._pages = pages

    def list_docs(self) -> list[dict[str, Any]]:
        self._load_if_needed()
        out: list[dict[str, Any]] = []
        for doc_id in self._doc_order:
            title = self._doc_titles.get(doc_id) or doc_id
            page_count = len(self._doc_pages.get(doc_id) or [])
            out.append({"doc_id": doc_id, "doc_title": title, "page_count": page_count})
        return out

    def list_pages(self, doc_id: str) -> dict[str, Any]:
        self._load_if_needed()
        doc_id = str(doc_id or "").strip()
        if doc_id not in self._doc_pages:
            raise KeyError(doc_id)
        title = self._doc_titles.get(doc_id) or doc_id
        pages = self._doc_pages.get(doc_id) or []
        return {
            "doc_id": doc_id,
            "doc_title": title,
            "pages": [
                {
                    "page_id": p.page_id,
                    "page_title": p.page_title,
                    "heading_path": p.heading_path,
                    "source_path": p.source_path,
                    "anchor": p.anchor,
                }
                for p in pages
            ],
        }

    def get_page(self, doc_id: str, page_id: str) -> PageRecord:
        self._load_if_needed()
        key = (str(doc_id or "").strip(), str(page_id or "").strip())
        page = self._pages.get(key)
        if not page:
            raise KeyError(f"{key[0]}/{key[1]}")
        return page

    def read_markdown(self, page: PageRecord) -> str:
        md_path = self._safe_resolve(self.root, page.markdown_path)
        if not md_path.exists() or not md_path.is_file():
            raise FileNotFoundError(f"Markdown not found: {md_path}")
        return md_path.read_text(encoding="utf-8", errors="ignore")

