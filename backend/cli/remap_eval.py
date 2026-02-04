from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import DEFAULT_VERSION


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _as_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]


def _load_pages_index(pages_path: Path, version: str) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    by_page_id: dict[str, dict[str, str]] = {}
    by_title: dict[str, dict[str, str]] = {}
    if not pages_path.exists():
        raise FileNotFoundError(f"pages.jsonl not found: {pages_path}")
    with pages_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("version")) != version:
                continue
            doc_id = str(row.get("doc_id") or "")
            page_id = str(row.get("page_id") or "")
            page_title = str(row.get("page_title") or "")
            if not doc_id or not page_id:
                continue
            by_page_id.setdefault(doc_id, {})[page_id] = page_title
            if page_title:
                by_title.setdefault(doc_id, {})[_normalize(page_title)] = page_id
    return by_page_id, by_title


def main() -> None:
    ap = argparse.ArgumentParser(description="Remap eval dataset to a different doc_id using page titles.")
    ap.add_argument("--dataset", required=True, help="Path to eval JSON dataset")
    ap.add_argument("--target-doc", required=True, help="Target doc_id to map to")
    ap.add_argument("--version", default=DEFAULT_VERSION, help="Datastore version (for pages.jsonl)")
    ap.add_argument("--source-doc", default="", help="Source doc_id (defaults to dataset meta.doc_id)")
    ap.add_argument("--output", required=True, help="Output dataset path")
    ap.add_argument("--annotate", action="store_true", help="Include mapping details on each question")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    meta = data.get("meta") or {}
    source_doc = args.source_doc or str(meta.get("doc_id") or "")
    if not source_doc:
        raise SystemExit("source doc_id not found: pass --source-doc or ensure meta.doc_id exists")

    pages_path = REPO_ROOT / "datastore" / args.version / "pages.jsonl"
    by_page_id, by_title = _load_pages_index(pages_path, args.version)
    source_titles = by_page_id.get(source_doc, {})
    target_titles = by_title.get(args.target_doc, {})
    if not source_titles:
        raise SystemExit(f"No pages found for source doc_id={source_doc}")
    if not target_titles:
        raise SystemExit(f"No pages found for target doc_id={args.target_doc}")

    questions = data.get("questions") or []
    for q in questions:
        expect = q.get("expect") or {}
        page_ids = _as_list(expect.get("page_ids") or expect.get("page_id"))
        mapped_ids: list[str] = []
        mapping_info: list[dict[str, Any]] = []
        for pid in page_ids:
            title = source_titles.get(pid, "")
            mapped = ""
            if title:
                mapped = target_titles.get(_normalize(title), "")
            if mapped:
                mapped_ids.append(mapped)
            else:
                mapped_ids.append(pid)
            if args.annotate:
                mapping_info.append({"source_page_id": pid, "title": title, "mapped_page_id": mapped or ""})

        if page_ids:
            expect["page_ids"] = mapped_ids
        expect["doc_id"] = args.target_doc
        if args.annotate:
            expect["mapping"] = mapping_info
        q["expect"] = expect

    meta["source_doc_id"] = source_doc
    meta["doc_id"] = args.target_doc
    data["meta"] = meta

    out_path = Path(args.output)
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
