from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.config import DEFAULT_VERSION
from backend.app.datastore_search import SearchEngine


def _as_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return [str(value)]


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _match_keywords(text: str, must_all: list[str], must_any: list[str]) -> bool:
    if not must_all and not must_any:
        return False
    text_l = _normalize(text)
    if must_all:
        for kw in must_all:
            if _normalize(kw) not in text_l:
                return False
    if must_any:
        if not any(_normalize(kw) in text_l for kw in must_any):
            return False
    return True


def _first_rank(results: list[dict[str, Any]], predicate) -> int:
    for idx, row in enumerate(results, start=1):
        if predicate(row):
            return idx
    return 0


def _best_rank(*ranks: int) -> int:
    best = 0
    for r in ranks:
        if r <= 0:
            continue
        if best == 0 or r < best:
            best = r
    return best


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    questions = data.get("questions") or []
    meta = data.get("meta") or {}

    engine = SearchEngine(version=args.version)
    if not engine.is_ready():
        raise RuntimeError(f"Datastore index not found for version: {args.version}")

    results_out: list[dict[str, Any]] = []
    total = len(questions)
    hits = 0
    hit1 = 0
    hit3 = 0
    hit5 = 0
    mrr_sum = 0.0
    rank_sum = 0.0
    ranked = 0

    for q in questions:
        qid = str(q.get("id") or "")
        question = str(q.get("question") or "").strip()
        expect = q.get("expect") or {}
        expected_doc_ids = _as_list(expect.get("doc_id") or expect.get("doc_ids"))
        expected_page_ids = _as_list(expect.get("page_ids") or expect.get("page_id"))
        keywords_all = _as_list(expect.get("keywords_all"))
        keywords_any = _as_list(expect.get("keywords_any"))

        debug: dict[str, Any] | None = {} if args.debug else None
        retrieved = await engine.search(
            question,
            k=args.k,
            mode=args.mode,
            mmr_enabled=args.mmr,
            mmr_lambda=args.mmr_lambda,
            mmr_candidates=args.mmr_candidates or None,
            summary_enabled=args.summary,
            summary_k=args.summary_k or None,
            summary_max_pages=args.summary_max_pages or None,
            reranker_enabled=args.reranker,
            reranker_model=args.reranker_model or None,
            reranker_top_k=args.reranker_top_k or None,
            reranker_min_score=args.reranker_min_score,
            reranker_max_chars=args.reranker_max_chars or None,
            max_per_page=args.max_per_page,
            max_per_doc=args.max_per_doc,
            debug_out=debug,
        )

        def doc_pred(row: dict[str, Any]) -> bool:
            if not expected_doc_ids:
                return False
            return str(row.get("doc_id") or "") in expected_doc_ids

        def page_pred(row: dict[str, Any]) -> bool:
            if not expected_page_ids:
                return False
            if expected_doc_ids and str(row.get("doc_id") or "") not in expected_doc_ids:
                return False
            return str(row.get("page_id") or "") in expected_page_ids

        def kw_pred(row: dict[str, Any]) -> bool:
            text = " ".join(
                [
                    str(row.get("text") or ""),
                    " ".join(row.get("heading_path") or []),
                ]
            )
            return _match_keywords(text, keywords_all, keywords_any)

        rank_doc = _first_rank(retrieved, doc_pred)
        rank_page = _first_rank(retrieved, page_pred)
        rank_kw = _first_rank(retrieved, kw_pred)

        best_rank = _best_rank(rank_page, rank_doc, rank_kw)
        passed = False
        if expected_page_ids:
            passed = rank_page > 0
        elif expected_doc_ids:
            passed = rank_doc > 0
        else:
            passed = rank_kw > 0

        if passed:
            hits += 1
        if best_rank > 0:
            if best_rank <= 1:
                hit1 += 1
            if best_rank <= 3:
                hit3 += 1
            if best_rank <= 5:
                hit5 += 1
            mrr_sum += 1.0 / float(best_rank)
            rank_sum += float(best_rank)
            ranked += 1

        results_out.append(
            {
                "id": qid,
                "question": question,
                "expect": expect,
                "rank_doc": rank_doc,
                "rank_page": rank_page,
                "rank_keywords": rank_kw,
                "best_rank": best_rank,
                "pass": bool(passed),
                "top": [
                    {
                        "rank": idx + 1,
                        "doc_id": row.get("doc_id"),
                        "page_id": row.get("page_id"),
                        "heading_path": row.get("heading_path"),
                        "score": row.get("score"),
                    }
                    for idx, row in enumerate(retrieved[: min(5, len(retrieved))])
                ],
                "debug": debug,
            }
        )

    summary = {
        "total": total,
        "passed": hits,
        "pass_rate": (hits / total) if total else 0.0,
        "hit_at_1": (hit1 / total) if total else 0.0,
        "hit_at_3": (hit3 / total) if total else 0.0,
        "hit_at_5": (hit5 / total) if total else 0.0,
        "mrr": (mrr_sum / total) if total else 0.0,
        "avg_best_rank": (rank_sum / ranked) if ranked else 0.0,
    }

    return {
        "meta": meta,
        "config": {
            "version": args.version,
            "mode": args.mode,
            "k": args.k,
            "summary": args.summary,
            "summary_k": args.summary_k,
            "summary_max_pages": args.summary_max_pages,
            "mmr": args.mmr,
            "mmr_lambda": args.mmr_lambda,
            "mmr_candidates": args.mmr_candidates,
            "reranker": args.reranker,
            "reranker_model": args.reranker_model,
            "reranker_top_k": args.reranker_top_k,
            "reranker_min_score": args.reranker_min_score,
            "reranker_max_chars": args.reranker_max_chars,
            "max_per_page": args.max_per_page,
            "max_per_doc": args.max_per_doc,
        },
        "summary": summary,
        "results": results_out,
    }


def build_parser() -> argparse.ArgumentParser:
    default_dataset = (
        Path(__file__).resolve().parents[1] / "eval" / "evo_s_installation_eval.json"
    )
    ap = argparse.ArgumentParser(description="Evaluate retrieval quality on a question set.")
    ap.add_argument("--dataset", default=str(default_dataset), help="Path to JSON eval dataset")
    ap.add_argument("--version", default=DEFAULT_VERSION, help="Datastore version")
    ap.add_argument("--mode", default="bm25", choices=["bm25", "embedding", "hybrid"])
    ap.add_argument("--k", type=int, default=8, help="Number of chunks to retrieve")
    ap.add_argument("--summary", action="store_true", help="Enable summary prefilter")
    ap.add_argument("--summary-k", type=int, default=0, help="Summary candidate size")
    ap.add_argument("--summary-max-pages", type=int, default=0, help="Max pages from summary")
    ap.add_argument("--mmr", action="store_true", help="Enable MMR")
    ap.add_argument("--mmr-lambda", type=float, default=0.7)
    ap.add_argument("--mmr-candidates", type=int, default=0)
    ap.add_argument("--reranker", action="store_true", help="Enable reranker")
    ap.add_argument("--reranker-model", default="", help="Reranker model id")
    ap.add_argument("--reranker-top-k", type=int, default=0)
    ap.add_argument("--reranker-min-score", type=float, default=0.0)
    ap.add_argument("--reranker-max-chars", type=int, default=0)
    ap.add_argument("--max-per-page", type=int, default=0)
    ap.add_argument("--max-per-doc", type=int, default=0)
    ap.add_argument("--debug", action="store_true", help="Include debug info")
    ap.add_argument("--output", default="", help="Write full results to JSON")
    ap.add_argument("--fail-only", action="store_true", help="Only print failing questions")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    report = asyncio.run(evaluate(args))
    summary = report["summary"]

    print(
        f"Dataset: {report.get('meta', {}).get('name', 'eval')} | "
        f"mode={report['config']['mode']} k={report['config']['k']} "
        f"version={report['config']['version']}"
    )
    print(
        "Pass: {}/{} ({:.1f}%) | Hit@1: {:.1f}% | Hit@3: {:.1f}% | Hit@5: {:.1f}% | MRR: {:.3f}".format(
            summary["passed"],
            summary["total"],
            summary["pass_rate"] * 100.0,
            summary["hit_at_1"] * 100.0,
            summary["hit_at_3"] * 100.0,
            summary["hit_at_5"] * 100.0,
            summary["mrr"],
        )
    )

    for row in report["results"]:
        if args.fail_only and row["pass"]:
            continue
        status = "PASS" if row["pass"] else "FAIL"
        best = row["best_rank"] if row["best_rank"] else "-"
        print(f"[{status}] {row['id']} rank={best} :: {row['question']}")
        if not row["pass"]:
            for top in row.get("top", []):
                print(
                    "  - #{rank} {doc}/{page} score={score}".format(
                        rank=top.get("rank"),
                        doc=top.get("doc_id"),
                        page=top.get("page_id"),
                        score=top.get("score"),
                    )
                )

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
