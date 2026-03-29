from __future__ import annotations

from collections import Counter
import argparse
from pathlib import Path
from typing import Any, Dict, List
import json
import sys

# Ensure repo root is on sys.path when this file is executed directly.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.rag.pipeline import RAGPipeline
from src.utils.jsonl import iter_jsonl, write_jsonl


def _preview(text: str, limit: int = 180) -> str:
    return (text or "").replace("\n", " ")[:limit]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _category_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    categories = sorted({r.get("category", "unknown") for r in rows})

    for cat in categories:
        cat_rows = [r for r in rows if r.get("category") == cat]
        total = len(cat_rows)
        correct = sum(1 for r in cat_rows if r.get("expected_type") == r.get("predicted_type"))
        pred_counts = Counter(r.get("predicted_type", "unknown") for r in cat_rows)

        out[cat] = {
            "count": total,
            "accuracy": (correct / total) if total else 0.0,
            "predicted_type_counts": dict(pred_counts),
            "avg_top_score": _mean([float(r.get("top_score", 0.0)) for r in cat_rows]),
            "avg_citations_count": _mean([float(r.get("citations_count", 0.0)) for r in cat_rows]),
        }

    return out


def _confusion_summary(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    return {
        "expected_answer_predicted_refuse": sum(
            1
            for r in rows
            if r.get("expected_type") == "answer" and r.get("predicted_type") == "refuse"
        ),
        "expected_answer_predicted_clarify": sum(
            1
            for r in rows
            if r.get("expected_type") == "answer" and r.get("predicted_type") == "clarify"
        ),
        "expected_refuse_predicted_answer": sum(
            1
            for r in rows
            if r.get("expected_type") == "refuse" and r.get("predicted_type") == "answer"
        ),
        "expected_refuse_predicted_clarify": sum(
            1
            for r in rows
            if r.get("expected_type") == "refuse" and r.get("predicted_type") == "clarify"
        ),
        "expected_clarify_predicted_answer": sum(
            1
            for r in rows
            if r.get("expected_type") == "clarify" and r.get("predicted_type") == "answer"
        ),
        "expected_clarify_predicted_refuse": sum(
            1
            for r in rows
            if r.get("expected_type") == "clarify" and r.get("predicted_type") == "refuse"
        ),
    }


def _worst_mismatches(rows: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
    mismatches = [r for r in rows if r.get("expected_type") != r.get("predicted_type")]
    mismatches = sorted(mismatches, key=lambda r: float(r.get("top_score", 0.0)), reverse=True)
    return [
        {
            "id": r.get("id"),
            "category": r.get("category"),
            "query": r.get("query"),
            "expected_type": r.get("expected_type"),
            "predicted_type": r.get("predicted_type"),
            "top_score": float(r.get("top_score", 0.0)),
            "citations_count": int(r.get("citations_count", 0)),
            "answer_preview": r.get("answer_preview", ""),
        }
        for r in mismatches[:n]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval_v2 dataset through current pipeline.")
    parser.add_argument(
        "--dataset",
        default="eval_v2/synthetic_scaffold_dataset.jsonl",
        help="Input JSONL benchmark dataset path (relative to repo root).",
    )
    parser.add_argument(
        "--results",
        default="eval_v2/synthetic_results.jsonl",
        help="Output JSONL results path (relative to repo root).",
    )
    parser.add_argument(
        "--summary",
        default="eval_v2/synthetic_summary.json",
        help="Output summary JSON path (relative to repo root).",
    )
    parser.add_argument(
        "--category-field",
        default="category",
        help="Dataset field to use as category (e.g., category or refined_category).",
    )
    parser.add_argument(
        "--expected-type-field",
        default="expected_type",
        help="Dataset field to use as expected label (e.g., expected_type or expected_type_refined).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    dataset_path = repo_root / args.dataset
    out_results = repo_root / args.results
    out_summary = repo_root / args.summary

    pipeline = RAGPipeline(repo_root)

    rows: List[Dict[str, Any]] = []
    for item in iter_jsonl(dataset_path):
        out = pipeline.run(str(item.get("query", "")))

        row = {
            "id": item.get("id"),
            "category": item.get(args.category_field, item.get("category")),
            "query": item.get("query"),
            "expected_type": item.get(args.expected_type_field, item.get("expected_type")),
            "predicted_type": out.get("type"),
            "citations_count": len(out.get("citations", [])),
            "top_score": float(out.get("meta", {}).get("top_score", 0.0)),
            "second_score": float(out.get("meta", {}).get("second_score", 0.0)),
            "score_margin": float(out.get("meta", {}).get("score_margin", 0.0)),
            "gate_decision": out.get("meta", {}).get("gate_decision"),
            "gate_rationale": str(out.get("meta", {}).get("gate_rationale", ""))[:400],
            "gate_tie_breaker_fired": bool(out.get("meta", {}).get("gate_tie_breaker_fired", False)),
            "postgen_refusal_override_triggered": bool(
                out.get("meta", {}).get("postgen_refusal_override_triggered", False)
            ),
            "postgen_refusal_override_blocked": bool(
                out.get("meta", {}).get("postgen_refusal_override_blocked", False)
            ),
            "postgen_refusal_override_block_reasons": out.get("meta", {}).get(
                "postgen_refusal_override_block_reasons", []
            ),
            "answer_preview": _preview(str(out.get("answer", ""))),
        }
        rows.append(row)

    write_jsonl(out_results, rows, append=False)

    total = len(rows)
    correct = sum(1 for r in rows if r.get("expected_type") == r.get("predicted_type"))
    pred_counts = Counter(r.get("predicted_type", "unknown") for r in rows)

    summary: Dict[str, Any] = {
        "total_queries": total,
        "overall_accuracy": (correct / total) if total else 0.0,
        "predicted_type_counts": dict(pred_counts),
        "category_summary": _category_summary(rows),
        "confusion_summary": _confusion_summary(rows),
        "worst_mismatches": _worst_mismatches(rows, n=5),
        "notes": [
            "Evaluation-only run over eval_v2 synthetic scaffold.",
            "No retrieval/generation/confidence logic changes were applied by this script.",
            f"Category field: {args.category_field}",
            f"Expected type field: {args.expected_type_field}",
        ],
    }

    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Wrote eval_v2 results: {out_results}")
    print(f"[OK] Wrote eval_v2 summary: {out_summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()