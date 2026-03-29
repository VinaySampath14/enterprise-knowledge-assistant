from __future__ import annotations

from collections import Counter
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


def _judgement(zero_count: int, bad_count: int) -> str:
    if zero_count == 0:
        return "No observed answer-with-zero-citations cases in this set; signal cannot be stress-tested here."

    ratio = bad_count / zero_count
    if ratio >= 0.8:
        return "Strongly justified: most answer-with-zero-citations cases looked bad/out-of-scope."
    if ratio >= 0.5:
        return "Partially justified: answer-with-zero-citations is a useful warning signal but not sufficient alone."
    return "Weakly justified: many answer-with-zero-citations cases did not look bad/out-of-scope."


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
    repo_root = Path(__file__).resolve().parents[2]
    q_path = repo_root / "eval" / "diagnostic_citation_signal_questions.jsonl"
    out_results = repo_root / "eval" / "diagnostic_citation_signal_results.jsonl"
    out_summary = repo_root / "eval" / "diagnostic_citation_signal_summary.json"

    pipeline = RAGPipeline(repo_root)

    rows: List[Dict[str, Any]] = []

    for q in iter_jsonl(q_path):
        out = pipeline.run(q["query"])

        row = {
            "id": q["id"],
            "category": q["category"],
            "query": q["query"],
            "expected_type": q["expected_type"],
            "predicted_type": out.get("type"),
            "citations_count": len(out.get("citations", [])),
            "top_score": float(out.get("meta", {}).get("top_score", 0.0)),
            "answer_preview": _preview(out.get("answer", "")),
        }
        rows.append(row)

    write_jsonl(out_results, rows, append=False)

    pred_counts = Counter(r["predicted_type"] for r in rows)
    total = len(rows)
    correct = sum(1 for r in rows if r["expected_type"] == r["predicted_type"])
    answer_rows = [r for r in rows if r["predicted_type"] == "answer"]
    answer_zero = [r for r in answer_rows if r["citations_count"] == 0]

    # "Bad/out-of-scope" proxy for this diagnostic:
    # answer predicted for anything not expected to be answer.
    answer_zero_bad = [r for r in answer_zero if r["expected_type"] != "answer"]

    summary: Dict[str, Any] = {
        "total_queries": total,
        "overall_accuracy": (correct / total) if total else 0.0,
        "predicted_type_counts": dict(pred_counts),
        "category_summary": _category_summary(rows),
        "confusion_summary": _confusion_summary(rows),
        "answer_count": len(answer_rows),
        "answer_with_zero_citations_count": len(answer_zero),
        "answer_with_zero_citations_bad_or_out_of_scope_count": len(answer_zero_bad),
        "answer_with_zero_citations_bad_or_out_of_scope_ratio": (
            (len(answer_zero_bad) / len(answer_zero)) if answer_zero else 0.0
        ),
        "judgement": _judgement(len(answer_zero), len(answer_zero_bad)),
        "zero_citation_answer_examples": [
            {
                "query": r["query"],
                "expected_type": r["expected_type"],
                "predicted_type": r["predicted_type"],
                "top_score": r["top_score"],
                "answer_preview": r["answer_preview"],
            }
            for r in answer_zero[:8]
        ],
        "worst_mismatches": _worst_mismatches(rows, n=5),
    }

    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Wrote results: {out_results}")
    print(f"[OK] Wrote summary: {out_summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
