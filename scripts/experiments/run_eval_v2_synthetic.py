from __future__ import annotations

from collections import Counter
import argparse
from pathlib import Path
from typing import Any, Dict, List
import json
import re
import sys

# Ensure repo root is on sys.path when this file is executed directly.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.rag.pipeline import RAGPipeline
from src.monitoring.mlflow_tracking import log_eval_tracking, resolve_tracking_uri
from src.utils.jsonl import iter_jsonl, write_jsonl


def _preview(text: str, limit: int = 180) -> str:
    return (text or "").replace("\n", " ")[:limit]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_optional(values: List[float | None]) -> float:
    filtered = [float(v) for v in values if isinstance(v, (int, float))]
    return _mean(filtered)


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def _tokenize(text: str) -> List[str]:
    return [
        t
        for t in re.findall(r"[a-z0-9_]+", (text or "").lower())
        if len(t) > 1 and t not in _STOPWORDS
    ]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _overlap_f1(text_a: str, text_b: str) -> float:
    a = set(_tokenize(text_a))
    b = set(_tokenize(text_b))
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if overlap == 0:
        return 0.0
    precision = overlap / len(b)
    recall = overlap / len(a)
    if precision + recall == 0:
        return 0.0
    return _clip01((2.0 * precision * recall) / (precision + recall))


def _answer_relevancy_score(query: str, answer: str) -> float:
    return _overlap_f1(query, answer)


def _faithfulness_score(*, answer: str, citations_count: int, top_score: float) -> float:
    # Eval-only heuristic: combine citation presence with retrieval confidence.
    citation_signal = 1.0 if citations_count > 0 else 0.0
    confidence_signal = _clip01(top_score)
    has_content_signal = 1.0 if (answer or "").strip() else 0.0
    return _clip01((0.55 * citation_signal) + (0.35 * confidence_signal) + (0.10 * has_content_signal))


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]

    rank = (p / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def _category_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    categories = sorted({r.get("category", "unknown") for r in rows})

    for cat in categories:
        cat_rows = [r for r in rows if r.get("category") == cat]
        total = len(cat_rows)
        correct = sum(1 for r in cat_rows if r.get("expected_type") == r.get("predicted_type"))
        pred_counts = Counter(r.get("predicted_type", "unknown") for r in cat_rows)
        answer_cat_rows = [r for r in cat_rows if r.get("predicted_type") == "answer"]
        faithfulness_all = [
            float(r.get("faithfulness")) if isinstance(r.get("faithfulness"), (int, float)) else 0.0
            for r in cat_rows
        ]
        answer_relevancy_all = [
            float(r.get("answer_relevancy")) if isinstance(r.get("answer_relevancy"), (int, float)) else 0.0
            for r in cat_rows
        ]
        avg_faithfulness_answer_only = _mean_optional([r.get("faithfulness") for r in answer_cat_rows])
        avg_answer_relevancy_answer_only = _mean_optional([r.get("answer_relevancy") for r in answer_cat_rows])

        out[cat] = {
            "count": total,
            "accuracy": (correct / total) if total else 0.0,
            "predicted_type_counts": dict(pred_counts),
            "avg_top_score": _mean([float(r.get("top_score", 0.0)) for r in cat_rows]),
            "avg_citations_count": _mean([float(r.get("citations_count", 0.0)) for r in cat_rows]),
            "avg_faithfulness": avg_faithfulness_answer_only,
            "avg_answer_relevancy": avg_answer_relevancy_answer_only,
            "avg_faithfulness_answer_only": avg_faithfulness_answer_only,
            "avg_answer_relevancy_answer_only": avg_answer_relevancy_answer_only,
            "avg_faithfulness_all_predictions": _mean(faithfulness_all),
            "avg_answer_relevancy_all_predictions": _mean(answer_relevancy_all),
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
    parser.add_argument(
        "--mlflow-experiment",
        default="eka-eval",
        help="MLflow experiment name for tracking runs.",
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Disable MLflow tracking for this run.",
    )
    parser.add_argument(
        "--ablation-version",
        default="",
        help="Optional version label (e.g., H0_clean_gate, v3_intent_clf) for MLflow grouping.",
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
        query_text = str(item.get("query", ""))
        answer_text = str(out.get("answer", ""))
        predicted_type = out.get("type")
        citations_count = len(out.get("citations", []))
        top_score = float(out.get("meta", {}).get("top_score", 0.0))

        faithfulness = None
        answer_relevancy = None
        if predicted_type == "answer":
            faithfulness = _faithfulness_score(
                answer=answer_text,
                citations_count=citations_count,
                top_score=top_score,
            )
            answer_relevancy = _answer_relevancy_score(query_text, answer_text)

        row = {
            "id": item.get("id"),
            "category": item.get(args.category_field, item.get("category")),
            "query": query_text,
            "expected_type": item.get(args.expected_type_field, item.get("expected_type")),
            "predicted_type": predicted_type,
            "citations_count": citations_count,
            "top_score": top_score,
            "latency_ms_total": float(out.get("meta", {}).get("latency_ms_total", 0.0)),
            "latency_ms_retrieval": float(out.get("meta", {}).get("latency_ms_retrieval", 0.0)),
            "latency_ms_generation": float(out.get("meta", {}).get("latency_ms_generation", 0.0)),
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
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "answer_preview": _preview(answer_text),
        }
        rows.append(row)

    write_jsonl(out_results, rows, append=False)

    total = len(rows)
    correct = sum(1 for r in rows if r.get("expected_type") == r.get("predicted_type"))
    pred_counts = Counter(r.get("predicted_type", "unknown") for r in rows)
    lat_total = [float(r.get("latency_ms_total", 0.0)) for r in rows]
    lat_retrieval = [float(r.get("latency_ms_retrieval", 0.0)) for r in rows]
    lat_generation = [float(r.get("latency_ms_generation", 0.0)) for r in rows]
    answer_rows = [r for r in rows if r.get("predicted_type") == "answer"]
    avg_faithfulness_answer_only = _mean_optional([r.get("faithfulness") for r in answer_rows])
    avg_answer_relevancy_answer_only = _mean_optional([r.get("answer_relevancy") for r in answer_rows])
    avg_faithfulness_all_predictions = _mean(
        [float(r.get("faithfulness")) if isinstance(r.get("faithfulness"), (int, float)) else 0.0 for r in rows]
    )
    avg_answer_relevancy_all_predictions = _mean(
        [
            float(r.get("answer_relevancy"))
            if isinstance(r.get("answer_relevancy"), (int, float))
            else 0.0
            for r in rows
        ]
    )

    summary: Dict[str, Any] = {
        "total_queries": total,
        "total_answer_predictions": len(answer_rows),
        "overall_accuracy": (correct / total) if total else 0.0,
        "predicted_type_counts": dict(pred_counts),
        "avg_latency_ms_total": _mean(lat_total),
        "avg_latency_ms_retrieval": _mean(lat_retrieval),
        "avg_latency_ms_generation": _mean(lat_generation),
        "p50_latency_ms_total": _percentile(lat_total, 50.0),
        "p95_latency_ms_total": _percentile(lat_total, 95.0),
        "avg_faithfulness": avg_faithfulness_answer_only,
        "avg_answer_relevancy": avg_answer_relevancy_answer_only,
        "avg_faithfulness_answer_only": avg_faithfulness_answer_only,
        "avg_answer_relevancy_answer_only": avg_answer_relevancy_answer_only,
        "avg_faithfulness_all_predictions": avg_faithfulness_all_predictions,
        "avg_answer_relevancy_all_predictions": avg_answer_relevancy_all_predictions,
        "category_summary": _category_summary(rows),
        "confusion_summary": _confusion_summary(rows),
        "worst_mismatches": _worst_mismatches(rows, n=5),
        "notes": [
            "Evaluation-only run over eval_v2 synthetic scaffold.",
            "No retrieval/generation/confidence logic changes were applied by this script.",
            "Faithfulness and answer relevancy are heuristic eval-only signals; not used for phase-gate promotion.",
            f"Category field: {args.category_field}",
            f"Expected type field: {args.expected_type_field}",
        ],
    }

    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.disable_mlflow:
        tracking_uri = log_eval_tracking(
            repo_root=repo_root,
            experiment_name=args.mlflow_experiment,
            run_name=f"eval::{Path(args.dataset).name}",
            params={
                "dataset": args.dataset,
                "category_field": args.category_field,
                "expected_type_field": args.expected_type_field,
                "results_path": args.results,
                "summary_path": args.summary,
            },
            tags={
                "run_type": "eval_only",
                "script": "run_eval_v2_synthetic.py",
                "tracking_uri": resolve_tracking_uri(repo_root),
                "ablation_version": args.ablation_version,
            },
            summary=summary,
            summary_path=out_summary,
            results_path=out_results,
        )
        if tracking_uri:
            print(f"[OK] MLflow tracked: {tracking_uri}")

    print(f"[OK] Wrote eval_v2 results: {out_results}")
    print(f"[OK] Wrote eval_v2 summary: {out_summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()