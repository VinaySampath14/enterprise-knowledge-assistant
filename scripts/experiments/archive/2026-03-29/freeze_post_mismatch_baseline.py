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

from src.utils.jsonl import iter_jsonl


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


def _build_summary_from_results(
    dataset_path: Path,
    results_path: Path,
    *,
    category_field: str,
    expected_type_field: str,
) -> Dict[str, Any]:
    expected_by_id: Dict[str, Dict[str, Any]] = {}
    for item in iter_jsonl(dataset_path):
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            continue
        expected_by_id[item_id] = {
            "id": item_id,
            "query": item.get("query"),
            "category": item.get(category_field, item.get("category", "unknown")),
            "expected_type": item.get(expected_type_field, item.get("expected_type", "unknown")),
        }

    rows: List[Dict[str, Any]] = []
    for item in iter_jsonl(results_path):
        item_id = str(item.get("id", "")).strip()
        if not item_id or item_id not in expected_by_id:
            continue
        expected = expected_by_id[item_id]
        rows.append(
            {
                "id": item_id,
                "query": expected.get("query"),
                "category": expected.get("category"),
                "expected_type": expected.get("expected_type"),
                "predicted_type": item.get("predicted_type", item.get("pred_type", "unknown")),
                "top_score": float(item.get("top_score", 0.0)),
                "citations_count": int(item.get("citations_count", item.get("num_sources", 0))),
                "answer_preview": item.get("answer_preview", ""),
            }
        )

    total = len(rows)
    correct = sum(1 for r in rows if r.get("expected_type") == r.get("predicted_type"))
    pred_counts = Counter(r.get("predicted_type", "unknown") for r in rows)

    return {
        "total_queries": total,
        "overall_accuracy": (correct / total) if total else 0.0,
        "predicted_type_counts": dict(pred_counts),
        "category_summary": _category_summary(rows),
        "confusion_summary": _confusion_summary(rows),
        "worst_mismatches": _worst_mismatches(rows),
    }


def _extract_category_accuracy(summary: Dict[str, Any]) -> Dict[str, float]:
    return {
        cat: float(vals.get("accuracy", 0.0))
        for cat, vals in summary.get("category_summary", {}).items()
    }


def _comparison(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    before_acc = _extract_category_accuracy(before)
    after_acc = _extract_category_accuracy(after)
    all_cats = sorted(set(before_acc.keys()) | set(after_acc.keys()))
    category_delta = {
        c: {
            "before": before_acc.get(c, 0.0),
            "after": after_acc.get(c, 0.0),
            "delta": after_acc.get(c, 0.0) - before_acc.get(c, 0.0),
        }
        for c in all_cats
    }

    return {
        "overall_accuracy": {
            "before": float(before.get("overall_accuracy", 0.0)),
            "after": float(after.get("overall_accuracy", 0.0)),
            "delta": float(after.get("overall_accuracy", 0.0)) - float(before.get("overall_accuracy", 0.0)),
        },
        "predicted_type_counts": {
            "before": before.get("predicted_type_counts", {}),
            "after": after.get("predicted_type_counts", {}),
        },
        "confusion_summary": {
            "before": before.get("confusion_summary", {}),
            "after": after.get("confusion_summary", {}),
        },
        "category_accuracy": category_delta,
    }


def _get_acc(summary: Dict[str, Any], category: str) -> float:
    cat = summary.get("category_summary", {}).get(category, {})
    return float(cat.get("accuracy", 0.0))


def _build_conclusion(manual_before: Dict[str, Any], manual_after: Dict[str, Any], synthetic_before: Dict[str, Any], synthetic_after: Dict[str, Any]) -> Dict[str, Any]:
    near_before = _get_acc(synthetic_before, "near_domain_should_refuse")
    near_after = _get_acc(synthetic_after, "near_domain_should_refuse")

    rec_before = _get_acc(manual_before, "recoverable_should_clarify")
    rec_after = _get_acc(manual_after, "recoverable_should_clarify")

    in_domain_before = _get_acc(synthetic_before, "in_domain_answerable")
    in_domain_after = _get_acc(synthetic_after, "in_domain_answerable")
    in_scope_before = _get_acc(manual_before, "in_scope_should_answer")
    in_scope_after = _get_acc(manual_after, "in_scope_should_answer")

    ood_syn_before = _get_acc(synthetic_before, "out_of_domain_unanswerable")
    ood_syn_after = _get_acc(synthetic_after, "out_of_domain_unanswerable")
    ood_manual_before = _get_acc(manual_before, "out_of_scope_should_refuse")
    ood_manual_after = _get_acc(manual_after, "out_of_scope_should_refuse")

    improved = []
    if near_after > near_before:
        improved.append("near_domain_should_refuse improved")

    remaining = []
    if rec_after < 1.0:
        remaining.append("recoverable_should_clarify still under-target")
    if _get_acc(manual_after, "python_general_out_of_scope") < 1.0:
        remaining.append("python_general_out_of_scope still has clarify leakage")

    return {
        "near_domain_should_refuse": {
            "before": near_before,
            "after": near_after,
            "improved": near_after > near_before,
        },
        "recoverable_should_clarify": {
            "before": rec_before,
            "after": rec_after,
            "remained_correct": rec_after >= rec_before,
        },
        "in_domain_answer_behavior_stable": {
            "synthetic_in_domain_answerable": {
                "before": in_domain_before,
                "after": in_domain_after,
            },
            "manual_in_scope_should_answer": {
                "before": in_scope_before,
                "after": in_scope_after,
            },
            "stable": (in_domain_after >= in_domain_before) and (in_scope_after >= in_scope_before),
        },
        "ood_refusal_stable": {
            "synthetic_out_of_domain_unanswerable": {
                "before": ood_syn_before,
                "after": ood_syn_after,
            },
            "manual_out_of_scope_should_refuse": {
                "before": ood_manual_before,
                "after": ood_manual_after,
            },
            "stable": (ood_syn_after >= ood_syn_before) and (ood_manual_after >= ood_manual_before),
        },
        "improved_points": improved,
        "remaining_gaps": remaining,
    }


def _to_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Post-Change Baseline Freeze")
    lines.append("")
    lines.append("## Scope")
    lines.append("- Evaluation/reporting only (no threshold or logic changes in this pass).")
    lines.append("- Benchmarks: refined manual/diagnostic + refined synthetic.")
    lines.append("")

    for bench_name in ("manual_refined", "synthetic_refined"):
        comp = report["comparison"][bench_name]
        lines.append(f"## {bench_name}")
        lines.append(f"- overall_accuracy before={comp['overall_accuracy']['before']:.4f} after={comp['overall_accuracy']['after']:.4f} delta={comp['overall_accuracy']['delta']:+.4f}")
        lines.append(f"- predicted_type_counts before={comp['predicted_type_counts']['before']} after={comp['predicted_type_counts']['after']}")
        lines.append(f"- confusion_summary before={comp['confusion_summary']['before']} after={comp['confusion_summary']['after']}")
        lines.append("- per-category accuracy:")
        for cat, vals in comp["category_accuracy"].items():
            lines.append(
                f"  - {cat}: before={vals['before']:.4f} after={vals['after']:.4f} delta={vals['delta']:+.4f}"
            )
        lines.append("")

    concl = report["conclusion"]
    lines.append("## Target Checks")
    lines.append(
        "- near_domain_should_refuse: "
        f"before={concl['near_domain_should_refuse']['before']:.4f}, "
        f"after={concl['near_domain_should_refuse']['after']:.4f}, "
        f"improved={concl['near_domain_should_refuse']['improved']}"
    )
    lines.append(
        "- recoverable_should_clarify: "
        f"before={concl['recoverable_should_clarify']['before']:.4f}, "
        f"after={concl['recoverable_should_clarify']['after']:.4f}, "
        f"remained_correct={concl['recoverable_should_clarify']['remained_correct']}"
    )
    lines.append(
        "- in-domain answer behavior stable: "
        f"{concl['in_domain_answer_behavior_stable']['stable']}"
    )
    lines.append("- OOD refusal stable: " f"{concl['ood_refusal_stable']['stable']}")
    lines.append("")

    lines.append("## Conclusion")
    if concl.get("improved_points"):
        lines.append("- Improved: " + "; ".join(concl["improved_points"]))
    else:
        lines.append("- Improved: no major gains detected versus archived baseline.")

    if concl.get("remaining_gaps"):
        lines.append("- Remaining: " + "; ".join(concl["remaining_gaps"]))
    else:
        lines.append("- Remaining: no major residual gaps flagged by these benchmarks.")

    return "\n".join(lines) + "\n"


def main() -> None:
    root = Path(__file__).resolve().parents[2]

    manual_dataset = root / "eval" / "diagnostic_citation_signal_questions_refined.jsonl"
    synthetic_dataset = root / "eval_v2" / "synthetic_scaffold_dataset_refined.jsonl"

    manual_before_results = root / "artifacts" / "archive" / "2026-03-22" / "eval" / "diagnostic_citation_signal_results.jsonl"
    manual_after_results = root / "eval" / "diagnostic_refined_results.jsonl"

    synthetic_before_results = root / "artifacts" / "archive" / "2026-03-22" / "eval_v2" / "synthetic_refined_results.jsonl"
    synthetic_after_results = root / "eval_v2" / "synthetic_refined_results.jsonl"

    manual_before = _build_summary_from_results(
        manual_dataset,
        manual_before_results,
        category_field="refined_category",
        expected_type_field="expected_type_refined",
    )
    manual_after = _build_summary_from_results(
        manual_dataset,
        manual_after_results,
        category_field="refined_category",
        expected_type_field="expected_type_refined",
    )

    synthetic_before = _build_summary_from_results(
        synthetic_dataset,
        synthetic_before_results,
        category_field="refined_category",
        expected_type_field="expected_type_refined",
    )
    synthetic_after = _build_summary_from_results(
        synthetic_dataset,
        synthetic_after_results,
        category_field="refined_category",
        expected_type_field="expected_type_refined",
    )

    report = {
        "artifacts": {
            "manual_before_results": str(manual_before_results),
            "manual_after_results": str(manual_after_results),
            "synthetic_before_results": str(synthetic_before_results),
            "synthetic_after_results": str(synthetic_after_results),
        },
        "before": {
            "manual_refined": manual_before,
            "synthetic_refined": synthetic_before,
        },
        "after": {
            "manual_refined": manual_after,
            "synthetic_refined": synthetic_after,
        },
        "comparison": {
            "manual_refined": _comparison(manual_before, manual_after),
            "synthetic_refined": _comparison(synthetic_before, synthetic_after),
        },
        "conclusion": _build_conclusion(manual_before, manual_after, synthetic_before, synthetic_after),
    }

    out_dir = root / "artifacts" / "baselines" / "post_mismatch_subtype"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "baseline_comparison.json"
    out_md = out_dir / "baseline_comparison.md"

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(report), encoding="utf-8")

    print(f"[OK] Wrote baseline comparison JSON: {out_json}")
    print(f"[OK] Wrote baseline comparison Markdown: {out_md}")


if __name__ == "__main__":
    main()
