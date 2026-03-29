from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _load_bundle(run_dir: Path) -> Dict[str, Any]:
    required = [
        "summary.json",
        "per_category_metrics.json",
        "gate_decision_breakdown.json",
        "metadata.json",
    ]
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {run_dir}: {missing}")

    return {
        "summary": _load_json(run_dir / "summary.json"),
        "per_category": _load_json(run_dir / "per_category_metrics.json"),
        "gate": _load_json(run_dir / "gate_decision_breakdown.json"),
        "metadata": _load_json(run_dir / "metadata.json"),
    }


def _metric_delta(base: float, current: float) -> Dict[str, float]:
    return {
        "baseline": base,
        "current": current,
        "delta": current - base,
    }


def _build_core_deltas(base: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    base_summary = base["summary"]
    cur_summary = current["summary"]
    base_gate = base["gate"]
    cur_gate = current["gate"]

    return {
        "overall_accuracy": _metric_delta(
            _safe_float(base_summary.get("overall_accuracy")),
            _safe_float(cur_summary.get("overall_accuracy")),
        ),
        "clarify_rate": _metric_delta(
            _safe_float(base_gate.get("clarify_rate")),
            _safe_float(cur_gate.get("clarify_rate")),
        ),
        "false_refusal_rate": _metric_delta(
            _safe_float(base_gate.get("false_refusal_rate")),
            _safe_float(cur_gate.get("false_refusal_rate")),
        ),
        "false_answer_rate": _metric_delta(
            _safe_float(base_gate.get("false_answer_rate")),
            _safe_float(cur_gate.get("false_answer_rate")),
        ),
    }


def _build_category_deltas(base: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    base_cat = base["per_category"]
    cur_cat = current["per_category"]

    categories = sorted(set(base_cat.keys()) | set(cur_cat.keys()))
    out: Dict[str, Any] = {}
    for cat in categories:
        b = base_cat.get(cat, {})
        c = cur_cat.get(cat, {})
        out[cat] = {
            "count_baseline": _safe_int(b.get("count")),
            "count_current": _safe_int(c.get("count")),
            "accuracy": _metric_delta(
                _safe_float(b.get("accuracy")),
                _safe_float(c.get("accuracy")),
            ),
            "avg_top_score": _metric_delta(
                _safe_float(b.get("avg_top_score")),
                _safe_float(c.get("avg_top_score")),
            ),
            "avg_citations_count": _metric_delta(
                _safe_float(b.get("avg_citations_count")),
                _safe_float(c.get("avg_citations_count")),
            ),
        }
    return out


def _build_predicted_type_deltas(base: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    base_counts = base["gate"].get("predicted_type_counts", {})
    cur_counts = current["gate"].get("predicted_type_counts", {})
    keys = sorted(set(base_counts.keys()) | set(cur_counts.keys()))
    out: Dict[str, Any] = {}
    for k in keys:
        b = _safe_int(base_counts.get(k, 0))
        c = _safe_int(cur_counts.get(k, 0))
        out[k] = {"baseline": b, "current": c, "delta": c - b}
    return out


def _evaluate_guardrails(comparison: Dict[str, Any]) -> Dict[str, Any]:
    # Defaults come from execution spec:
    # - in-domain false refusal must not worsen > 0.05
    # - out-of-domain false answer must not worsen > 0.02
    # We map these to global false_refusal_rate / false_answer_rate proxies available in bundles.
    core = comparison["core_deltas"]
    false_refusal_delta = _safe_float(core["false_refusal_rate"].get("delta"))
    false_answer_delta = _safe_float(core["false_answer_rate"].get("delta"))

    fr_ok = false_refusal_delta <= 0.05
    fa_ok = false_answer_delta <= 0.02

    return {
        "false_refusal_guardrail": {
            "delta": false_refusal_delta,
            "max_allowed": 0.05,
            "passed": fr_ok,
        },
        "false_answer_guardrail": {
            "delta": false_answer_delta,
            "max_allowed": 0.02,
            "passed": fa_ok,
        },
        "overall_passed": fr_ok and fa_ok,
    }


def _format_report_markdown(
    baseline_dir: Path,
    current_dir: Path,
    comparison: Dict[str, Any],
) -> str:
    core = comparison["core_deltas"]
    cats = comparison["category_deltas"]
    pred = comparison["predicted_type_deltas"]
    guardrails = comparison["guardrails"]
    meta = comparison["metadata"]

    lines: List[str] = []
    lines.append("# Ablation Comparison Report")
    lines.append("")
    lines.append(f"- baseline_run_dir: {baseline_dir}")
    lines.append(f"- current_run_dir: {current_dir}")
    lines.append(f"- generated_utc: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- baseline_run_id: {meta.get('baseline_run_id', '')}")
    lines.append(f"- current_run_id: {meta.get('current_run_id', '')}")
    lines.append("")

    lines.append("## Core Metrics")
    lines.append(
        f"- overall_accuracy: base={core['overall_accuracy']['baseline']:.4f}, "
        f"current={core['overall_accuracy']['current']:.4f}, delta={core['overall_accuracy']['delta']:+.4f}"
    )
    lines.append(
        f"- clarify_rate: base={core['clarify_rate']['baseline']:.4f}, "
        f"current={core['clarify_rate']['current']:.4f}, delta={core['clarify_rate']['delta']:+.4f}"
    )
    lines.append(
        f"- false_refusal_rate: base={core['false_refusal_rate']['baseline']:.4f}, "
        f"current={core['false_refusal_rate']['current']:.4f}, delta={core['false_refusal_rate']['delta']:+.4f}"
    )
    lines.append(
        f"- false_answer_rate: base={core['false_answer_rate']['baseline']:.4f}, "
        f"current={core['false_answer_rate']['current']:.4f}, delta={core['false_answer_rate']['delta']:+.4f}"
    )
    lines.append("")

    lines.append("## Predicted Type Counts")
    for k in sorted(pred.keys()):
        row = pred[k]
        lines.append(
            f"- {k}: base={row['baseline']}, current={row['current']}, delta={row['delta']:+d}"
        )
    lines.append("")

    lines.append("## Per-Category Accuracy")
    lines.append("| Category | Baseline Count | Current Count | Baseline Acc | Current Acc | Delta |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for cat in sorted(cats.keys()):
        row = cats[cat]
        acc = row["accuracy"]
        lines.append(
            f"| {cat} | {row['count_baseline']} | {row['count_current']} | "
            f"{acc['baseline']:.4f} | {acc['current']:.4f} | {acc['delta']:+.4f} |"
        )
    lines.append("")

    lines.append("## Guardrail Check")
    fr = guardrails["false_refusal_guardrail"]
    fa = guardrails["false_answer_guardrail"]
    lines.append(
        f"- false_refusal_guardrail: passed={fr['passed']}, "
        f"delta={fr['delta']:+.4f}, max_allowed={fr['max_allowed']:+.4f}"
    )
    lines.append(
        f"- false_answer_guardrail: passed={fa['passed']}, "
        f"delta={fa['delta']:+.4f}, max_allowed={fa['max_allowed']:+.4f}"
    )
    lines.append(f"- overall_guardrail_status: {'PASS' if guardrails['overall_passed'] else 'FAIL'}")
    lines.append("")

    lines.append("## Final Decision")
    decision = "GO" if guardrails["overall_passed"] else "NO-GO"
    lines.append(f"- promotion_recommendation: {decision}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two ablation run bundles and generate deltas")
    parser.add_argument("--baseline-run", required=True, help="Path to baseline run directory")
    parser.add_argument("--current-run", required=True, help="Path to current run directory")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for comparison artifacts. Defaults to current-run/comparison_vs_<baseline_run_id>",
    )
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_run).resolve()
    current_dir = Path(args.current_run).resolve()

    baseline = _load_bundle(baseline_dir)
    current = _load_bundle(current_dir)

    comparison: Dict[str, Any] = {
        "metadata": {
            "baseline_run_id": baseline.get("metadata", {}).get("run_id", baseline_dir.name),
            "current_run_id": current.get("metadata", {}).get("run_id", current_dir.name),
            "baseline_run_dir": str(baseline_dir),
            "current_run_dir": str(current_dir),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
        "core_deltas": _build_core_deltas(baseline, current),
        "category_deltas": _build_category_deltas(baseline, current),
        "predicted_type_deltas": _build_predicted_type_deltas(baseline, current),
    }
    comparison["guardrails"] = _evaluate_guardrails(comparison)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else current_dir / f"comparison_vs_{comparison['metadata']['baseline_run_id']}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_json = output_dir / "comparison.json"
    comparison_json.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    report_md = output_dir / "report.md"
    report_md.write_text(
        _format_report_markdown(baseline_dir, current_dir, comparison),
        encoding="utf-8",
    )

    print(f"[OK] Wrote comparison JSON: {comparison_json}")
    print(f"[OK] Wrote comparison report: {report_md}")


if __name__ == "__main__":
    main()
