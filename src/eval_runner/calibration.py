from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
import json
import math

from src.config import load_app_config
from src.utils.jsonl import iter_jsonl


@dataclass
class EvalRow:
    id: str
    query: str
    category: str
    expected_type: str
    predicted_type: str
    top_score: float
    source_file: str


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)

    arr = sorted(values)
    pos = (len(arr) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(arr[lo])
    frac = pos - lo
    return float(arr[lo] * (1 - frac) + arr[hi] * frac)


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        }

    arr = sorted(values)
    count = len(arr)
    mean = sum(arr) / count
    median = _percentile(arr, 0.5)

    return {
        "count": count,
        "min": float(arr[0]),
        "max": float(arr[-1]),
        "mean": float(mean),
        "median": float(median),
        "p10": float(_percentile(arr, 0.10)),
        "p25": float(_percentile(arr, 0.25)),
        "p75": float(_percentile(arr, 0.75)),
        "p90": float(_percentile(arr, 0.90)),
    }


def _infer_category(obj: Dict[str, Any]) -> str:
    category = obj.get("category")
    if isinstance(category, str) and category.strip():
        return category.strip()

    tags = obj.get("tags")
    if isinstance(tags, list) and tags:
        first = tags[0]
        if isinstance(first, str) and first.strip():
            return first.strip()

    return "unknown"


def _infer_predicted_type(obj: Dict[str, Any]) -> str:
    pred = obj.get("predicted_type")
    if isinstance(pred, str) and pred.strip():
        return pred.strip()

    pred = obj.get("pred_type")
    if isinstance(pred, str) and pred.strip():
        return pred.strip()

    return "unknown"


def _load_eval_rows(paths: Iterable[Path]) -> List[EvalRow]:
    rows: List[EvalRow] = []

    for path in paths:
        if not path.exists():
            continue

        for obj in iter_jsonl(path):
            expected = obj.get("expected_type")
            top_score = obj.get("top_score")
            if expected is None or top_score is None:
                continue

            try:
                score = float(top_score)
            except (TypeError, ValueError):
                continue

            expected_str = str(expected).strip()
            if not expected_str:
                continue

            rows.append(
                EvalRow(
                    id=str(obj.get("id", "")),
                    query=str(obj.get("query", "")),
                    category=_infer_category(obj),
                    expected_type=expected_str,
                    predicted_type=_infer_predicted_type(obj),
                    top_score=score,
                    source_file=str(path),
                )
            )

    return rows


def _score_histogram(rows: List[EvalRow], width: float = 0.05) -> Dict[str, int]:
    buckets: Dict[str, int] = Counter()
    for row in rows:
        lo = math.floor(row.top_score / width) * width
        hi = lo + width
        label = f"[{lo:.2f},{hi:.2f})"
        buckets[label] += 1
    return dict(sorted(buckets.items(), key=lambda kv: kv[0]))


def _compute_thresholds(rows: List[EvalRow]) -> Dict[str, Any]:
    by_expected: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_expected[r.expected_type].append(r.top_score)

    answer = by_expected.get("answer", [])
    clarify = by_expected.get("clarify", [])
    refuse = by_expected.get("refuse", [])
    non_answer = clarify + refuse

    # High threshold: separate strong answerable cases from non-answer tail.
    answer_p25 = _percentile(answer, 0.25) if answer else 0.60
    non_answer_p90 = _percentile(non_answer, 0.90) if non_answer else 0.50
    threshold_high = (answer_p25 + non_answer_p90) / 2.0

    # Low threshold: separate clear refuse region from ambiguous clarify middle.
    refuse_p75 = _percentile(refuse, 0.75) if refuse else 0.25
    clarify_p25 = _percentile(clarify, 0.25) if clarify else 0.45
    threshold_low = (refuse_p75 + clarify_p25) / 2.0

    if threshold_low >= threshold_high:
        mid = (threshold_low + threshold_high) / 2.0
        threshold_low = mid - 0.05
        threshold_high = mid + 0.05

    return {
        "threshold_high": round(float(threshold_high), 4),
        "threshold_low": round(float(threshold_low), 4),
        "method": {
            "threshold_high": "midpoint(answer_p25, non_answer_p90)",
            "threshold_low": "midpoint(refuse_p75, clarify_p25)",
        },
        "anchors": {
            "answer_p25": round(float(answer_p25), 6),
            "non_answer_p90": round(float(non_answer_p90), 6),
            "clarify_p25": round(float(clarify_p25), 6),
            "refuse_p75": round(float(refuse_p75), 6),
        },
    }


def _simulate(rows: List[EvalRow], threshold_high: float, threshold_low: float) -> Dict[str, Any]:
    zone_counts: Counter[str] = Counter()
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    source_zone_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    source_total: Counter[str] = Counter()
    source_correct: Counter[str] = Counter()

    mismatches: List[Dict[str, Any]] = []

    for r in rows:
        if r.top_score >= threshold_high:
            pred = "answer"
            zone = "answer_zone"
        elif r.top_score < threshold_low:
            pred = "refuse"
            zone = "refuse_zone"
        else:
            pred = "clarify"
            zone = "clarify_zone"

        zone_counts[zone] += 1
        confusion[r.expected_type][pred] += 1
        source_zone_counts[r.source_file][zone] += 1
        source_total[r.source_file] += 1

        if pred == r.expected_type:
            source_correct[r.source_file] += 1

        if pred != r.expected_type:
            mismatches.append(
                {
                    "id": r.id,
                    "query": r.query,
                    "category": r.category,
                    "source_file": r.source_file,
                    "expected_type": r.expected_type,
                    "predicted_type_simulated": pred,
                    "zone": zone,
                    "top_score": float(r.top_score),
                    "existing_predicted_type": r.predicted_type,
                }
            )

    total = len(rows)
    correct = total - len(mismatches)

    label_metrics: Dict[str, Dict[str, float]] = {}
    for label in ("answer", "clarify", "refuse"):
        tp = confusion[label][label]
        col_total = sum(confusion[x][label] for x in confusion.keys())
        row_total = sum(confusion[label].values())
        precision = (tp / col_total) if col_total else 0.0
        recall = (tp / row_total) if row_total else 0.0
        label_metrics[label] = {
            "support": row_total,
            "precision": precision,
            "recall": recall,
            "accuracy_within_label": recall,
        }

    source_breakdown: Dict[str, Dict[str, Any]] = {}
    for source, total_n in source_total.items():
        source_breakdown[source] = {
            "count": total_n,
            "zone_counts": dict(source_zone_counts[source]),
            "overall_accuracy": (source_correct[source] / total_n) if total_n else 0.0,
        }

    return {
        "rule": {
            "if top_score >= threshold_high": "answer",
            "if top_score < threshold_low": "refuse",
            "otherwise": "clarify",
        },
        "zone_counts": dict(zone_counts),
        "confusion_summary": {k: dict(v) for k, v in confusion.items()},
        "label_metrics": label_metrics,
        "overall_accuracy": (correct / total) if total else 0.0,
        "source_file_breakdown": source_breakdown,
        "remaining_mismatches": mismatches,
    }


def _stats_by(rows: List[EvalRow], attr: str) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        key = getattr(row, attr)
        grouped[key].append(row.top_score)

    out: Dict[str, Dict[str, float]] = {}
    for key in sorted(grouped.keys()):
        out[key] = _stats(grouped[key])
    return out


def _format_markdown(summary: Dict[str, Any]) -> str:
    current = summary["current_thresholds"]
    proposed = summary["proposed_thresholds"]
    sim = summary["simulation"]

    lines: List[str] = []
    lines.append("# Threshold Calibration Summary")
    lines.append("")
    lines.append(f"- total_rows: {summary['total_rows']}")
    lines.append(f"- current.threshold_high: {current['threshold_high']}")
    lines.append(f"- current.threshold_low: {current['threshold_low']}")
    lines.append(f"- proposed.threshold_high: {proposed['threshold_high']}")
    lines.append(f"- proposed.threshold_low: {proposed['threshold_low']}")
    lines.append("")
    lines.append("## Rationale")
    lines.append("- threshold_high separates strong answerable tail from non-answer high tail.")
    lines.append("- threshold_low separates clear refuse region from ambiguous middle region.")
    lines.append("")
    lines.append("## Simulation")
    lines.append(f"- zone_counts: {sim['zone_counts']}")
    lines.append(f"- overall_accuracy: {sim['overall_accuracy']:.4f}")
    lines.append(f"- remaining_mismatches: {len(sim['remaining_mismatches'])}")
    lines.append("")
    lines.append("## Config Update Preview")
    lines.append("```yaml")
    lines.append("confidence:")
    lines.append(f"  threshold_high: {proposed['threshold_high']}")
    lines.append(f"  threshold_low: {proposed['threshold_low']}")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def build_threshold_calibration_summary(
    repo_root: Path,
    input_files: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    config, _ = load_app_config(repo_root)

    files = input_files or [
        repo_root / "eval" / "results.jsonl",
        repo_root / "eval" / "diagnostic_citation_signal_results.jsonl",
        repo_root / "eval_v2" / "synthetic_results.jsonl",
    ]

    rows = _load_eval_rows(files)

    proposed = _compute_thresholds(rows)
    sim = _simulate(rows, proposed["threshold_high"], proposed["threshold_low"])

    summary: Dict[str, Any] = {
        "total_rows": len(rows),
        "labels": ["answer", "clarify", "refuse"],
        "current_thresholds": {
            "threshold_high": float(config.confidence.threshold_high),
            "threshold_low": float(config.confidence.threshold_low),
        },
        "proposed_thresholds": proposed,
        "score_stats_by_expected_type": _stats_by(rows, "expected_type"),
        "score_stats_by_category": _stats_by(rows, "category"),
        "score_stats_by_source_file": _stats_by(rows, "source_file"),
        "score_histogram_buckets": _score_histogram(rows),
        "simulation": sim,
        "recommendation": {
            "config_patch_preview": {
                "confidence": {
                    "threshold_high": proposed["threshold_high"],
                    "threshold_low": proposed["threshold_low"],
                }
            },
            "apply_now": False,
            "notes": [
                "This report recommends threshold updates based on observed evaluation score bands.",
                "Serving/runtime thresholds are unchanged in this step.",
            ],
        },
    }

    return summary


def write_threshold_calibration_reports(
    repo_root: Path,
    summary: Dict[str, Any],
    *,
    json_path: Path,
    markdown_path: Optional[Path] = None,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_format_markdown(summary), encoding="utf-8")
