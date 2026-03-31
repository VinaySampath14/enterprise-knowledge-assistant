from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class AnswerRow:
    dataset: str
    id: str
    expected_type: str
    citations_count: int
    faithfulness: float
    answer_relevancy: float

    @property
    def is_bad_answer(self) -> bool:
        return self.expected_type != "answer"

    @property
    def is_uncited(self) -> bool:
        return self.citations_count <= 0

    @property
    def is_bad_or_uncited(self) -> bool:
        return self.is_bad_answer or self.is_uncited


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _dataset_name_from_path(path: Path) -> str:
    name = path.name.lower()
    if "manual" in name:
        return "manual"
    if "synthetic" in name:
        return "synthetic"
    if "holdout" in name:
        return "holdout"
    return path.stem


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _load_answer_rows(paths: List[Path]) -> List[AnswerRow]:
    rows: List[AnswerRow] = []
    for path in paths:
        if not path.exists():
            continue
        dataset = _dataset_name_from_path(path)
        for obj in _iter_jsonl(path):
            if str(obj.get("predicted_type", "")) != "answer":
                continue
            rows.append(
                AnswerRow(
                    dataset=dataset,
                    id=str(obj.get("id", "")),
                    expected_type=str(obj.get("expected_type", "")),
                    citations_count=int(obj.get("citations_count", 0) or 0),
                    faithfulness=_to_float(obj.get("faithfulness"), 0.0),
                    answer_relevancy=_to_float(obj.get("answer_relevancy"), 0.0),
                )
            )
    return rows


def _metrics_for_threshold(values: List[float], labels_bad: List[bool], threshold: float) -> Dict[str, float]:
    # Alert when signal is low.
    pred_alert = [v <= threshold for v in values]

    tp = sum(1 for pa, bad in zip(pred_alert, labels_bad) if pa and bad)
    fp = sum(1 for pa, bad in zip(pred_alert, labels_bad) if pa and not bad)
    fn = sum(1 for pa, bad in zip(pred_alert, labels_bad) if (not pa) and bad)
    tn = sum(1 for pa, bad in zip(pred_alert, labels_bad) if (not pa) and (not bad))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    alert_rate = (tp + fp) / len(values) if values else 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": alert_rate,
    }


def _find_best_threshold(values: List[float], labels_bad: List[bool]) -> Dict[str, Any]:
    if not values:
        return {
            "threshold": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "alert_rate": 0.0,
            "candidates_checked": 0,
        }

    candidates = sorted(set(values))
    best = None
    for t in candidates:
        m = _metrics_for_threshold(values, labels_bad, t)
        # Primary objective: F1. Secondary: higher precision, then lower alert rate.
        score = (m["f1"], m["precision"], -m["alert_rate"])
        if best is None or score > best[0]:
            best = (score, m)

    result = dict(best[1]) if best is not None else _metrics_for_threshold(values, labels_bad, 0.0)
    result["candidates_checked"] = len(candidates)
    return result


def _shadow_breakdown(rows: List[AnswerRow], faith_th: float, rel_th: float) -> Dict[str, Any]:
    by_dataset: Dict[str, Dict[str, Any]] = {}

    datasets = sorted({r.dataset for r in rows})
    for ds in datasets:
        ds_rows = [r for r in rows if r.dataset == ds]
        alerts = [r for r in ds_rows if (r.faithfulness <= faith_th or r.answer_relevancy <= rel_th)]

        if ds_rows:
            bad_or_uncited_total = sum(1 for r in ds_rows if r.is_bad_or_uncited)
            alert_bad_or_uncited = sum(1 for r in alerts if r.is_bad_or_uncited)
            precision_bad_or_uncited = alert_bad_or_uncited / len(alerts) if alerts else 0.0
            recall_bad_or_uncited = (
                alert_bad_or_uncited / bad_or_uncited_total if bad_or_uncited_total else 0.0
            )
        else:
            precision_bad_or_uncited = 0.0
            recall_bad_or_uncited = 0.0

        by_dataset[ds] = {
            "total_answer_predictions": len(ds_rows),
            "alert_count": len(alerts),
            "alert_rate": (len(alerts) / len(ds_rows)) if ds_rows else 0.0,
            "precision_bad_or_uncited": precision_bad_or_uncited,
            "recall_bad_or_uncited": recall_bad_or_uncited,
        }

    return by_dataset


def _avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _build_summary(version: str, inputs: List[Path], rows: List[AnswerRow]) -> Dict[str, Any]:
    faith_values = [r.faithfulness for r in rows]
    rel_values = [r.answer_relevancy for r in rows]

    labels_bad_answer = [r.is_bad_answer for r in rows]
    labels_bad_or_uncited = [r.is_bad_or_uncited for r in rows]

    best_faith = _find_best_threshold(faith_values, labels_bad_or_uncited)
    best_rel = _find_best_threshold(rel_values, labels_bad_or_uncited)

    shadow = _shadow_breakdown(rows, best_faith["threshold"], best_rel["threshold"])

    return {
        "ablation_version": version,
        "input_files": [str(p) for p in inputs],
        "total_answer_predictions": len(rows),
        "label_breakdown": {
            "bad_answer_count": sum(1 for r in rows if r.is_bad_answer),
            "uncited_answer_count": sum(1 for r in rows if r.is_uncited),
            "bad_or_uncited_count": sum(1 for r in rows if r.is_bad_or_uncited),
        },
        "signal_stats": {
            "faithfulness": {
                "min": min(faith_values) if faith_values else 0.0,
                "max": max(faith_values) if faith_values else 0.0,
                "mean": _avg(faith_values),
            },
            "answer_relevancy": {
                "min": min(rel_values) if rel_values else 0.0,
                "max": max(rel_values) if rel_values else 0.0,
                "mean": _avg(rel_values),
            },
        },
        "recommended_thresholds": {
            "faithfulness_low_alert_threshold": best_faith["threshold"],
            "answer_relevancy_low_alert_threshold": best_rel["threshold"],
            "fit_target": "bad_or_uncited_answer",
            "faithfulness_fit_metrics": best_faith,
            "answer_relevancy_fit_metrics": best_rel,
            "combined_rule": "alert_if faithfulness <= faithfulness_low_alert_threshold OR answer_relevancy <= answer_relevancy_low_alert_threshold",
        },
        "shadow_report": {
            "note": "Reporting-only shadow pass. No serving, gating, or promotion logic changed.",
            "dataset_breakdown": shadow,
        },
    }


def _format_markdown(summary: Dict[str, Any]) -> str:
    rec = summary["recommended_thresholds"]
    lines = [
        "# Hardening Signal Calibration",
        "",
        f"- ablation_version: {summary['ablation_version']}",
        f"- total_answer_predictions: {summary['total_answer_predictions']}",
        f"- bad_answer_count: {summary['label_breakdown']['bad_answer_count']}",
        f"- uncited_answer_count: {summary['label_breakdown']['uncited_answer_count']}",
        f"- bad_or_uncited_count: {summary['label_breakdown']['bad_or_uncited_count']}",
        "",
        "## Recommended Thresholds",
        f"- faithfulness_low_alert_threshold: {rec['faithfulness_low_alert_threshold']:.6f}",
        f"- answer_relevancy_low_alert_threshold: {rec['answer_relevancy_low_alert_threshold']:.6f}",
        f"- combined_rule: {rec['combined_rule']}",
        "",
        "## Fit Metrics",
        f"- faithfulness: precision={rec['faithfulness_fit_metrics']['precision']:.4f}, recall={rec['faithfulness_fit_metrics']['recall']:.4f}, f1={rec['faithfulness_fit_metrics']['f1']:.4f}, alert_rate={rec['faithfulness_fit_metrics']['alert_rate']:.4f}",
        f"- answer_relevancy: precision={rec['answer_relevancy_fit_metrics']['precision']:.4f}, recall={rec['answer_relevancy_fit_metrics']['recall']:.4f}, f1={rec['answer_relevancy_fit_metrics']['f1']:.4f}, alert_rate={rec['answer_relevancy_fit_metrics']['alert_rate']:.4f}",
        "",
        "## Shadow Breakdown",
    ]

    for ds, obj in summary["shadow_report"]["dataset_breakdown"].items():
        lines.append(
            f"- {ds}: answers={obj['total_answer_predictions']}, alerts={obj['alert_count']} ({obj['alert_rate']:.4f}), precision_bad_or_uncited={obj['precision_bad_or_uncited']:.4f}, recall_bad_or_uncited={obj['recall_bad_or_uncited']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "- Reporting-only output for Step 12 hardening.",
            "- Promotion guardrails are intentionally unchanged.",
        ]
    )

    return "\n".join(lines)


def _default_inputs(repo_root: Path, version: str) -> List[Path]:
    return [
        repo_root / "eval" / f"{version}_manual_results.jsonl",
        repo_root / "eval_v2" / f"{version}_synthetic_results.jsonl",
        repo_root / "eval" / f"{version}_holdout_results.jsonl",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate heuristic hardening signals (faithfulness + answer relevancy) from eval outputs."
    )
    parser.add_argument(
        "--ablation-version",
        default="step11_eval_hardening_metrics",
        help="Ablation version prefix to load results files.",
    )
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="Optional explicit result JSONL files. If omitted, uses --ablation-version defaults.",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Optional explicit output JSON path. Defaults to artifacts/calibration/hardening_signal_calibration_<version>.json",
    )
    parser.add_argument(
        "--out-md",
        default="",
        help="Optional explicit output Markdown path. Defaults to artifacts/calibration/hardening_signal_calibration_<version>.md",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    input_paths = [repo_root / p for p in args.inputs] if args.inputs else _default_inputs(repo_root, args.ablation_version)

    rows = _load_answer_rows(input_paths)
    summary = _build_summary(args.ablation_version, input_paths, rows)

    out_json = (
        repo_root / args.out_json
        if args.out_json
        else repo_root / "artifacts" / "calibration" / f"hardening_signal_calibration_{args.ablation_version}.json"
    )
    out_md = (
        repo_root / args.out_md
        if args.out_md
        else repo_root / "artifacts" / "calibration" / f"hardening_signal_calibration_{args.ablation_version}.md"
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    out_md.write_text(_format_markdown(summary), encoding="utf-8")

    print(f"[OK] Wrote calibration JSON: {out_json}")
    print(f"[OK] Wrote calibration Markdown: {out_md}")
    print("[INFO] Recommended reporting-only thresholds:")
    print(json.dumps(summary["recommended_thresholds"], indent=2))


if __name__ == "__main__":
    main()
