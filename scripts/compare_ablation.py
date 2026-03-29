from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_num(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _fmt_faith(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}"


def _false_rates_from_summary(summary: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    total = summary.get("total_queries")
    if not isinstance(total, int) or total <= 0:
        return None, None

    confusion = summary.get("confusion_summary", {}) or {}
    false_ans = confusion.get("expected_refuse_predicted_answer")
    false_ref = confusion.get("expected_answer_predicted_refuse")

    if not isinstance(false_ans, int) or not isinstance(false_ref, int):
        return None, None

    return false_ans / total, false_ref / total


def _collect_from_summary_files(repo_root: Path) -> Dict[str, Dict[str, Any]]:
    eval_dir = repo_root / "eval"
    eval_v2_dir = repo_root / "eval_v2"

    rows: Dict[str, Dict[str, Any]] = {}

    for manual_path in sorted(eval_dir.glob("*_manual_summary.json")):
        prefix = manual_path.name.removesuffix("_manual_summary.json")
        synthetic_path = eval_v2_dir / f"{prefix}_synthetic_summary.json"
        holdout_path = eval_dir / f"{prefix}_holdout_summary.json"

        if not synthetic_path.exists() or not holdout_path.exists():
            continue

        manual = _load_json(manual_path)
        synthetic = _load_json(synthetic_path)
        holdout = _load_json(holdout_path)

        false_ans, false_ref = _false_rates_from_summary(manual)

        rows[prefix] = {
            "version": prefix,
            "manual": float(manual.get("overall_accuracy", 0.0)),
            "synthetic": float(synthetic.get("overall_accuracy", 0.0)),
            "holdout": float(holdout.get("overall_accuracy", 0.0)),
            "false_ans": false_ans,
            "false_ref": false_ref,
            "faithfulness": None,
        }

    return rows


def _dataset_bucket(dataset_param: str) -> Optional[str]:
    name = Path(dataset_param).name.lower()
    if "manual" in name:
        return "manual"
    if "synthetic" in name:
        return "synthetic"
    if "holdout" in name:
        return "holdout"
    return None


def _version_sort_key(version: str) -> Tuple[int, int, int, str]:
    v = version.strip().lower()

    # Put h0/h1-style baselines first when present.
    m_h = re.search(r"(?:^|[_\-])h(\d+)(?:$|[_\-])", v)
    if m_h:
        return (0, int(m_h.group(1)), 0, v)

    # Then sort vN, vNa, vNb style progression naturally.
    m_v = re.match(r"^v(\d+)([a-z]?)", v)
    if m_v:
        major = int(m_v.group(1))
        suffix = m_v.group(2)
        suffix_ord = (ord(suffix) - ord("a") + 1) if suffix else 0
        return (1, major, suffix_ord, v)

    # Then stepN_hM style versions.
    m_step = re.search(r"(?:^|[_\-])step(\d+)(?:$|[_\-])", v)
    if m_step:
        return (2, int(m_step.group(1)), 0, v)

    # Fallback alphabetical.
    return (3, 9999, 0, v)


def _collect_from_mlflow(
    repo_root: Path,
    *,
    experiment_name: str,
    tracking_uri: Optional[str],
    version_tag: str,
) -> Dict[str, Dict[str, Any]]:
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        env_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
        if env_uri:
            mlflow.set_tracking_uri(env_uri)
        else:
            mlflow.set_tracking_uri((repo_root / "artifacts" / "mlflow").resolve().as_uri())

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return {}

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
    )

    out: Dict[str, Dict[str, Any]] = {}

    for _, run in runs.iterrows():
        tags = {k[5:]: run[k] for k in run.index if str(k).startswith("tags.")}
        params = {k[7:]: run[k] for k in run.index if str(k).startswith("params.")}
        metrics = {k[8:]: run[k] for k in run.index if str(k).startswith("metrics.")}

        version = str(tags.get(version_tag) or tags.get("version") or "").strip()
        if not version:
            continue

        dataset = str(params.get("dataset") or "").strip()
        bucket = _dataset_bucket(dataset)
        if bucket is None:
            continue

        rec = out.setdefault(
            version,
            {
                "version": version,
                "manual": None,
                "synthetic": None,
                "holdout": None,
                "false_ans": None,
                "false_ref": None,
                "faithfulness": None,
            },
        )

        rec[bucket] = float(metrics.get("overall_accuracy", 0.0))

        if bucket == "manual":
            total = metrics.get("total_queries")
            false_ans_count = metrics.get("expected_refuse_predicted_answer")
            false_ref_count = metrics.get("expected_answer_predicted_refuse")
            if isinstance(total, (int, float)) and total > 0:
                if isinstance(false_ans_count, (int, float)):
                    rec["false_ans"] = float(false_ans_count) / float(total)
                if isinstance(false_ref_count, (int, float)):
                    rec["false_ref"] = float(false_ref_count) / float(total)

        faith = tags.get("faithfulness") or params.get("faithfulness")
        if faith not in (None, ""):
            try:
                rec["faithfulness"] = float(faith)
            except ValueError:
                pass

    return out


def _autofill_phase_gate_from_mlflow(
    repo_root: Path,
    *,
    rows: Dict[str, Dict[str, Any]],
    experiment_name: str,
    tracking_uri: Optional[str],
    version_tag: str,
) -> None:
    if not rows:
        return

    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        env_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
        if env_uri:
            mlflow.set_tracking_uri(env_uri)
        else:
            mlflow.set_tracking_uri((repo_root / "artifacts" / "mlflow").resolve().as_uri())

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        return

    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
    )

    seen_versions: set[str] = set()

    for _, run in runs.iterrows():
        run_id = str(run.get("run_id") or run.get("Run ID") or "").strip()
        tags = {k[5:]: run[k] for k in run.index if str(k).startswith("tags.")}

        version = str(tags.get(version_tag) or tags.get("version") or "").strip()
        if not version or version not in rows or version in seen_versions:
            continue

        verdict = str(tags.get("promotion_recommendation") or "").strip().upper()

        row = rows[version]
        if verdict:
            row["gate_verdict"] = verdict
        if run_id:
            row["run_id"] = run_id

        seen_versions.add(version)


def _apply_faithfulness_file(rows: Dict[str, Dict[str, Any]], faith_path: Optional[Path]) -> None:
    if faith_path is None or not faith_path.exists():
        return

    data = _load_json(faith_path)
    if not isinstance(data, dict):
        return

    for version, value in data.items():
        if version in rows and isinstance(value, (int, float)):
            rows[version]["faithfulness"] = float(value)


def _apply_metadata_file(rows: Dict[str, Dict[str, Any]], metadata_path: Optional[Path]) -> None:
    if metadata_path is None or not metadata_path.exists():
        return

    data = _load_json(metadata_path)
    if not isinstance(data, dict):
        return

    for version, obj in data.items():
        if version not in rows or not isinstance(obj, dict):
            continue
        row = rows[version]
        row["what_changed"] = str(obj.get("what_changed", row.get("what_changed", ""))).strip()
        row["gate_verdict"] = str(obj.get("gate_verdict", row.get("gate_verdict", ""))).strip().upper()
        row["run_id"] = str(obj.get("run_id", row.get("run_id", ""))).strip()
        row["notes"] = str(obj.get("notes", row.get("notes", ""))).strip()
        if "faithfulness" in obj and isinstance(obj["faithfulness"], (int, float)):
            row["faithfulness"] = float(obj["faithfulness"])


def _safety_status(false_ans: Optional[float], false_ref: Optional[float], *, fa_max: float, fr_max: float) -> str:
    if false_ans is None or false_ref is None:
        return "unknown"
    if false_ans <= fa_max and false_ref <= fr_max:
        return "pass"
    if false_ans <= (fa_max * 1.5) and false_ref <= (fr_max * 1.5):
        return "warn"
    return "fail"


def _is_baseline_version(version: str) -> bool:
    v = version.strip().lower()
    return bool(
        re.search(r"(^|[_\-])(baseline|h0)([_\-]|$)", v)
        or v.startswith("step2_h0")
        or v.startswith("h0_")
    )


def _infer_change_label(version: str, what_changed: str) -> str:
    text = what_changed.strip()
    if text and text != "-":
        return text

    v = version.strip().lower()
    if re.search(r"intent|step3|h1", v):
        return "intent layer"
    if _is_baseline_version(v):
        return "baseline"
    return "change"


def _decorate_rows(rows: List[Dict[str, Any]], *, fa_max: float, fr_max: float) -> List[Dict[str, Any]]:
    prev: Optional[Dict[str, Any]] = None
    out: List[Dict[str, Any]] = []
    version_counter = 0

    for r in rows:
        row = dict(r)
        row.setdefault("what_changed", "-")
        row.setdefault("gate_verdict", "-")
        row.setdefault("run_id", "-")
        row.setdefault("notes", "-")

        raw_version = str(row.get("version", "")).strip()
        change_label = _infer_change_label(raw_version, str(row.get("what_changed", "-")))
        if _is_baseline_version(raw_version):
            row["version_label"] = "baseline"
        else:
            version_counter += 1
            row["version_label"] = f"v{version_counter} ({change_label})"

        if prev is None:
            row["delta_prev_manual"] = None
            row["delta_prev_synth"] = None
            row["delta_prev_holdout"] = None
        else:
            m_prev = prev.get("manual")
            s_prev = prev.get("synthetic")
            h_prev = prev.get("holdout")
            m_cur = row.get("manual")
            s_cur = row.get("synthetic")
            h_cur = row.get("holdout")
            row["delta_prev_manual"] = (m_cur - m_prev) if isinstance(m_cur, float) and isinstance(m_prev, float) else None
            row["delta_prev_synth"] = (s_cur - s_prev) if isinstance(s_cur, float) and isinstance(s_prev, float) else None
            row["delta_prev_holdout"] = (h_cur - h_prev) if isinstance(h_cur, float) and isinstance(h_prev, float) else None

        row["safety_status"] = _safety_status(row.get("false_ans"), row.get("false_ref"), fa_max=fa_max, fr_max=fr_max)
        out.append(row)
        prev = row

    return out


def _build_executive_summary(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return ["## Executive Summary", "- No rows found."]

    best_manual = max(rows, key=lambda r: float(r.get("manual") or -1.0))
    best_holdout = max(rows, key=lambda r: float(r.get("holdout") or -1.0))

    lines = [
        "## Executive Summary",
        f"- Best manual accuracy: {best_manual['version']} ({_fmt_num(best_manual.get('manual'))}).",
        f"- Best holdout accuracy: {best_holdout['version']} ({_fmt_num(best_holdout.get('holdout'))}).",
        "- Safety status uses false-answer and false-refusal thresholds configured in this script run.",
    ]
    return lines


def _build_legend() -> List[str]:
    return [
        "## Legend",
        "- Manual/Synth/Holdout: Overall accuracy on each evaluation slice.",
        "- FalseAns: expected_refuse_predicted_answer / total_queries.",
        "- FalseRef: expected_answer_predicted_refuse / total_queries.",
        "- DeltaPrev(M/S/H): Manual, Synthetic, and Holdout deltas versus previous version row.",
        "- Safety: pass/warn/fail based on configured false-answer and false-refusal limits.",
        "- Faithfulness: Grounding quality score (placeholder until computed/logged).",
        "- Decision and Run ID can be auto-filled from phase-gate MLflow runs by ablation version.",
        "- Rows with excluded prefixes (default: dryrun_, tmp_) are hidden unless explicitly included.",
    ]


def _filter_rows_by_prefix(
    rows_map: Dict[str, Dict[str, Any]],
    *,
    exclude_prefixes: List[str],
) -> Dict[str, Dict[str, Any]]:
    if not exclude_prefixes:
        return rows_map

    normalized = [p.strip().lower() for p in exclude_prefixes if p.strip()]
    if not normalized:
        return rows_map

    out: Dict[str, Dict[str, Any]] = {}
    for version, row in rows_map.items():
        v = version.lower()
        if any(v.startswith(prefix) for prefix in normalized):
            continue
        out[version] = row
    return out


def _render_table(rows: List[Dict[str, Any]]) -> str:
    header = (
        "Version | What Changed | Manual | Synth | Holdout | FalseAns | FalseRef "
        "| DeltaPrev(M/S/H) | Safety | Faith | Decision | Run ID | Notes"
    )
    sep = (
        "--------|--------------|--------|-------|---------|----------|----------"
        "|------------------|--------|-------|----------|--------|------"
    )
    lines = [header, sep]

    for r in rows:
        delta_m = _fmt_pct(r.get("delta_prev_manual"))
        delta_s = _fmt_pct(r.get("delta_prev_synth"))
        delta_h = _fmt_pct(r.get("delta_prev_holdout"))
        lines.append(
            f"{str(r.get('version_label', r['version']))}"
            f" | {str(r.get('what_changed', '-'))}"
            f" | {_fmt_num(r.get('manual'))}"
            f" | {_fmt_num(r.get('synthetic'))}"
            f" | {_fmt_num(r.get('holdout'))}"
            f" | {_fmt_num(r.get('false_ans'))}"
            f" | {_fmt_num(r.get('false_ref'))}"
            f" | {delta_m}/{delta_s}/{delta_h}"
            f" | {r.get('safety_status', '-') }"
            f" | {_fmt_faith(r.get('faithfulness'))}"
            f" | {str(r.get('gate_verdict', '-'))}"
            f" | {str(r.get('run_id', '-'))}"
            f" | {str(r.get('notes', '-'))}"
        )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ablation comparison table from summary files or MLflow.")
    parser.add_argument(
        "--source",
        choices=["files", "mlflow"],
        default="files",
        help="Data source for the table.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        default="eka-eval",
        help="MLflow experiment name when --source mlflow.",
    )
    parser.add_argument(
        "--phase-gate-mlflow-experiment",
        default="eka-phase-gate",
        help="MLflow experiment name used to auto-fill GO/NO-GO and Run ID.",
    )
    parser.add_argument(
        "--tracking-uri",
        default="",
        help="Optional MLflow tracking URI override.",
    )
    parser.add_argument(
        "--version-tag",
        default="ablation_version",
        help="MLflow tag key holding the ablation version label.",
    )
    parser.add_argument(
        "--faithfulness-file",
        default="",
        help="Optional JSON file mapping version -> faithfulness score.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/baselines/ablation_comparison_table.md",
        help="Output markdown path relative to repo root.",
    )
    parser.add_argument(
        "--metadata-file",
        default="",
        help=(
            "Optional JSON mapping version -> metadata fields: "
            "what_changed, gate_verdict, run_id, notes, faithfulness"
        ),
    )
    parser.add_argument(
        "--false-ans-max",
        type=float,
        default=0.02,
        help="Safety pass threshold for FalseAns.",
    )
    parser.add_argument(
        "--false-ref-max",
        type=float,
        default=0.05,
        help="Safety pass threshold for FalseRef.",
    )
    parser.add_argument(
        "--disable-phase-gate-autofill",
        action="store_true",
        help="Disable auto-fill of GO/NO-GO and Run ID from phase-gate MLflow runs.",
    )
    parser.add_argument(
        "--exclude-prefixes",
        default="dryrun_,tmp_",
        help=(
            "Comma-separated version prefixes to exclude from table output "
            "(default: dryrun_,tmp_)."
        ),
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if args.source == "files":
        rows_map = _collect_from_summary_files(repo_root)
    else:
        rows_map = _collect_from_mlflow(
            repo_root,
            experiment_name=args.mlflow_experiment,
            tracking_uri=args.tracking_uri or None,
            version_tag=args.version_tag,
        )

    faith_path = Path(args.faithfulness_file) if args.faithfulness_file else None
    if faith_path is not None and not faith_path.is_absolute():
        faith_path = repo_root / faith_path

    _apply_faithfulness_file(rows_map, faith_path)

    metadata_path = Path(args.metadata_file) if args.metadata_file else None
    if metadata_path is not None and not metadata_path.is_absolute():
        metadata_path = repo_root / metadata_path
    _apply_metadata_file(rows_map, metadata_path)

    if not args.disable_phase_gate_autofill:
        _autofill_phase_gate_from_mlflow(
            repo_root,
            rows=rows_map,
            experiment_name=args.phase_gate_mlflow_experiment,
            tracking_uri=args.tracking_uri or None,
            version_tag=args.version_tag,
        )

    exclude_prefixes = [p for p in args.exclude_prefixes.split(",") if p.strip()]
    rows_map = _filter_rows_by_prefix(rows_map, exclude_prefixes=exclude_prefixes)

    rows = sorted(rows_map.values(), key=lambda x: _version_sort_key(str(x["version"])))
    rows = _decorate_rows(rows, fa_max=args.false_ans_max, fr_max=args.false_ref_max)

    summary_lines = _build_executive_summary(rows)
    legend_lines = _build_legend()
    table = _render_table(rows)

    doc = "\n".join(summary_lines + [""] + legend_lines + ["", "## Comparison Table", "", table])

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc + "\n", encoding="utf-8")

    print(doc)
    print(f"\n[OK] Wrote table: {out_path}")


if __name__ == "__main__":
    main()
