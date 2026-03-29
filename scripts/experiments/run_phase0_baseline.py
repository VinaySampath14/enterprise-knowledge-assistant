from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# Ensure repo root is on sys.path when this file is executed directly.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.utils.jsonl import iter_jsonl


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _run_eval_once(
    root: Path,
    dataset_rel: str,
    results_out: Path,
    summary_out: Path,
    category_field: str,
    expected_type_field: str,
) -> Tuple[int, str, str, List[str]]:
    script = root / "scripts" / "experiments" / "run_eval_v2_synthetic.py"
    cmd = [
        sys.executable,
        str(script),
        "--dataset",
        dataset_rel,
        "--results",
        str(results_out.relative_to(root)).replace("\\", "/"),
        "--summary",
        str(summary_out.relative_to(root)).replace("\\", "/"),
        "--category-field",
        category_field,
        "--expected-type-field",
        expected_type_field,
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr, cmd


def _index_by_id(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rid = str(row.get("id", "")).strip()
        if rid:
            out[rid] = row
    return out


def _compare_predictions(
    run_a: List[Dict[str, Any]],
    run_b: List[Dict[str, Any]],
    top_score_eps: float = 1e-9,
) -> Dict[str, Any]:
    by_a = _index_by_id(run_a)
    by_b = _index_by_id(run_b)

    ids = sorted(set(by_a.keys()) | set(by_b.keys()))
    comparable_count = 0
    matched_count = 0
    diffs: List[Dict[str, Any]] = []
    citation_drift: List[Dict[str, Any]] = []

    # Phase 0 gate should be driven by decision stability, not citation extraction variance.
    # Citation-count drift is tracked separately as a non-blocking diagnostic.
    keys = ["category", "expected_type", "predicted_type"]
    for rid in ids:
        a = by_a.get(rid)
        b = by_b.get(rid)
        if a is None or b is None:
            diffs.append({"id": rid, "reason": "missing_in_one_run"})
            continue

        comparable_count += 1
        row_match = True
        row_diff: Dict[str, Any] = {"id": rid, "fields": {}}

        for k in keys:
            if a.get(k) != b.get(k):
                row_match = False
                row_diff["fields"][k] = {"run_a": a.get(k), "run_b": b.get(k)}

        a_score = float(a.get("top_score", 0.0))
        b_score = float(b.get("top_score", 0.0))
        if abs(a_score - b_score) > top_score_eps:
            row_match = False
            row_diff["fields"]["top_score"] = {"run_a": a_score, "run_b": b_score}

        a_cit = int(a.get("citations_count", 0))
        b_cit = int(b.get("citations_count", 0))
        if a_cit != b_cit:
            citation_drift.append(
                {
                    "id": rid,
                    "run_a": a_cit,
                    "run_b": b_cit,
                    "delta": b_cit - a_cit,
                }
            )

        if row_match:
            matched_count += 1
        else:
            diffs.append(row_diff)

    match_rate = (matched_count / comparable_count) if comparable_count else 0.0
    return {
        "comparable_count": comparable_count,
        "matched_count": matched_count,
        "match_rate": match_rate,
        "diff_count": len(diffs),
        "diff_samples": diffs[:20],
        "citation_count_drift_count": len(citation_drift),
        "citation_count_drift_samples": citation_drift[:20],
        "notes": [
            "Match rate is computed from decision-stability fields: category, expected_type, predicted_type, and top_score tolerance.",
            "Citation-count drift is tracked separately and does not affect reproducibility gate.",
        ],
    }


def _build_gate_decision_breakdown(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(predictions)
    pred_counts: Dict[str, int] = defaultdict(int)

    expected_answer_total = 0
    expected_refuse_total = 0
    expected_answer_pred_refuse = 0
    expected_refuse_pred_answer = 0
    clarify_total = 0

    for row in predictions:
        expected = str(row.get("expected_type", "unknown"))
        predicted = str(row.get("predicted_type", "unknown"))
        pred_counts[predicted] += 1

        if predicted == "clarify":
            clarify_total += 1
        if expected == "answer":
            expected_answer_total += 1
            if predicted == "refuse":
                expected_answer_pred_refuse += 1
        if expected == "refuse":
            expected_refuse_total += 1
            if predicted == "answer":
                expected_refuse_pred_answer += 1

    return {
        "total_queries": total,
        "predicted_type_counts": dict(pred_counts),
        "clarify_rate": (clarify_total / total) if total else 0.0,
        "false_refusal_rate": (
            expected_answer_pred_refuse / expected_answer_total if expected_answer_total else 0.0
        ),
        "false_answer_rate": (
            expected_refuse_pred_answer / expected_refuse_total if expected_refuse_total else 0.0
        ),
        "raw": {
            "expected_answer_total": expected_answer_total,
            "expected_refuse_total": expected_refuse_total,
            "expected_answer_pred_refuse": expected_answer_pred_refuse,
            "expected_refuse_pred_answer": expected_refuse_pred_answer,
            "clarify_total": clarify_total,
        },
    }


def _build_failure_taxonomy(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    mismatches = [p for p in predictions if p.get("expected_type") != p.get("predicted_type")]

    by_transition: Dict[str, int] = defaultdict(int)
    by_category: Dict[str, int] = defaultdict(int)
    by_category_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in sorted(mismatches, key=lambda r: float(r.get("top_score", 0.0)), reverse=True):
        expected = str(row.get("expected_type", "unknown"))
        predicted = str(row.get("predicted_type", "unknown"))
        category = str(row.get("category", "unknown"))
        trans = f"{expected}_to_{predicted}"

        by_transition[trans] += 1
        by_category[category] += 1

        if len(by_category_examples[category]) < 5:
            by_category_examples[category].append(
                {
                    "id": row.get("id"),
                    "query": row.get("query"),
                    "expected_type": expected,
                    "predicted_type": predicted,
                    "top_score": float(row.get("top_score", 0.0)),
                    "answer_preview": row.get("answer_preview", ""),
                }
            )

    return {
        "total_mismatches": len(mismatches),
        "mismatch_transitions": dict(sorted(by_transition.items())),
        "mismatches_by_category": dict(sorted(by_category.items())),
        "representative_examples_by_category": dict(by_category_examples),
    }


def _build_report_md(
    run_id: str,
    dataset_rel: str,
    dataset_hash: str,
    summary: Dict[str, Any],
    gate: Dict[str, Any],
    repro: Dict[str, Any],
    go_no_go: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Phase 0 Baseline Report")
    lines.append("")
    lines.append(f"- run_id: {run_id}")
    lines.append(f"- dataset: {dataset_rel}")
    lines.append(f"- dataset_sha256: {dataset_hash}")
    lines.append(f"- total_queries: {summary.get('total_queries', 0)}")
    lines.append(f"- overall_accuracy: {float(summary.get('overall_accuracy', 0.0)):.4f}")
    lines.append("")

    lines.append("## Core Decision Metrics")
    lines.append(f"- clarify_rate: {float(gate.get('clarify_rate', 0.0)):.4f}")
    lines.append(f"- false_refusal_rate: {float(gate.get('false_refusal_rate', 0.0)):.4f}")
    lines.append(f"- false_answer_rate: {float(gate.get('false_answer_rate', 0.0)):.4f}")
    lines.append(f"- predicted_type_counts: {gate.get('predicted_type_counts', {})}")
    lines.append("")

    lines.append("## Reproducibility")
    lines.append(f"- comparable_rows: {repro.get('comparable_count', 0)}")
    lines.append(f"- matched_rows: {repro.get('matched_count', 0)}")
    lines.append(f"- match_rate: {float(repro.get('match_rate', 0.0)):.4f}")
    lines.append(f"- diff_count: {repro.get('diff_count', 0)}")
    lines.append("")

    lines.append("## Per-Category Accuracy")
    category_summary = summary.get("category_summary", {})
    for cat in sorted(category_summary.keys()):
        acc = float(category_summary[cat].get("accuracy", 0.0))
        cnt = int(category_summary[cat].get("count", 0))
        lines.append(f"- {cat}: accuracy={acc:.4f}, count={cnt}")
    lines.append("")

    lines.append("## Phase 0 Gate")
    lines.append(f"- required_artifacts_present: {go_no_go.get('required_artifacts_present')}")
    lines.append(f"- reproducibility_passed: {go_no_go.get('reproducibility_passed')}")
    lines.append(f"- final_decision: {go_no_go.get('decision')}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 baseline automation and gate check")
    parser.add_argument(
        "--dataset",
        default="eval_v2/synthetic_scaffold_dataset_refined.jsonl",
        help="Dataset path relative to repo root.",
    )
    parser.add_argument(
        "--category-field",
        default="refined_category",
        help="Category field in dataset.",
    )
    parser.add_argument(
        "--expected-type-field",
        default="expected_type_refined",
        help="Expected type field in dataset.",
    )
    parser.add_argument(
        "--phase-dir",
        default="artifacts/experiments/phase0",
        help="Phase output directory relative to repo root.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run id; auto-generated if omitted.",
    )
    parser.add_argument(
        "--repro-threshold",
        type=float,
        default=0.95,
        help="Minimum required reproducibility match rate.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    dataset_path = root / args.dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    run_id = args.run_id.strip() or (
        f"V1_phase0_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_baseline"
    )
    run_dir = root / args.phase_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Capture config snapshot.
    config_path = root / "config.yaml"
    config_snapshot_path = run_dir / "config_snapshot.yaml"
    shutil.copyfile(config_path, config_snapshot_path)

    # Run eval twice for reproducibility checks.
    results_a = run_dir / "predictions_run_a.jsonl"
    summary_a = run_dir / "summary_run_a.json"
    code_a, stdout_a, stderr_a, cmd_a = _run_eval_once(
        root,
        args.dataset,
        results_a,
        summary_a,
        args.category_field,
        args.expected_type_field,
    )
    if code_a != 0:
        raise RuntimeError(f"Phase0 run A failed:\nSTDOUT:\n{stdout_a}\nSTDERR:\n{stderr_a}")

    results_b = run_dir / "predictions_run_b.jsonl"
    summary_b = run_dir / "summary_run_b.json"
    code_b, stdout_b, stderr_b, cmd_b = _run_eval_once(
        root,
        args.dataset,
        results_b,
        summary_b,
        args.category_field,
        args.expected_type_field,
    )
    if code_b != 0:
        raise RuntimeError(f"Phase0 run B failed:\nSTDOUT:\n{stdout_b}\nSTDERR:\n{stderr_b}")

    rows_a = list(iter_jsonl(results_a))
    rows_b = list(iter_jsonl(results_b))
    summary_obj = _load_json(summary_a)

    repro = _compare_predictions(rows_a, rows_b)
    _save_json(run_dir / "reproducibility.json", repro)

    # Canonical outputs from run A.
    shutil.copyfile(results_a, run_dir / "predictions.jsonl")
    shutil.copyfile(summary_a, run_dir / "summary.json")

    per_category_metrics = summary_obj.get("category_summary", {})
    _save_json(run_dir / "per_category_metrics.json", per_category_metrics)

    failure_taxonomy = _build_failure_taxonomy(rows_a)
    _save_json(run_dir / "failure_taxonomy.json", failure_taxonomy)

    gate_breakdown = _build_gate_decision_breakdown(rows_a)
    _save_json(run_dir / "gate_decision_breakdown.json", gate_breakdown)

    required = [
        "config_snapshot.yaml",
        "predictions.jsonl",
        "summary.json",
        "per_category_metrics.json",
        "failure_taxonomy.json",
        "gate_decision_breakdown.json",
        "reproducibility.json",
    ]
    required_artifacts_present = all((run_dir / r).exists() for r in required)
    reproducibility_passed = float(repro.get("match_rate", 0.0)) >= float(args.repro_threshold)

    go_no_go = {
        "required_artifacts_present": required_artifacts_present,
        "reproducibility_passed": reproducibility_passed,
        "reproducibility_threshold": args.repro_threshold,
        "decision": "GO" if (required_artifacts_present and reproducibility_passed) else "NO-GO",
    }
    _save_json(run_dir / "go_no_go.json", go_no_go)

    dataset_hash = _sha256_file(dataset_path)
    config_hash = _sha256_file(config_snapshot_path)
    cfg = yaml.safe_load(config_snapshot_path.read_text(encoding="utf-8"))

    report_md = _build_report_md(
        run_id,
        args.dataset,
        dataset_hash,
        summary_obj,
        gate_breakdown,
        repro,
        go_no_go,
    )
    (run_dir / "report.md").write_text(report_md, encoding="utf-8")

    metadata = {
        "run_id": run_id,
        "phase": "phase0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": args.dataset,
            "sha256": dataset_hash,
            "category_field": args.category_field,
            "expected_type_field": args.expected_type_field,
        },
        "config": {
            "path": "config.yaml",
            "sha256": config_hash,
            "retrieval_top_k": cfg.get("retrieval", {}).get("top_k"),
            "threshold_high": cfg.get("confidence", {}).get("threshold_high"),
            "threshold_low": cfg.get("confidence", {}).get("threshold_low"),
            "generation_enabled": cfg.get("generation", {}).get("enabled"),
            "generation_model": cfg.get("generation", {}).get("model"),
        },
        "commands": {
            "phase0_invocation": " ".join([sys.executable] + sys.argv),
            "eval_run_a": " ".join(cmd_a),
            "eval_run_b": " ".join(cmd_b),
        },
        "gate": go_no_go,
    }
    _save_json(run_dir / "metadata.json", metadata)

    command_lines = [
        "# Phase0 baseline invocation",
        " ".join([sys.executable] + sys.argv),
        "",
        "# Run A",
        " ".join(cmd_a),
        "",
        "# Run B",
        " ".join(cmd_b),
    ]
    (run_dir / "command.txt").write_text("\n".join(command_lines) + "\n", encoding="utf-8")

    print(f"[OK] Phase 0 bundle written: {run_dir}")
    print(f"[OK] GO/NO-GO decision: {go_no_go['decision']}")


if __name__ == "__main__":
    main()
