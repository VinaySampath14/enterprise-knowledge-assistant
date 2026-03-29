from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _parse_float_csv(s: str) -> List[float]:
    vals: List[float] = []
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    return vals


def _run_cmd(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _extract_line_value(stdout: str, key: str) -> Optional[str]:
    pat = re.compile(rf"^{re.escape(key)}\s*:\s*(.+)$")
    for line in stdout.splitlines():
        m = pat.match(line.strip())
        if m:
            return m.group(1).strip()
    return None


def _build_markdown_report(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Phase 1 Threshold Sweep Report")
    lines.append("")
    lines.append(f"- generated_utc: {summary.get('generated_utc')}")
    lines.append(f"- phase0_baseline_run: {summary.get('phase0_baseline_run')}")
    lines.append(f"- total_combos: {summary.get('total_combos')}")
    lines.append(f"- completed_combos: {summary.get('completed_combos')}")
    lines.append("")

    lines.append("## Ranked Results")
    lines.append("| Rank | threshold_low | threshold_high | guardrails | acc_delta | false_refusal_delta | false_answer_delta | candidate_run |")
    lines.append("|---:|---:|---:|---|---:|---:|---:|---|")

    for i, row in enumerate(summary.get("ranked_results", []), start=1):
        lines.append(
            "| "
            f"{i} | {row.get('threshold_low'):.2f} | {row.get('threshold_high'):.2f} | "
            f"{row.get('guardrails_passed')} | {row.get('overall_accuracy_delta', 0.0):+.4f} | "
            f"{row.get('false_refusal_delta', 0.0):+.4f} | {row.get('false_answer_delta', 0.0):+.4f} | "
            f"{row.get('candidate_run', '')} |"
        )

    lines.append("")
    rec = summary.get("recommended_candidate")
    if rec:
        lines.append("## Recommendation")
        lines.append(
            f"- Recommended: low={rec.get('threshold_low'):.2f}, high={rec.get('threshold_high'):.2f}, "
            f"candidate={rec.get('candidate_run')}"
        )
    else:
        lines.append("## Recommendation")
        lines.append("- No guardrail-passing candidate found.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 threshold sweep using Phase 0 gate workflow")
    parser.add_argument(
        "--lows",
        default="0.28,0.30,0.32,0.35",
        help="Comma-separated threshold_low values.",
    )
    parser.add_argument(
        "--highs",
        default="0.45,0.50,0.55",
        help="Comma-separated threshold_high values.",
    )
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
        help="Expected label field in dataset.",
    )
    parser.add_argument(
        "--phase0-dir",
        default="artifacts/experiments/phase0",
        help="Phase 0 run dir used by run_phase_gate.",
    )
    parser.add_argument(
        "--phase1-dir",
        default="artifacts/experiments/phase1",
        help="Phase 1 output root.",
    )
    parser.add_argument(
        "--promoted-baseline-file",
        default="artifacts/experiments/phase0/promoted_baseline.json",
        help="Promoted baseline pointer JSON path relative to repo root.",
    )
    parser.add_argument(
        "--python-exe",
        default="",
        help="Optional python executable path for child commands (defaults to current interpreter).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    py = args.python_exe.strip() or sys.executable

    lows = _parse_float_csv(args.lows)
    highs = _parse_float_csv(args.highs)
    combos: List[Tuple[float, float]] = [(lo, hi) for lo in lows for hi in highs if lo < hi]
    if not combos:
        raise ValueError("No valid threshold pairs found. Ensure low < high.")

    config_path = root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    promoted_path = root / args.promoted_baseline_file
    if not promoted_path.exists():
        raise FileNotFoundError(
            f"promoted baseline pointer not found: {promoted_path}. Run phase0 gate first."
        )

    promoted_obj = _load_json(promoted_path)
    phase0_baseline = str(promoted_obj.get("promoted_baseline_run_dir", ""))

    phase1_root = root / args.phase1_dir
    sweep_id = f"threshold_sweep_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    sweep_dir = phase1_root / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    original_text = config_path.read_text(encoding="utf-8")
    original_cfg = yaml.safe_load(original_text)

    run_records: List[Dict[str, Any]] = []

    try:
        for idx, (lo, hi) in enumerate(combos, start=1):
            cfg = yaml.safe_load(original_text)
            cfg.setdefault("confidence", {})
            cfg["confidence"]["threshold_low"] = float(lo)
            cfg["confidence"]["threshold_high"] = float(hi)
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            cmd = [
                py,
                str(root / "scripts" / "experiments" / "run_phase_gate.py"),
                "--dataset",
                args.dataset,
                "--category-field",
                args.category_field,
                "--expected-type-field",
                args.expected_type_field,
                "--phase-dir",
                args.phase0_dir,
            ]

            proc = _run_cmd(cmd, root)
            combo_tag = f"{idx:02d}_low_{lo:.2f}_high_{hi:.2f}".replace(".", "p")
            log_path = sweep_dir / f"run_{combo_tag}.log"
            log_path.write_text(
                "\n".join(
                    [
                        "# Command",
                        " ".join(cmd),
                        "",
                        "# STDOUT",
                        proc.stdout,
                        "",
                        "# STDERR",
                        proc.stderr,
                        "",
                        f"# exit_code={proc.returncode}",
                    ]
                ),
                encoding="utf-8",
            )

            rec: Dict[str, Any] = {
                "threshold_low": lo,
                "threshold_high": hi,
                "exit_code": proc.returncode,
                "log": str(log_path.relative_to(root)).replace("\\", "/"),
            }

            if proc.returncode != 0:
                rec["status"] = "failed"
                run_records.append(rec)
                continue

            baseline_run = _extract_line_value(proc.stdout, "baseline_run")
            candidate_run = _extract_line_value(proc.stdout, "candidate_run")
            comparison_json = _extract_line_value(proc.stdout, "comparison_json")
            recommendation = _extract_line_value(proc.stdout, "promotion_recommendation")

            rec["status"] = "ok"
            rec["baseline_run"] = baseline_run
            rec["candidate_run"] = candidate_run
            rec["comparison_json"] = comparison_json
            rec["promotion_recommendation"] = recommendation

            if comparison_json and comparison_json != "None":
                comp_path = Path(comparison_json)
                if comp_path.exists():
                    comp = _load_json(comp_path)
                    core = comp.get("core_deltas", {})
                    guard = comp.get("guardrails", {})
                    rec["overall_accuracy_delta"] = float(
                        core.get("overall_accuracy", {}).get("delta", 0.0)
                    )
                    rec["false_refusal_delta"] = float(
                        core.get("false_refusal_rate", {}).get("delta", 0.0)
                    )
                    rec["false_answer_delta"] = float(
                        core.get("false_answer_rate", {}).get("delta", 0.0)
                    )
                    rec["guardrails_passed"] = bool(guard.get("overall_passed", False))
                else:
                    rec["guardrails_passed"] = False
            else:
                # No comparison_json means no candidate was created (likely no-op).
                rec["guardrails_passed"] = False

            run_records.append(rec)
    finally:
        # Restore original config exactly.
        config_path.write_text(original_text, encoding="utf-8")

    successful = [r for r in run_records if r.get("status") == "ok" and r.get("comparison_json")]
    ranked = sorted(
        successful,
        key=lambda r: (
            bool(r.get("guardrails_passed", False)),
            float(r.get("overall_accuracy_delta", 0.0)),
            -float(r.get("false_refusal_delta", 0.0)),
            -float(r.get("false_answer_delta", 0.0)),
        ),
        reverse=True,
    )

    recommended = next((r for r in ranked if r.get("guardrails_passed")), None)

    summary: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase0_baseline_run": phase0_baseline,
        "phase1_sweep_dir": str(sweep_dir),
        "dataset": args.dataset,
        "category_field": args.category_field,
        "expected_type_field": args.expected_type_field,
        "combos": [{"threshold_low": lo, "threshold_high": hi} for lo, hi in combos],
        "total_combos": len(combos),
        "completed_combos": len(successful),
        "ranked_results": ranked,
        "recommended_candidate": recommended,
        "all_runs": run_records,
        "notes": [
            "Config is restored to original content after sweep completion.",
            "Each combo is executed through run_phase_gate.py for consistent artifact contracts.",
        ],
    }

    summary_json = sweep_dir / "sweep_summary.json"
    _save_json(summary_json, summary)

    report_md = sweep_dir / "sweep_report.md"
    report_md.write_text(_build_markdown_report(summary), encoding="utf-8")

    print(f"[OK] Sweep summary: {summary_json}")
    print(f"[OK] Sweep report: {report_md}")
    if recommended:
        print(
            "[OK] Recommended candidate: "
            f"low={recommended['threshold_low']:.2f}, high={recommended['threshold_high']:.2f}, "
            f"run={recommended.get('candidate_run')}"
        )
    else:
        print("[WARN] No guardrail-passing candidate found.")


if __name__ == "__main__":
    main()
