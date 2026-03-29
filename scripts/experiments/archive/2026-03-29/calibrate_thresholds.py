from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

# Ensure repo root is on sys.path when this file is executed directly.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.eval_runner.calibration import (
    build_threshold_calibration_summary,
    write_threshold_calibration_reports,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate confidence thresholds from eval results.")
    parser.add_argument(
        "--inputs",
        nargs="*",
        default=[
            "eval/results.jsonl",
            "eval/diagnostic_citation_signal_results.jsonl",
            "eval_v2/synthetic_results.jsonl",
        ],
        help="Input result JSONL files (relative to repo root).",
    )
    parser.add_argument(
        "--out-json",
        default="eval_v2/threshold_calibration_summary.json",
        help="Output summary JSON path (relative to repo root).",
    )
    parser.add_argument(
        "--out-md",
        default="eval_v2/threshold_calibration_summary.md",
        help="Output summary Markdown path (relative to repo root).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    input_files = [repo_root / p for p in args.inputs]

    summary = build_threshold_calibration_summary(repo_root, input_files=input_files)

    out_json = repo_root / args.out_json
    out_md = repo_root / args.out_md

    write_threshold_calibration_reports(
        repo_root,
        summary,
        json_path=out_json,
        markdown_path=out_md,
    )

    print(f"[OK] Wrote calibration summary JSON: {out_json}")
    print(f"[OK] Wrote calibration summary Markdown: {out_md}")
    print("[INFO] Proposed confidence threshold update (not applied):")
    print(json.dumps(summary["recommendation"]["config_patch_preview"], indent=2))


if __name__ == "__main__":
    main()
