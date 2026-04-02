from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List
import re


def _run(cmd: List[str], *, cwd: Path, label: str) -> None:
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"step failed: {label} (exit_code={proc.returncode})")


def _strip_run_suffix(version: str) -> str:
    return re.sub(r"(?:[_\-])r\d+$", "", version)


def _compute_effective_ablation_version(root: Path, requested: str, auto_suffix: bool) -> str:
    if not auto_suffix:
        return requested

    base = _strip_run_suffix(requested)

    def _exists_for(v: str) -> bool:
        return any(
            (root / p).exists()
            for p in [
                f"eval/{v}_manual_summary.json",
                f"eval/{v}_holdout_summary.json",
                f"eval_v2/{v}_synthetic_summary.json",
                f"eval/{v}_manual_results.jsonl",
                f"eval/{v}_holdout_results.jsonl",
                f"eval_v2/{v}_synthetic_results.jsonl",
            ]
        )

    if not _exists_for(base):
        return base

    i = 1
    while True:
        candidate = f"{base}_r{i}"
        if not _exists_for(candidate):
            return candidate
        i += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run eval + phase-gate across manual/synthetic/holdout and regenerate "
            "ablation comparison table in one command."
        )
    )
    parser.add_argument(
        "--ablation-version",
        required=True,
        help="Version label used for MLflow and table lineage (e.g., v4b_bm25_alpha06).",
    )
    parser.add_argument(
        "--auto-run-suffix",
        action="store_true",
        help=(
            "Auto-append _rN to avoid overwriting an existing ablation version's files. "
            "Example: step10_reranker_cross_encoder -> step10_reranker_cross_encoder_r1"
        ),
    )
    parser.add_argument(
        "--phase-dir",
        default="artifacts/experiments/phase0",
        help="Phase directory used by run_phase_gate.py.",
    )
    parser.add_argument(
        "--metadata-file",
        default="artifacts/baselines/ablation_metadata.json",
        help="Optional metadata file for comparison table enrichment.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip eval runs.",
    )
    parser.add_argument(
        "--skip-phase-gate",
        action="store_true",
        help="Skip phase-gate runs.",
    )
    parser.add_argument(
        "--skip-table",
        action="store_true",
        help="Skip compare_ablation table generation.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    py = sys.executable
    effective_version = _compute_effective_ablation_version(
        root,
        requested=args.ablation_version,
        auto_suffix=args.auto_run_suffix,
    )
    if effective_version != args.ablation_version:
        print(
            f"[INFO] Existing outputs found for '{args.ablation_version}'. "
            f"Using '{effective_version}' to preserve prior runs."
        )
    else:
        print(f"[INFO] Using ablation version: {effective_version}")

    eval_jobs = [
        {
            "name": "manual",
            "dataset": "eval/manual.jsonl",
            "results": f"eval/{effective_version}_manual_results.jsonl",
            "summary": f"eval/{effective_version}_manual_summary.json",
            "category_field": "category",
            "expected_type_field": "expected_type",
        },
        {
            "name": "synthetic",
            "dataset": "eval_v2/synthetic_scaffold_dataset_refined.jsonl",
            "results": f"eval_v2/{effective_version}_synthetic_results.jsonl",
            "summary": f"eval_v2/{effective_version}_synthetic_summary.json",
            "category_field": "refined_category",
            "expected_type_field": "expected_type_refined",
        },
        {
            "name": "holdout",
            "dataset": "eval/holdout_paraphrases.jsonl",
            "results": f"eval/{effective_version}_holdout_results.jsonl",
            "summary": f"eval/{effective_version}_holdout_summary.json",
            "category_field": "category",
            "expected_type_field": "expected_type",
        },
    ]

    if not args.skip_eval:
        for job in eval_jobs:
            cmd = [
                py,
                "scripts/experiments/run_eval_v2_synthetic.py",
                "--dataset",
                job["dataset"],
                "--results",
                job["results"],
                "--summary",
                job["summary"],
                "--category-field",
                job["category_field"],
                "--expected-type-field",
                job["expected_type_field"],
                "--ablation-version",
                effective_version,
            ]
            _run(cmd, cwd=root, label=f"Eval: {job['name']}")

    if not args.skip_phase_gate:
        for job in eval_jobs:
            cmd = [
                py,
                "scripts/experiments/run_phase_gate.py",
                "--dataset",
                job["dataset"],
                "--category-field",
                job["category_field"],
                "--expected-type-field",
                job["expected_type_field"],
                "--phase-dir",
                args.phase_dir,
                "--force-candidate",
                "--ablation-version",
                effective_version,
            ]
            _run(cmd, cwd=root, label=f"Phase Gate: {job['name']}")

    if not args.skip_table:
        table_cmd = [
            py,
            "scripts/compare_ablation.py",
            "--source",
            "files",
        ]

        metadata_path = Path(args.metadata_file)
        if not metadata_path.is_absolute():
            metadata_path = root / metadata_path
        if metadata_path.exists():
            rel = metadata_path.relative_to(root)
            table_cmd.extend(["--metadata-file", str(rel).replace("\\", "/")])

        _run(table_cmd, cwd=root, label="Comparison Table")

    print("\n[OK] Bundle run completed.")


if __name__ == "__main__":
    main()
