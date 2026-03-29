from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _run_cmd(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )


def _latest_run_dir(phase_dir: Path) -> Optional[Path]:
    if not phase_dir.exists():
        return None
    candidates = [p for p in phase_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _print_proc(label: str, proc: subprocess.CompletedProcess[str]) -> None:
    print(f"\n=== {label} ===")
    if proc.stdout.strip():
        print(proc.stdout)
    if proc.stderr.strip():
        print(proc.stderr)
    print(f"exit_code={proc.returncode}")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _slug(value: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()[:120] if s else "default"


def _build_signature(
    root: Path,
    dataset_rel: str,
    category_field: str,
    expected_type_field: str,
) -> Dict[str, Any]:
    dataset_path = root / dataset_rel
    config_path = root / "config.yaml"
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    return {
        "dataset_path": dataset_rel,
        "dataset_sha256": _sha256_file(dataset_path),
        "config_path": "config.yaml",
        "config_sha256": _sha256_file(config_path),
        "category_field": category_field,
        "expected_type_field": expected_type_field,
    }


def _extract_signature_from_run(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in baseline run: {run_dir}")
    meta = _load_json(meta_path)

    dataset = meta.get("dataset", {})
    config = meta.get("config", {})
    return {
        "dataset_path": dataset.get("path"),
        "dataset_sha256": dataset.get("sha256"),
        "config_path": config.get("path"),
        "config_sha256": config.get("sha256"),
        "category_field": dataset.get("category_field"),
        "expected_type_field": dataset.get("expected_type_field"),
    }


def _signature_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    keys = [
        "dataset_path",
        "dataset_sha256",
        "config_path",
        "config_sha256",
        "category_field",
        "expected_type_field",
    ]
    return all(a.get(k) == b.get(k) for k in keys)


def _save_promoted_baseline(pointer_path: Path, run_dir: Path, signature: Dict[str, Any]) -> None:
    obj = {
        "promoted_baseline_run_dir": str(run_dir.resolve()),
        "promoted_baseline_run_id": run_dir.name,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "signature": signature,
    }
    _save_json(pointer_path, obj)


def _load_promoted_baseline(pointer_path: Path) -> Optional[Dict[str, Any]]:
    if not pointer_path.exists():
        return None
    return _load_json(pointer_path)


def _resolve_promoted_pointer_path(
    phase_dir: Path,
    dataset_rel: str,
    category_field: str,
    expected_type_field: str,
) -> Path:
    scope = f"{dataset_rel}|{category_field}|{expected_type_field}"
    return phase_dir / f"promoted_baseline__{_slug(scope)}.json"


def _find_usable_promoted_baseline(
    scoped_pointer: Path,
    fallback_pointer: Path,
    current_signature: Dict[str, Any],
) -> Optional[Tuple[Path, Dict[str, Any]]]:
    # Prefer dataset-scoped pointer.
    scoped_obj = _load_promoted_baseline(scoped_pointer)
    if scoped_obj is not None:
        return scoped_pointer, scoped_obj

    # Legacy fallback pointer is only usable when signatures match.
    fallback_obj = _load_promoted_baseline(fallback_pointer)
    if fallback_obj is not None:
        sig = fallback_obj.get("signature", {})
        if _signature_equal(sig, current_signature):
            return fallback_pointer, fallback_obj

    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase gate runner with promoted baseline reuse. "
            "Creates candidate only when inputs changed unless forced."
        )
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
        "--phase-dir",
        default="artifacts/experiments/phase0",
        help="Phase artifact root relative to repo root.",
    )
    parser.add_argument(
        "--baseline-run",
        default="",
        help=(
            "Optional existing baseline run dir. "
            "If omitted, promoted baseline pointer is used."
        ),
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Deprecated alias for reuse mode; requires --baseline-run.",
    )
    parser.add_argument(
        "--rebaseline",
        action="store_true",
        help="Intentionally create a new baseline and promote it.",
    )
    parser.add_argument(
        "--force-candidate",
        action="store_true",
        help="Create candidate even when no input signature changes are detected.",
    )
    parser.add_argument(
        "--promoted-baseline-file",
        default="",
        help=(
            "Path to promoted baseline pointer JSON. "
            "Defaults to <phase-dir>/promoted_baseline.json"
        ),
    )
    parser.add_argument(
        "--repro-threshold",
        type=float,
        default=0.95,
        help="Reproducibility threshold passed to phase0 runner.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    phase_dir = root / args.phase_dir
    phase_dir.mkdir(parents=True, exist_ok=True)

    fallback_promoted_file = phase_dir / "promoted_baseline.json"
    scoped_promoted_file = _resolve_promoted_pointer_path(
        phase_dir,
        args.dataset,
        args.category_field,
        args.expected_type_field,
    )

    promoted_file = (
        Path(args.promoted_baseline_file).resolve()
        if args.promoted_baseline_file
        else scoped_promoted_file
    )

    current_signature = _build_signature(
        root,
        args.dataset,
        args.category_field,
        args.expected_type_field,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    baseline_run_dir: Optional[Path] = None
    if args.promoted_baseline_file:
        promoted_obj = _load_promoted_baseline(promoted_file)
    else:
        found = _find_usable_promoted_baseline(
            scoped_promoted_file,
            fallback_promoted_file,
            current_signature,
        )
        promoted_obj = found[1] if found is not None else None

    if args.skip_baseline and not args.baseline_run:
        raise ValueError("--skip-baseline requires --baseline-run")

    if args.rebaseline:
        baseline_run_id = f"V1_phase0_{ts}_baseline"
        baseline_cmd = [
            sys.executable,
            str(root / "scripts" / "experiments" / "run_phase0_baseline.py"),
            "--dataset",
            args.dataset,
            "--category-field",
            args.category_field,
            "--expected-type-field",
            args.expected_type_field,
            "--phase-dir",
            args.phase_dir,
            "--run-id",
            baseline_run_id,
            "--repro-threshold",
            str(args.repro_threshold),
        ]
        baseline_proc = _run_cmd(baseline_cmd, root)
        _print_proc("Baseline Run (Rebaseline)", baseline_proc)
        if baseline_proc.returncode != 0:
            raise RuntimeError("baseline run failed")
        baseline_run_dir = phase_dir / baseline_run_id
        _save_promoted_baseline(promoted_file, baseline_run_dir, current_signature)
        if promoted_file != fallback_promoted_file and not fallback_promoted_file.exists():
            _save_promoted_baseline(fallback_promoted_file, baseline_run_dir, current_signature)
        print(f"[OK] Promoted baseline updated: {baseline_run_dir}")

        if not args.force_candidate:
            print("\n=== Final Phase Gate Summary ===")
            print(f"baseline_run: {baseline_run_dir}")
            print("candidate_run: skipped (rebaseline only)")
            print("promotion_recommendation: GO (new baseline promoted)")
            return

    if baseline_run_dir is None:
        if args.baseline_run:
            baseline_run_dir = Path(args.baseline_run).resolve()
            if not baseline_run_dir.exists():
                raise FileNotFoundError(f"baseline run dir not found: {baseline_run_dir}")
        elif promoted_obj is not None:
            baseline_run_dir = Path(str(promoted_obj.get("promoted_baseline_run_dir", ""))).resolve()
            if not baseline_run_dir.exists():
                raise FileNotFoundError(
                    f"promoted baseline run dir not found: {baseline_run_dir}. "
                    "Run with --rebaseline to create a new promoted baseline."
                )
        else:
            baseline_run_id = f"V1_phase0_{ts}_baseline"
            baseline_cmd = [
                sys.executable,
                str(root / "scripts" / "experiments" / "run_phase0_baseline.py"),
                "--dataset",
                args.dataset,
                "--category-field",
                args.category_field,
                "--expected-type-field",
                args.expected_type_field,
                "--phase-dir",
                args.phase_dir,
                "--run-id",
                baseline_run_id,
                "--repro-threshold",
                str(args.repro_threshold),
            ]
            baseline_proc = _run_cmd(baseline_cmd, root)
            _print_proc("Baseline Run (Initial)", baseline_proc)
            if baseline_proc.returncode != 0:
                raise RuntimeError("baseline run failed")
            baseline_run_dir = phase_dir / baseline_run_id
            _save_promoted_baseline(promoted_file, baseline_run_dir, current_signature)
            if promoted_file != fallback_promoted_file and not fallback_promoted_file.exists():
                _save_promoted_baseline(fallback_promoted_file, baseline_run_dir, current_signature)
            print(f"[OK] Promoted baseline created: {baseline_run_dir}")

            if not args.force_candidate:
                print("\n=== Final Phase Gate Summary ===")
                print(f"baseline_run: {baseline_run_dir}")
                print("candidate_run: skipped (initial baseline creation)")
                print("promotion_recommendation: GO (baseline established)")
                return

    baseline_signature = _extract_signature_from_run(baseline_run_dir)
    has_changes = not _signature_equal(current_signature, baseline_signature)

    if not has_changes and not args.force_candidate:
        print("\n=== Final Phase Gate Summary ===")
        print(f"baseline_run: {baseline_run_dir}")
        print("candidate_run: skipped (no signature changes detected)")
        print(f"promoted_baseline_file: {promoted_file}")
        print("promotion_recommendation: NO-OP (reuse promoted baseline)")
        return

    candidate_run_id = f"V2_phase0_{ts}_candidate"
    candidate_cmd = [
        sys.executable,
        str(root / "scripts" / "experiments" / "run_phase0_baseline.py"),
        "--dataset",
        args.dataset,
        "--category-field",
        args.category_field,
        "--expected-type-field",
        args.expected_type_field,
        "--phase-dir",
        args.phase_dir,
        "--run-id",
        candidate_run_id,
        "--repro-threshold",
        str(args.repro_threshold),
    ]
    candidate_proc = _run_cmd(candidate_cmd, root)
    _print_proc("Candidate Run", candidate_proc)
    if candidate_proc.returncode != 0:
        raise RuntimeError("candidate run failed")

    candidate_run_dir = phase_dir / candidate_run_id

    compare_cmd = [
        sys.executable,
        str(root / "scripts" / "experiments" / "compare_ablation_runs.py"),
        "--baseline-run",
        str(baseline_run_dir),
        "--current-run",
        str(candidate_run_dir),
    ]
    compare_proc = _run_cmd(compare_cmd, root)
    _print_proc("Comparison Run", compare_proc)
    if compare_proc.returncode != 0:
        raise RuntimeError("comparison run failed")

    comparison_dir = candidate_run_dir / f"comparison_vs_{baseline_run_dir.name}"
    comparison_json = comparison_dir / "comparison.json"
    if not comparison_json.exists():
        # Best effort fallback in case comparator output path changes.
        fallback = _latest_run_dir(candidate_run_dir)
        if fallback is not None and (fallback / "comparison.json").exists():
            comparison_json = fallback / "comparison.json"

    if not comparison_json.exists():
        raise FileNotFoundError("comparison.json was not produced")

    cmp_obj = _load_json(comparison_json)
    guardrails = cmp_obj.get("guardrails", {})
    decision = "GO" if guardrails.get("overall_passed") else "NO-GO"

    print("\n=== Final Phase Gate Summary ===")
    print(f"baseline_run: {baseline_run_dir}")
    print(f"candidate_run: {candidate_run_dir}")
    print(f"comparison_json: {comparison_json}")
    print(f"guardrail_overall_passed: {guardrails.get('overall_passed')}")
    print(f"promotion_recommendation: {decision}")
    print(f"promoted_baseline_file: {promoted_file}")


if __name__ == "__main__":
    main()
