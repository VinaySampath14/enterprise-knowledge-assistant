from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RUN_DIR_PATTERN = re.compile(r"^V\d+_phase\d+_\d{8}_\d{6}_(baseline|candidate)$")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _kind(name: str) -> str:
    if name.endswith("_baseline"):
        return "baseline"
    if name.endswith("_candidate"):
        return "candidate"
    return "other"


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and RUN_DIR_PATTERN.match(path.name) is not None


def _list_run_dirs(phase_dir: Path) -> List[Path]:
    runs = [p for p in phase_dir.iterdir() if _is_run_dir(p)]
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def _promoted_baselines_all(phase_dir: Path) -> List[Path]:
    pointers = list(phase_dir.glob("promoted_baseline*.json"))
    out: List[Path] = []
    seen: set[Path] = set()

    for pointer in pointers:
        try:
            obj = _load_json(pointer)
        except Exception:
            continue

        run_dir_raw = obj.get("promoted_baseline_run_dir")
        if not run_dir_raw:
            continue

        run_dir = Path(str(run_dir_raw))
        if run_dir.exists() and _is_run_dir(run_dir) and run_dir not in seen:
            seen.add(run_dir)
            out.append(run_dir)

    return out


def _plan_cleanup(
    phase_dir: Path,
    keep_baselines: int,
    keep_candidates: int,
    keep_latest_any: int,
) -> Tuple[List[Path], List[Path]]:
    runs = _list_run_dirs(phase_dir)
    promoted_all = _promoted_baselines_all(phase_dir)

    baselines = [p for p in runs if _kind(p.name) == "baseline"]
    candidates = [p for p in runs if _kind(p.name) == "candidate"]

    keep: List[Path] = []
    keep.extend(promoted_all)
    keep.extend(baselines[: max(0, keep_baselines)])
    keep.extend(candidates[: max(0, keep_candidates)])
    keep.extend(runs[: max(0, keep_latest_any)])

    seen: set[Path] = set()
    keep_unique: List[Path] = []
    for p in keep:
        if p in seen:
            continue
        seen.add(p)
        keep_unique.append(p)

    archive = [p for p in runs if p not in keep_unique]
    return keep_unique, archive


def _archive_runs(archive_runs: List[Path], archive_root: Path, dry_run: bool) -> Optional[Path]:
    if not archive_runs:
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_dir = archive_root / f"phase_cleanup_{stamp}"

    if dry_run:
        return batch_dir

    batch_dir.mkdir(parents=True, exist_ok=True)
    for run in archive_runs:
        target = batch_dir / run.name
        shutil.move(str(run), str(target))
    return batch_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive old phase run folders with promoted-baseline-safe retention"
    )
    parser.add_argument(
        "--phase-dir",
        default="artifacts/experiments/phase0",
        help="Phase run directory relative to repo root",
    )
    parser.add_argument(
        "--archive-dir",
        default="artifacts/archive/experiments/phase0",
        help="Archive root relative to repo root",
    )
    parser.add_argument(
        "--keep-baselines",
        type=int,
        default=2,
        help="How many most-recent baseline runs to keep",
    )
    parser.add_argument(
        "--keep-candidates",
        type=int,
        default=3,
        help="How many most-recent candidate runs to keep",
    )
    parser.add_argument(
        "--keep-latest-any",
        type=int,
        default=4,
        help="How many newest runs to keep regardless of type",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually archive runs. Without this flag, a dry-run plan is printed.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    phase_dir = root / args.phase_dir
    archive_root = root / args.archive_dir

    if not phase_dir.exists():
        raise FileNotFoundError(f"phase dir not found: {phase_dir}")

    keep, archive = _plan_cleanup(
        phase_dir,
        keep_baselines=args.keep_baselines,
        keep_candidates=args.keep_candidates,
        keep_latest_any=args.keep_latest_any,
    )

    dry_run = not args.execute
    batch_dir = _archive_runs(archive, archive_root, dry_run=dry_run)
    promoted_all = _promoted_baselines_all(phase_dir)

    print("=== Cleanup Plan ===")
    print(f"phase_dir: {phase_dir}")
    print(f"dry_run: {dry_run}")
    if promoted_all:
        print(f"promoted_runs_protected: {[p.name for p in promoted_all]}")
    print(f"keep_count: {len(keep)}")
    print(f"archive_count: {len(archive)}")
    if batch_dir is not None:
        print(f"archive_batch_dir: {batch_dir}")

    print("\nKeep:")
    for p in keep:
        print(f"- {p.name}")

    print("\nArchive:")
    for p in archive:
        print(f"- {p.name}")

    if dry_run:
        print("\nNo files moved. Re-run with --execute to archive the listed runs.")
    else:
        print("\nArchive completed.")


if __name__ == "__main__":
    main()
