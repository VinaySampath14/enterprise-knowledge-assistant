from __future__ import annotations

from pathlib import Path
import json
import sys

# Ensure repo root is on sys.path when this file is executed directly.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.eval_runner.synthetic_prep import build_synthetic_scaffold


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    out = build_synthetic_scaffold(repo_root)

    print(f"[OK] Wrote synthetic dataset: {out['dataset_path']}")
    print(f"[OK] Wrote synthetic summary: {out['summary_path']}")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
