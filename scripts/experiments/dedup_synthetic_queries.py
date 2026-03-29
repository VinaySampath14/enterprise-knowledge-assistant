from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dedup_by_query(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    dropped = 0

    for row in rows:
        q = str(row.get("query", "")).strip()
        if not q:
            out.append(row)
            continue

        key = q.lower()
        if key in seen:
            dropped += 1
            continue

        seen.add(key)
        out.append(row)

    return out, len(rows), dropped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate synthetic dataset rows by identical query text."
    )
    parser.add_argument(
        "--input",
        default="eval_v2/synthetic_scaffold_dataset_refined.jsonl",
        help="Input JSONL dataset path relative to repo root.",
    )
    parser.add_argument(
        "--output",
        default="eval_v2/synthetic_scaffold_dataset_refined_dedup.jsonl",
        help="Output JSONL dataset path relative to repo root.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    in_path = repo_root / args.input
    out_path = repo_root / args.output

    if not in_path.exists():
        raise FileNotFoundError(f"input dataset not found: {in_path}")

    rows = _read_jsonl(in_path)
    deduped, total_before, dropped = _dedup_by_query(rows)

    _write_jsonl(out_path, deduped)

    print(f"[OK] Wrote deduplicated dataset: {out_path}")
    print(f"total_before={total_before}")
    print(f"total_after={len(deduped)}")
    print(f"dropped_duplicates={dropped}")


if __name__ == "__main__":
    main()
