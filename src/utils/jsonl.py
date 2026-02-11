from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(
    path: Path,
    records: Iterable[Dict[str, Any]],
    *,
    append: bool = False,
    ensure_ascii: bool = False,
) -> int:
    """Write dict records to JSONL. Returns number of records written."""
    ensure_parent_dir(path)
    mode = "a" if append else "w"
    count = 0
    with path.open(mode, encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=ensure_ascii) + "\n")
            count += 1
    return count


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Stream JSONL dict records from disk."""
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path} line {lineno}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path} line {lineno}")
            yield obj


def read_jsonl(path: Path, *, max_records: Optional[int] = None) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for i, obj in enumerate(iter_jsonl(path), start=1):
        out.append(obj)
        if max_records is not None and i >= max_records:
            break
    return out
