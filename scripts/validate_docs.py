from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.utils.jsonl import iter_jsonl


REQUIRED_TOP_LEVEL_KEYS = {"id", "module", "source", "text", "sha256", "created_at"}
REQUIRED_SOURCE_KEYS = {"path", "type"}


def _is_hex_sha256(s: str) -> bool:
    if not isinstance(s, str) or len(s) != 64:
        return False
    try:
        int(s, 16)
        return True
    except ValueError:
        return False


def validate_record(rec: Dict[str, Any], lineno: int) -> List[str]:
    """Return a list of validation error messages for this record."""
    errors: List[str] = []

    missing = REQUIRED_TOP_LEVEL_KEYS.difference(rec.keys())
    if missing:
        errors.append(f"line {lineno}: missing keys: {sorted(missing)}")
        return errors  # can't continue safely

    if not isinstance(rec["id"], str) or not rec["id"].strip():
        errors.append(f"line {lineno}: 'id' must be a non-empty string")

    if not isinstance(rec["module"], str) or not rec["module"].strip():
        errors.append(f"line {lineno}: 'module' must be a non-empty string")

    src = rec["source"]
    if not isinstance(src, dict):
        errors.append(f"line {lineno}: 'source' must be an object/dict")
    else:
        missing_src = REQUIRED_SOURCE_KEYS.difference(src.keys())
        if missing_src:
            errors.append(f"line {lineno}: source missing keys: {sorted(missing_src)}")
        else:
            if src.get("type") != "rst":
                errors.append(f"line {lineno}: source.type must be 'rst'")
            if not isinstance(src.get("path"), str) or not src["path"].strip():
                errors.append(f"line {lineno}: source.path must be a non-empty string")

    if not isinstance(rec["text"], str) or not rec["text"].strip():
        errors.append(f"line {lineno}: 'text' must be a non-empty string")

    if not _is_hex_sha256(rec["sha256"]):
        errors.append(f"line {lineno}: 'sha256' must be a 64-char hex string")

    if not isinstance(rec["created_at"], str) or not rec["created_at"].strip():
        errors.append(f"line {lineno}: 'created_at' must be a non-empty string (ISO-8601 recommended)")

    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    docs_path = repo_root / "data" / "processed" / "docs.jsonl"

    if not docs_path.exists():
        print(f"[ERROR] File not found: {docs_path}")
        print("Run: python -m scripts.build_docs")
        return 2

    total_docs = 0
    total_chars = 0
    total_nonempty_lines = 0

    # Track largest doc
    largest_id = None
    largest_module = None
    largest_chars = -1
    largest_source_path = None

    seen_ids = set()
    all_errors: List[str] = []

    for lineno, rec in enumerate(iter_jsonl(docs_path), start=1):
        total_docs += 1

        # schema validation
        errs = validate_record(rec, lineno)
        if errs:
            all_errors.extend(errs)
            continue

        # uniqueness
        doc_id = rec["id"]
        if doc_id in seen_ids:
            all_errors.append(f"line {lineno}: duplicate id '{doc_id}'")
        else:
            seen_ids.add(doc_id)

        text = rec["text"]
        n_chars = len(text)
        total_chars += n_chars

        # approximate "lines" stat (not required, but helpful)
        total_nonempty_lines += sum(1 for ln in text.splitlines() if ln.strip())

        if n_chars > largest_chars:
            largest_chars = n_chars
            largest_id = rec["id"]
            largest_module = rec["module"]
            largest_source_path = rec["source"]["path"]

    # Print errors if any
    if all_errors:
        print("[FAIL] Schema validation errors found:\n")
        for e in all_errors[:50]:
            print(" -", e)
        if len(all_errors) > 50:
            print(f"\n... plus {len(all_errors) - 50} more errors.")
        return 1

    # Print stats
    avg_chars = (total_chars / total_docs) if total_docs else 0.0
    avg_nonempty_lines = (total_nonempty_lines / total_docs) if total_docs else 0.0

    print("[OK] docs.jsonl validated successfully\n")
    print("Basic stats")
    print("-----------")
    print(f"Docs count           : {total_docs}")
    print(f"Total characters     : {total_chars}")
    print(f"Avg chars per doc    : {avg_chars:.1f}")
    print(f"Avg non-empty lines  : {avg_nonempty_lines:.1f}")
    print("")
    print("Largest document")
    print("---------------")
    print(f"Doc id               : {largest_id}")
    print(f"Module               : {largest_module}")
    print(f"Source path          : {largest_source_path}")
    print(f"Characters           : {largest_chars}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
