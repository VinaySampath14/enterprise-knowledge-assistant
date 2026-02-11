from __future__ import annotations

from pathlib import Path

from src.ingest.load_raw import load_raw_docs
from src.utils.jsonl import write_jsonl


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = repo_root / "data" / "raw" / "python_stdlib"
    out_path = repo_root / "data" / "processed" / "docs.jsonl"

    docs = load_raw_docs(raw_dir)
    n = write_jsonl(out_path, docs, append=False)

    print(f"[OK] Wrote {n} docs to: {out_path}")


if __name__ == "__main__":
    main()
