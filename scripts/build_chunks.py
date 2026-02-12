from __future__ import annotations

from pathlib import Path

from src.chunking.splitter import SplitConfig, chunk_doc_record
from src.utils.jsonl import iter_jsonl, write_jsonl


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    docs_path = repo_root / "data" / "processed" / "docs.jsonl"
    out_path = repo_root / "data" / "processed" / "chunks.jsonl"

    # Your config.yaml says:
    # chunk_size: 800, overlap: 150
    cfg = SplitConfig(chunk_size=800, overlap=150)

    def gen_chunks():
        for doc in iter_jsonl(docs_path):
            for chunk in chunk_doc_record(doc, cfg):
                yield chunk

    n = write_jsonl(out_path, gen_chunks(), append=False)
    print(f"[OK] Wrote {n} chunks to: {out_path}")


if __name__ == "__main__":
    main()
