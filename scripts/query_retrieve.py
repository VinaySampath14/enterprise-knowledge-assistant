from __future__ import annotations

import sys
from pathlib import Path

from sympy import preview

from src.retrieval.retriever import Retriever


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    r = Retriever(repo_root)

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = "How do I join paths in Python?"

    hits = r.retrieve(query)

    print(f"\nQUERY: {query}\n")
    for i, h in enumerate(hits, start=1):
        src = h.meta.get("source_path")
        print(f"[{i}] score={h.score:.4f}  module={h.module}  chunk_id={h.chunk_id}")
        if src:
            print(f"    source={src}")
        preview = h.text[:200].replace("\n", " ")
        print(f"    text_preview={preview}")

        print("")


if __name__ == "__main__":
    main()
