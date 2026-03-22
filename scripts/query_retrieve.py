from __future__ import annotations

import sys
from dataclasses import asdict
import json
from pathlib import Path

from sympy import preview

from src.retrieval.retriever import Retriever
from src.rag.prompt import format_retrieved_chunks


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    r = Retriever(repo_root)

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = "How do I join paths in Python?"

    hits = r.retrieve(query)
    formatted_context, source_map = format_retrieved_chunks(hits)

    print(f"\nQUERY: {query}\n")
    if hits:
        print("[DEBUG] First retrieved item (all fields):")
        for k, v in asdict(hits[0]).items():
            print(f"  {k}: {v}")
        print("")

    if formatted_context:
        print("[DEBUG] Citation-aware formatted context:")
        print(formatted_context)
        print("")

    if source_map:
        print("[DEBUG] Structured source mapping:")
        print(json.dumps(source_map, indent=2, ensure_ascii=False))
        print("")

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
