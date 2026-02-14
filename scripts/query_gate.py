from __future__ import annotations

import sys
from pathlib import Path

from src.retrieval.retriever import Retriever
from src.rag.confidence import ConfidenceGate


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = "How do I join paths in Python?"

    retriever = Retriever(repo_root)
    gate = ConfidenceGate(repo_root)

    hits = retriever.retrieve(query)
    res = gate.decide(hits)

    print(f"\nQUERY: {query}")
    print(f"DECISION: {res.decision}")
    print(f"top_score={res.top_score:.4f} second={res.second_score:.4f} margin={res.margin:.4f}")
    print(f"rationale: {res.rationale}\n")

    for i, h in enumerate(res.used_chunks[:5], start=1):
        print(f"[{i}] score={h.score:.4f} module={h.module} chunk_id={h.chunk_id}")
        print(f"    source={h.meta.get('source_path')}")
        preview = h.text[:200].replace("\n", " ")
        print(f"    preview={preview}\n")



if __name__ == "__main__":
    main()
