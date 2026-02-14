from pathlib import Path
import math

from src.retrieval.retriever import Retriever


def test_retriever_topk_and_scores():
    repo_root = Path(__file__).resolve().parents[1]
    r = Retriever(repo_root)

    k = 5
    hits = r.retrieve("How do I create a temporary file?", top_k=k)

    assert len(hits) == k or len(hits) > 0  # if corpus small, can be <k
    for h in hits:
        assert isinstance(h.score, float)
        assert not math.isnan(h.score)
        assert h.text.strip()
        assert h.meta.get("source_path")  # ensures traceability


def test_retriever_deterministic_order():
    repo_root = Path(__file__).resolve().parents[1]
    r = Retriever(repo_root)

    q = "How do I join paths in Python?"
    hits1 = [h.chunk_id for h in r.retrieve(q, top_k=5)]
    hits2 = [h.chunk_id for h in r.retrieve(q, top_k=5)]
    assert hits1 == hits2
