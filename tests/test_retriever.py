from pathlib import Path
import math

from src.retrieval.retriever import Retriever, RetrievedChunk


def _fake_chunk(module: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{module}-{score}",
        doc_id=f"doc-{module}",
        module=module,
        score=score,
        text=text,
        source_path=f"data/raw/python_stdlib/{module}.rst",
        heading=None,
        meta={"source_path": f"data/raw/python_stdlib/{module}.rst"},
        start_char=0,
        end_char=100,
        chunk_index=0,
        vector_id=0,
    )


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


def test_extract_symbol_mentions():
    symbols = Retriever._extract_symbol_mentions("What practical job does itertools.chain solve?")
    assert symbols == ["itertools.chain"]


def test_symbol_rerank_prefers_symbol_hit_even_if_raw_score_lower():
    hits = [
        _fake_chunk("collections", "collections.Counter counts frequencies", 0.62),
        _fake_chunk("itertools", "itertools.chain links iterables", 0.58),
    ]

    ranked = Retriever._symbol_rerank(hits, "What practical job does itertools.chain solve?")

    assert ranked[0].module == "itertools"
