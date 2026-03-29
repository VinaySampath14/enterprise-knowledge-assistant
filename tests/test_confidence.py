from pathlib import Path
from src.rag.confidence import ConfidenceGate
from src.retrieval.retriever import RetrievedChunk


def _chunk(module: str, score: float, doc_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{module}_{doc_id}_{score}",
        doc_id=doc_id,
        module=module,
        score=score,
        text=f"Documentation snippet for {module}",
        source_path=f"data/raw/python_stdlib/{module}.rst",
        heading=None,
        meta={"source_path": f"data/raw/python_stdlib/{module}.rst"},
        start_char=0,
        end_char=100,
        chunk_index=0,
        vector_id=0,
    )


def _gate() -> ConfidenceGate:
    repo_root = Path(__file__).resolve().parents[1]
    return ConfidenceGate(repo_root)


def test_confidence_answer_case_with_coherent_high_scores():
    gate = _gate()
    s1 = max(gate.th_high + 0.03, 0.45)
    hits = [
        _chunk("sqlite3", s1, "d1"),
        _chunk("sqlite3", s1 - 0.01, "d1"),
        _chunk("sqlite3", s1 - 0.02, "d1"),
    ]

    result = gate.decide(hits, query="How do I open a sqlite3 connection?")

    assert result.decision == "answer"


def test_confidence_refuse_case_for_low_score():
    gate = _gate()
    s1 = max(gate.th_low - 0.01, 0.05)
    hits = [
        _chunk("sqlite3", s1, "d1"),
        _chunk("sqlite3", max(s1 - 0.01, 0.0), "d1"),
    ]

    result = gate.decide(hits, query="What is the capital of France?")

    assert result.decision == "refuse"


def test_confidence_hard_mismatch_refuse_case():
    gate = _gate()
    s1 = max(gate.th_high + 0.02, 0.45)
    hits = [
        _chunk("heapq", s1, "d1"),
        _chunk("heapq", s1 - 0.01, "d1"),
        _chunk("heapq", s1 - 0.02, "d1"),
    ]

    query = "In argparse, how do I use heappush exactly as documented in argparse?"
    result = gate.decide(hits, query=query)

    assert result.decision == "refuse"


def test_confidence_recoverable_typo_clarify_case():
    gate = _gate()
    s1 = max(gate.th_high + 0.02, 0.45)
    hits = [
        _chunk("sqlite3", s1, "d1"),
        _chunk("sqlite3", s1 - 0.01, "d1"),
        _chunk("sqlite3", s1 - 0.02, "d1"),
    ]

    query = "How do I use sqlite_conect in sqlite3?"
    result = gate.decide(hits, query=query)

    assert result.decision == "clarify"


def test_mismatch_not_applicable_for_non_explicit_query():
    gate = _gate()

    query = "How do I open a sqlite3 connection?"
    hits = [
        _chunk("sqlite3", 0.5, "d1"),
        _chunk("sqlite3", 0.48, "d1"),
    ]

    subtype, reasons = gate._classify_mismatch(query, hits)

    assert subtype == "not_applicable"
    assert reasons


def test_middle_band_clarify_case():
    gate = _gate()
    s1 = max(gate.th_low + 0.03, 0.32)
    s1 = min(s1, gate.th_high - 0.01)

    hits = [
        _chunk("asyncio", s1, "d1"),
        _chunk("asyncio", max(s1 - 0.02, 0.0), "d1"),
    ]

    result = gate.decide(hits, query="How does asyncio scheduling work?")

    assert result.decision == "clarify"


def test_strong_conflicting_topics_clarify_case():
    gate = _gate()
    s1 = gate.th_high + 0.005
    s2 = gate.th_high - 0.01

    hits = [
        _chunk("asyncio", s1, "d1"),
        _chunk("threading", s2, "d2"),
        _chunk("threading", s2 - 0.02, "d2"),
    ]

    result = gate.decide(hits, query="How should I manage parallel tasks?")

    assert result.decision == "clarify"


def test_no_hits_refuse_case():
    gate = _gate()
    result = gate.decide([], query="Any query")

    assert result.decision == "refuse"


def test_no_lexical_intent_override_in_middle_band():
    gate = _gate()
    s1 = max(gate.th_low + 0.02, 0.30)
    s1 = min(s1, gate.th_high - 0.01)

    hits = [
        _chunk("pathlib", s1, "d1"),
        _chunk("logging", s1 - 0.01, "d2"),
    ]

    query = "how do I center a div in CSS?"
    result = gate.decide(hits, query=query)

    assert result.decision == "clarify"
