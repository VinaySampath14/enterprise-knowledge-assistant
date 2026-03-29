from pathlib import Path
from src.rag.confidence import ConfidenceGate
from src.retrieval.retriever import Retriever, RetrievedChunk


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


def test_confidence_answer_case():
    repo_root = Path(__file__).resolve().parents[1]
    retriever = Retriever(repo_root)
    gate = ConfidenceGate(repo_root)

    hits = retriever.retrieve("How do I open a sqlite3 connection?")
    result = gate.decide(hits, query="How do I open a sqlite3 connection?")

    assert result.decision == "answer"


def test_confidence_refuse_case():
    repo_root = Path(__file__).resolve().parents[1]
    retriever = Retriever(repo_root)
    gate = ConfidenceGate(repo_root)

    hits = retriever.retrieve("What is the capital of France?")
    result = gate.decide(hits, query="What is the capital of France?")

    assert result.decision == "refuse"


def test_confidence_hard_mismatch_refuse_case():
    repo_root = Path(__file__).resolve().parents[1]
    retriever = Retriever(repo_root)
    gate = ConfidenceGate(repo_root)

    query = "In argparse, how do I use glob exactly as documented in argparse?"
    hits = retriever.retrieve(query)
    result = gate.decide(hits, query=query)

    assert result.decision == "refuse"


def test_confidence_recoverable_typo_clarify_case():
    repo_root = Path(__file__).resolve().parents[1]
    retriever = Retriever(repo_root)
    gate = ConfidenceGate(repo_root)

    query = "how to set up squilite connection"
    hits = retriever.retrieve(query)
    result = gate.decide(hits, query=query)

    assert result.decision == "clarify"


def test_mismatch_not_applicable_for_non_explicit_query():
    repo_root = Path(__file__).resolve().parents[1]
    retriever = Retriever(repo_root)
    gate = ConfidenceGate(repo_root)

    query = "How do I open a sqlite3 connection?"
    hits = retriever.retrieve(query)

    subtype, reasons = gate._classify_mismatch(query, hits)

    assert subtype == "not_applicable"
    assert reasons


def test_ambiguity_signals_for_broad_multi_module_query():
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    hits = [
        _chunk("asyncio", 0.56, "d1"),
        _chunk("threading", 0.54, "d2"),
        _chunk("concurrent.futures", 0.50, "d3"),
    ]

    reasons = gate._ambiguity_signals(
        "What is the best way to handle concurrency in Python?",
        hits,
        margin=0.02,
        module_hints=set(),
        use_target=None,
    )

    assert reasons
    assert any("broad/ambiguous phrasing" in r for r in reasons)
    assert any("spans multiple modules" in r for r in reasons)


def test_phase1b_tie_breaker_switches_clarify_to_refuse_for_python_general_query():
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    s1 = max(gate.th_low + 0.01, 0.35)
    s2 = max(s1 - 0.01, gate.th_low)
    hits = [
        _chunk("pathlib", s1, "d1"),
        _chunk("subprocess", s2, "d2"),
        _chunk("venv", s2 - 0.01, "d3"),
    ]

    query = "Which library should I use to install numpy in VSCode?"
    result = gate.decide(hits, query=query)

    assert result.decision == "refuse"
    assert "PHASE1B_INTENT_TIEBREAKER_REFUSE" in result.rationale


def test_phase1b_tie_breaker_does_not_fire_with_plausible_stdlib_anchor():
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    s1 = max(gate.th_low + 0.01, 0.35)
    s2 = max(s1 - 0.01, gate.th_low)
    hits = [
        _chunk("collections", s1, "d1"),
        _chunk("collections", s2, "d1"),
        _chunk("collections", s2 - 0.01, "d1"),
    ]

    query = "give me an overview of what collections.defaultdict does"
    result = gate.decide(hits, query=query)

    assert result.decision == "clarify"
    assert "PHASE1B_INTENT_TIEBREAKER_REFUSE" not in result.rationale


def test_phase1c_tie_breaker_allows_oos_use_target_token_like_pandas():
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    s1 = max(gate.th_high + 0.01, 0.45)
    hits = [
        _chunk("logging", s1, "d1"),
        _chunk("pathlib", s1 - 0.003, "d2"),
        _chunk("statistics", s1 - 0.01, "d3"),
    ]

    query = "how do I use pandas to filter a dataframe?"
    result = gate.decide(hits, query=query)

    assert result.decision == "refuse"
    assert "PHASE1B_INTENT_TIEBREAKER_REFUSE" in result.rationale


def test_phase1c_tie_breaker_covers_ml_concept_pattern():
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    s1 = max(gate.th_low + 0.01, 0.35)
    hits = [
        _chunk("math", s1, "d1"),
        _chunk("statistics", s1 - 0.01, "d2"),
        _chunk("random", s1 - 0.02, "d3"),
    ]

    query = "explain gradient descent in machine learning"
    result = gate.decide(hits, query=query)

    assert result.decision == "refuse"
    assert "PHASE1B_INTENT_TIEBREAKER_REFUSE" in result.rationale


def test_phase2_conceptual_rescue_answers_coherent_stdlib_query():
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    s1 = max(gate.th_high + 0.01, 0.45)
    hits = [
        _chunk("collections", s1, "d1"),
        _chunk("collections", s1 - 0.028, "d1"),
        _chunk("collections", s1 - 0.03, "d1"),
    ]

    query = "give me an overview of what collections.defaultdict does"
    result = gate.decide(hits, query=query)

    assert result.decision == "answer"
    assert "PHASE2_CONCEPTUAL_RESCUE_ANSWER" in result.rationale


def test_phase2_conceptual_rescue_does_not_override_python_general_intent():
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    s1 = max(gate.th_high + 0.01, 0.45)
    hits = [
        _chunk("collections", s1, "d1"),
        _chunk("collections", s1 - 0.028, "d1"),
        _chunk("collections", s1 - 0.03, "d1"),
    ]

    query = "which library should I use to install numpy"
    result = gate.decide(hits, query=query)

    assert result.decision != "answer"
    assert "PHASE2_CONCEPTUAL_RESCUE_ANSWER" not in result.rationale


def test_phase2_intent_scope_refuse_for_near_domain_capability_mismatch():
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    s1 = max(gate.th_high + 0.01, 0.45)
    hits = [
        _chunk("argparse", s1, "d1"),
        _chunk("argparse", s1 - 0.02, "d1"),
        _chunk("argparse", s1 - 0.03, "d1"),
    ]

    query = "does argparse support wildcard file matching?"
    result = gate.decide(hits, query=query)

    assert result.decision == "refuse"
    assert "PHASE2_INTENT_SCOPE_REFUSE" in result.rationale


def test_phase2_intent_scope_refuse_for_exact_doc_in_module_use_pattern():
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    s1 = max(gate.th_high + 0.01, 0.45)
    hits = [
        _chunk("argparse", s1, "d1"),
        _chunk("argparse", s1 - 0.01, "d1"),
        _chunk("heapq", s1 - 0.02, "d2"),
    ]

    query = "In argparse, how do I use heappush exactly as documented in argparse?"
    result = gate.decide(hits, query=query)

    assert result.decision == "refuse"
    assert "PHASE2_INTENT_SCOPE_REFUSE" in result.rationale


def test_phase2_python_general_signals_include_broad_list_tuple_and_gc():
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    signals_a = gate._python_general_out_of_scope_signals(
        "what is the difference between a list and a tuple in Python?"
    )
    signals_b = gate._python_general_out_of_scope_signals(
        "how does Python garbage collection work?"
    )

    assert signals_a
    assert signals_b


def test_phase2_ood_clarify_to_refuse_for_explicit_out_of_domain_query():
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    s1 = max(gate.th_low + 0.01, 0.27)
    hits = [
        _chunk("logging", s1, "d1"),
        _chunk("pathlib", s1 - 0.02, "d2"),
        _chunk("heapq", s1 - 0.03, "d3"),
    ]

    query = "how do I center a div in CSS?"
    result = gate.decide(hits, query=query)

    assert result.decision == "refuse"
    assert "PHASE2_OOD_CLARIFY_TO_REFUSE" in result.rationale


def test_phase2_ood_clarify_to_refuse_does_not_fire_for_typo_like_in_domain_query():
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    s1 = max(gate.th_low + 0.02, 0.30)
    hits = [
        _chunk("sqlite3", s1, "d1"),
        _chunk("sqlite3", s1 - 0.01, "d1"),
        _chunk("sqlite3", s1 - 0.02, "d1"),
    ]

    query = "how to set up squilite connection"
    result = gate.decide(hits, query=query)

    assert "PHASE2_OOD_CLARIFY_TO_REFUSE" not in result.rationale


def test_phase2_python_general_concept_answer_to_refuse_for_gil_query_without_anchor():
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    s1 = max(gate.th_high + 0.02, 0.45)
    hits = [
        _chunk("threading", s1, "d1"),
        _chunk("threading", s1 - 0.02, "d1"),
        _chunk("threading", s1 - 0.03, "d1"),
    ]

    query = "what does the GIL mean for multithreaded Python code?"
    result = gate.decide(hits, query=query)

    assert result.decision == "refuse"
    assert "PHASE2_PY_GENERAL_CONCEPT_REFUSE" in result.rationale


def test_phase2_python_general_concept_answer_to_refuse_does_not_fire_with_module_anchor():
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    s1 = max(gate.th_high + 0.02, 0.45)
    hits = [
        _chunk("threading", s1, "d1"),
        _chunk("threading", s1 - 0.02, "d1"),
        _chunk("threading", s1 - 0.03, "d1"),
    ]

    query = "how does the threading module handle locks?"
    result = gate.decide(hits, query=query)

    assert result.decision == "answer"
    assert "PHASE2_PY_GENERAL_CONCEPT_REFUSE" not in result.rationale
