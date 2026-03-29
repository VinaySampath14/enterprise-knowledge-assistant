from pathlib import Path

from src.rag.confidence import ConfidenceGate
from src.retrieval.retriever import RetrievedChunk
from src.rag.pipeline import (
    _should_block_post_generation_refusal_override,
)


def _chunk(module: str, score: float, doc_id: str, text: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{module}_{doc_id}_{score}",
        doc_id=doc_id,
        module=module,
        score=score,
        text=text or f"{module} module docs explain defaultdict behavior and usage examples.",
        source_path=f"data/raw/python_stdlib/{module}.rst",
        heading=None,
        meta={"source_path": f"data/raw/python_stdlib/{module}.rst"},
        start_char=0,
        end_char=120,
        chunk_index=0,
        vector_id=0,
    )


def test_override_guard_preserves_grounded_coherent_stdlib_answer() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    chunks = [
        _chunk("collections", 0.71, "d1"),
        _chunk("collections", 0.69, "d1"),
        _chunk("collections", 0.67, "d1"),
    ]
    answer_text = (
        "collections.defaultdict creates default values for missing keys and simplifies grouped counting; "
        "i don't have enough information about one edge-case variant."
    )

    fire, reasons = _should_block_post_generation_refusal_override(
        query="give me an overview of what collections.defaultdict does",
        answer_text=answer_text,
        gate_decision="answer",
        mismatch_subtype="not_applicable",
        used_chunks=chunks,
        citation_ids=[1],
        gate=gate,
    )

    assert fire is True
    assert reasons


def test_override_guard_does_not_fire_for_out_of_scope_intent() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    chunks = [
        _chunk("collections", 0.71, "d1"),
        _chunk("collections", 0.69, "d1"),
        _chunk("collections", 0.67, "d1"),
    ]
    answer_text = (
        "I don't have enough information in one section, but maybe use pandas and numpy "
        "for this workflow."
    )

    fire, _ = _should_block_post_generation_refusal_override(
        query="which library should I use to install numpy",
        answer_text=answer_text,
        gate_decision="answer",
        mismatch_subtype="not_applicable",
        used_chunks=chunks,
        citation_ids=[1],
        gate=gate,
    )

    assert fire is False


def test_override_guard_does_not_fire_for_unsupported_case() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    chunks = [
        _chunk("collections", 0.71, "d1"),
        _chunk("collections", 0.69, "d1"),
        _chunk("collections", 0.67, "d1"),
    ]

    fire, _ = _should_block_post_generation_refusal_override(
        query="what is the plot of dune",
        answer_text="I do not have enough information in the Python standard library documentation to answer that.",
        gate_decision="answer",
        mismatch_subtype="not_applicable",
        used_chunks=chunks,
        citation_ids=[],
        gate=gate,
    )

    assert fire is False


def test_override_guard_does_not_fire_without_in_domain_anchor() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    chunks = [
        _chunk("collections", 0.71, "d1"),
        _chunk("collections", 0.69, "d1"),
        _chunk("collections", 0.67, "d1"),
    ]

    fire, _ = _should_block_post_generation_refusal_override(
        query="please explain this concept broadly",
        answer_text=(
            "collections.defaultdict helps with grouped counting, though I don't have enough information "
            "about one edge-case variant."
        ),
        gate_decision="answer",
        mismatch_subtype="not_applicable",
        used_chunks=chunks,
        citation_ids=[1],
        gate=gate,
    )

    assert fire is False


def test_override_guard_does_not_fire_for_python_general_concept_even_with_citations() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    chunks = [
        _chunk("threading", 0.72, "d1"),
        _chunk("threading", 0.69, "d1"),
        _chunk("threading", 0.66, "d1"),
    ]

    fire, _ = _should_block_post_generation_refusal_override(
        query="what does the GIL mean for multithreaded Python code?",
        answer_text=(
            "The GIL affects multithreaded execution, but I don't have enough information "
            "for one low-level runtime detail."
        ),
        gate_decision="answer",
        mismatch_subtype="not_applicable",
        used_chunks=chunks,
        citation_ids=[1],
        gate=gate,
    )

    assert fire is False


def test_override_guard_can_fire_without_citations_for_strong_symbol_anchor() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    chunks = [
        _chunk("itertools", 0.56, "d1", text="itertools.chain links multiple iterables together."),
        _chunk("itertools", 0.54, "d1", text="Use itertools.chain for flattening-like iteration."),
        _chunk("itertools", 0.52, "d1", text="chain() yields values lazily from input iterables."),
    ]

    fire, reasons = _should_block_post_generation_refusal_override(
        query="What practical job does itertools.chain solve?",
        answer_text=(
            "itertools.chain can combine multiple iterables in sequence, but I don't have enough information "
            "for one edge case."
        ),
        gate_decision="answer",
        mismatch_subtype="not_applicable",
        used_chunks=chunks,
        citation_ids=[],
        gate=gate,
    )

    assert fire is True
    assert reasons


def test_override_guard_no_citation_symbol_fallback_requires_symbol_in_evidence() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    gate = ConfidenceGate(repo_root)

    chunks = [
        _chunk("collections", 0.56, "d1", text="collections.Counter helps frequency counting."),
        _chunk("collections", 0.54, "d1", text="Counter returns dict-like counts."),
        _chunk("collections", 0.52, "d1", text="Counter supports update and most_common."),
    ]

    fire, _ = _should_block_post_generation_refusal_override(
        query="What practical job does itertools.chain solve?",
        answer_text=(
            "itertools.chain can combine multiple iterables in sequence, but I don't have enough information "
            "for one edge case."
        ),
        gate_decision="answer",
        mismatch_subtype="not_applicable",
        used_chunks=chunks,
        citation_ids=[],
        gate=gate,
    )

    assert fire is False






