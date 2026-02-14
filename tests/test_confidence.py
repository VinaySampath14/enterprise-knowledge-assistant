from pathlib import Path
from src.rag.confidence import ConfidenceGate
from src.retrieval.retriever import Retriever


def test_confidence_answer_case():
    repo_root = Path(__file__).resolve().parents[1]
    retriever = Retriever(repo_root)
    gate = ConfidenceGate(repo_root)

    hits = retriever.retrieve("How do I open a sqlite3 connection?")
    result = gate.decide(hits)

    assert result.decision == "answer"


def test_confidence_refuse_case():
    repo_root = Path(__file__).resolve().parents[1]
    retriever = Retriever(repo_root)
    gate = ConfidenceGate(repo_root)

    hits = retriever.retrieve("What is the capital of France?")
    result = gate.decide(hits)

    assert result.decision == "refuse"
