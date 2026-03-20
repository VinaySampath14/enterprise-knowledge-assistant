from __future__ import annotations

from typing import List, Dict, Any
import re


def compute_groundedness(answer: str, used_chunks: list) -> float:
    """
    Lexical groundedness proxy:
    fraction of unique (>=3 chars) tokens in the answer that appear in the retrieved evidence text.
    used_chunks: List[RetrievedChunk] (or objects with `.text`)
    """
    if not answer or not used_chunks:
        return 0.0

    # Tokenize answer (keep words/identifiers, length>=3)
    answer_tokens = set(re.findall(r"\b[a-zA-Z_]{3,}\b", answer.lower()))
    if not answer_tokens:
        return 0.0

    evidence_text = " ".join(getattr(c, "text", "") or "" for c in used_chunks).lower()
    evidence_tokens = set(re.findall(r"\b[a-zA-Z_]{3,}\b", evidence_text))

    overlap = answer_tokens & evidence_tokens
    return len(overlap) / max(1, len(answer_tokens))


def avg_latency_for_type(results: List[Dict[str, Any]], t: str) -> float:
    vals = [r["latency_ms_total"] for r in results if r["pred_type"] == t]
    return sum(vals) / len(vals) if vals else 0.0


def avg_metric_for_type(results: List[Dict[str, Any]], t: str, key: str) -> float:
    vals = [float(r.get(key, 0.0)) for r in results if r["pred_type"] == t]
    return sum(vals) / len(vals) if vals else 0.0


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    if total == 0:
        return {"total": 0}

    correct_type = sum(1 for r in results if r["pred_type"] == r["expected_type"])

    expected_refuse = sum(1 for r in results if r["expected_type"] == "refuse")
    correct_refuse = sum(
        1 for r in results
        if r["expected_type"] == "refuse" and r["pred_type"] == "refuse"
    )

    expected_answer = sum(1 for r in results if r["expected_type"] == "answer")
    correct_answer = sum(
        1 for r in results
        if r["expected_type"] == "answer" and r["pred_type"] == "answer"
    )

    # "effective answer" (guards against refusal-like answers labeled as answer)
    effective_answer = sum(
        1 for r in results
        if r["pred_type"] == "answer"
        and "don't have enough" not in r.get("answer_preview", "").lower()
        and "do not have enough information" not in r.get("answer_preview", "").lower()
    )

    clarify_ood = sum(
        1 for r in results
        if r["expected_type"] == "refuse" and r["pred_type"] == "clarify"
    )

    avg_latency = sum(r["latency_ms_total"] for r in results) / total
    avg_top_score = sum(r["top_score"] for r in results) / total

    # ✅ NEW: groundedness aggregates (overall + by pred type)
    avg_groundedness = sum(float(r.get("groundedness_overlap", 0.0)) for r in results) / total

    return {
        "total": total,
        "accuracy_expected_type": correct_type / total,
        "refusal_correctness": (correct_refuse / expected_refuse if expected_refuse else 0.0),
        "answer_correctness": (correct_answer / expected_answer if expected_answer else 0.0),
        "effective_answer_rate": effective_answer / total,
        "clarify_ood_rate": (clarify_ood / expected_refuse if expected_refuse else 0.0),

        "avg_latency_ms_total": avg_latency,
        "avg_latency_answer": avg_latency_for_type(results, "answer"),
        "avg_latency_refuse": avg_latency_for_type(results, "refuse"),
        "avg_latency_clarify": avg_latency_for_type(results, "clarify"),

        "avg_top_score": avg_top_score,

        # ✅ NEW
        "avg_groundedness": avg_groundedness,
        "avg_groundedness_answer": avg_metric_for_type(results, "answer", "groundedness_overlap"),
        "avg_groundedness_refuse": avg_metric_for_type(results, "refuse", "groundedness_overlap"),
        "avg_groundedness_clarify": avg_metric_for_type(results, "clarify", "groundedness_overlap"),
    }
