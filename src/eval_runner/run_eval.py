from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
from collections import Counter
import re

from src.utils.jsonl import iter_jsonl, write_jsonl
from src.rag.pipeline import RAGPipeline


STOPWORDS = {
    "the", "and", "for", "with", "you", "can", "use", "this", "that", "from",
    "are", "was", "were", "will", "into", "then", "than", "also", "how", "what",
    "your", "their", "they", "them", "its", "it's", "not", "have", "has", "had",
    "to", "of", "in", "on", "at", "as", "by", "an", "a", "it", "is", "be", "or",
    "if", "do", "does", "did", "we", "i"
}


def compute_groundedness(answer: str, used_chunks: list) -> float:
    """
    Lexical groundedness proxy:
    fraction of meaningful answer tokens (>=3 chars, non-stopwords) that appear in evidence.
    used_chunks: List[RetrievedChunk] (objects with `.text`)
    """
    if not answer or not used_chunks:
        return 0.0

    answer_tokens = set(re.findall(r"\b[a-zA-Z_]{3,}\b", answer.lower()))
    answer_tokens = {t for t in answer_tokens if t not in STOPWORDS}
    if not answer_tokens:
        return 0.0

    evidence_text = " ".join((c.text or "") for c in used_chunks).lower()
    evidence_tokens = set(re.findall(r"\b[a-zA-Z_]{3,}\b", evidence_text))
    evidence_tokens = {t for t in evidence_tokens if t not in STOPWORDS}

    overlap = answer_tokens & evidence_tokens
    return len(overlap) / max(1, len(answer_tokens))


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pipeline = RAGPipeline(repo_root)

    q_path = repo_root / "eval" / "questions.jsonl"
    out_results = repo_root / "eval" / "results.jsonl"
    out_report_json = repo_root / "eval" / "report.json"
    out_report_md = repo_root / "eval" / "report.md"

    results: List[Dict[str, Any]] = []

    for q in iter_jsonl(q_path):
        qid = q["id"]
        query = q["query"]
        expected = q["expected_type"]

        # Run full pipeline
        out = pipeline.run(query)

        # Re-run retrieval+gate to access used_chunks for groundedness
        hits = pipeline.retriever.retrieve(query)
        decision = pipeline.gate.decide(hits)

        grounded = compute_groundedness(
            out["answer"],
            decision.used_chunks if decision.decision == "answer" else []
        )

        rec = {
            "id": qid,
            "query": query,
            "expected_type": expected,
            "pred_type": out["type"],
            "confidence": out["confidence"],
            "top_score": out["meta"]["top_score"],
            "latency_ms_total": out["meta"]["latency_ms_total"],
            "latency_ms_retrieval": out["meta"]["latency_ms_retrieval"],
            "latency_ms_generation": out["meta"]["latency_ms_generation"],
            "num_sources": len(out.get("sources", [])),
            "request_id": out["meta"].get("request_id"),
            "answer_preview": out["answer"][:200],
            "tags": q.get("tags", []),
            "groundedness_overlap": grounded,
        }
        results.append(rec)

    # Write results.jsonl
    write_jsonl(out_results, results, append=False)

    # ---------------- METRICS ----------------
    total = len(results)
    by_expected = Counter(r["expected_type"] for r in results)
    by_pred = Counter(r["pred_type"] for r in results)

    correct = sum(1 for r in results if r["expected_type"] == r["pred_type"])
    acc = correct / total if total else 0.0

    refuse_total = sum(1 for r in results if r["expected_type"] == "refuse")
    refuse_correct = sum(
        1 for r in results
        if r["expected_type"] == "refuse" and r["pred_type"] == "refuse"
    )
    refuse_rate = refuse_correct / refuse_total if refuse_total else 0.0

    answer_total = sum(1 for r in results if r["expected_type"] == "answer")
    answer_correct = sum(
        1 for r in results
        if r["expected_type"] == "answer" and r["pred_type"] == "answer"
    )
    answer_rate = answer_correct / answer_total if answer_total else 0.0

    avg_latency = sum(r["latency_ms_total"] for r in results) / total if total else 0.0
    avg_top_score = sum(r["top_score"] for r in results) / total if total else 0.0

    # Groundedness summary (overall)
    avg_groundedness = sum(r["groundedness_overlap"] for r in results) / total if total else 0.0

    # Groundedness by predicted type (meaningful!)
    answer_rows = [r for r in results if r["pred_type"] == "answer"]
    refuse_rows = [r for r in results if r["pred_type"] == "refuse"]
    clarify_rows = [r for r in results if r["pred_type"] == "clarify"]

    avg_groundedness_answer = (
        sum(r["groundedness_overlap"] for r in answer_rows) / len(answer_rows)
        if answer_rows else 0.0
    )
    avg_groundedness_refuse = (
        sum(r["groundedness_overlap"] for r in refuse_rows) / len(refuse_rows)
        if refuse_rows else 0.0
    )
    avg_groundedness_clarify = (
        sum(r["groundedness_overlap"] for r in clarify_rows) / len(clarify_rows)
        if clarify_rows else 0.0
    )

    report = {
        "total": total,
        "accuracy_expected_type": acc,
        "refusal_correctness": refuse_rate,
        "answer_correctness": answer_rate,
        "pred_type_counts": dict(by_pred),
        "expected_type_counts": dict(by_expected),
        "avg_latency_ms_total": avg_latency,
        "avg_top_score": avg_top_score,
        "avg_groundedness": avg_groundedness,
        "avg_groundedness_answer": avg_groundedness_answer,
        "avg_groundedness_refuse": avg_groundedness_refuse,
        "avg_groundedness_clarify": avg_groundedness_clarify,
    }

    out_report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Markdown report
    md = []
    md.append("# Evaluation Report\n")
    md.append(f"- Total questions: **{total}**\n")
    md.append(f"- Accuracy (expected_type): **{acc:.3f}**\n")
    md.append(f"- Refusal correctness: **{refuse_rate:.3f}**\n")
    md.append(f"- Answer correctness: **{answer_rate:.3f}**\n")
    md.append(f"- Avg latency total (ms): **{avg_latency:.1f}**\n")
    md.append(f"- Avg top_score: **{avg_top_score:.3f}**\n")
    md.append(f"- Avg groundedness (overall): **{avg_groundedness:.3f}**\n")
    md.append(f"- Avg groundedness (answer only): **{avg_groundedness_answer:.3f}**\n")

    md.append("\n## Prediction counts\n")
    for k, v in by_pred.items():
        md.append(f"- {k}: {v}\n")

    out_report_md.write_text("".join(md), encoding="utf-8")

    print(f"[OK] Wrote results: {out_results}")
    print(f"[OK] Wrote report: {out_report_json}")
    print(f"[OK] Wrote report: {out_report_md}")


if __name__ == "__main__":
    main()
