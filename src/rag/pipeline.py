from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List

from src.retrieval.retriever import Retriever
from src.rag.confidence import ConfidenceGate
from src.rag.generator import Generator
from src.rag.prompt import format_retrieved_chunks


def _is_refusal_text(text: str) -> bool:
    if not text:
        return True

    t = text.lower()

    patterns = [
        "i don't have enough information",
        "not in the documentation",
        "cannot find",
        "not available in the provided documentation",
        "i do not have enough information"
    ]

    return any(p in t for p in patterns)


def _extract_citation_ids(answer_text: str) -> List[int]:
    """
    Extract citation ids from patterns like [1], [1][2], [1, 2].
    Returns unique ids in first-appearance order.
    """
    if not answer_text:
        return []

    seen: set[int] = set()
    ordered: List[int] = []

    # Matches bracketed groups containing digits and optional comma-separated digits.
    # Examples captured: [1], [1,2], [1, 2], and each bracket in [1][2].
    for match in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", answer_text):
        for token in match.split(","):
            token = token.strip()
            if not token:
                continue
            cid = int(token)
            if cid not in seen:
                seen.add(cid)
                ordered.append(cid)

    return ordered


def _build_citations_from_ids(
    citation_ids: List[int],
    source_mapping: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_id = {int(s["id"]): s for s in source_mapping}
    return [by_id[cid] for cid in citation_ids if cid in by_id]


class RAGPipeline:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.retriever = Retriever(repo_root)
        self.gate = ConfidenceGate(repo_root)
        self.generator = Generator(repo_root)

    def run(self, query: str, request_id: str | None = None) -> Dict[str, Any]:
        request_id = request_id or str(uuid.uuid4())

        t0 = time.perf_counter()

        # --- Retrieval ---
        t_retrieval_start = time.perf_counter()
        hits = self.retriever.retrieve(query)
        t_retrieval_end = time.perf_counter()

        decision = self.gate.decide(hits)

        sources: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        answer_text = ""
        result_type = decision.decision

        # --- If refuse ---
        if decision.decision == "refuse":
            result_type = "refuse"
            answer_text = (
                "I do not have enough information in the Python standard library documentation to answer that."
            )

        # --- If clarify ---
        elif decision.decision == "clarify":
            result_type = "clarify"
            answer_text = (
                "Could you clarify your question (e.g., which module/function you mean) "
                "so I can look it up in the documentation?"
            )

        # --- If answer ---
        else:
            t_gen_start = time.perf_counter()
            answer_text = self.generator.generate(query, decision.used_chunks)
            t_gen_end = time.perf_counter()

            # 🔒 Post-generation refusal override
            if _is_refusal_text(answer_text):
                result_type = "refuse"
                sources = []
                citations = []
            else:
                result_type = "answer"
                try:
                    _, source_mapping = format_retrieved_chunks(decision.used_chunks)
                    citation_ids = _extract_citation_ids(answer_text)
                    citations = _build_citations_from_ids(citation_ids, source_mapping)
                except Exception:
                    citations = []

                for h in decision.used_chunks:
                    sources.append(
                        {
                            "chunk_id": h.chunk_id,
                            "doc_id": h.doc_id,
                            "module": h.module,
                            "score": float(h.score),
                            "source_path": h.source_path,
                        }
                    )

        t1 = time.perf_counter()

        return {
            "type": result_type,
            "answer": answer_text,
            "confidence": float(decision.confidence),
            "sources": sources,
            "citations": citations,
            "meta": {
                "top_score": float(decision.top_score),
                "retrieved_k": len(hits),
                "latency_ms_total": (t1 - t0) * 1000,
                "latency_ms_retrieval": (t_retrieval_end - t_retrieval_start) * 1000,
                "latency_ms_generation": (
                    (t_gen_end - t_gen_start) * 1000
                    if decision.decision == "answer"
                    else 0.0
                ),
                "request_id": request_id,
            },
        }
