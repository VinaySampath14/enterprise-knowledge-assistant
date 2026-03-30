from __future__ import annotations

import re
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

from src.config import load_app_config
from src.retrieval.retriever import Retriever
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.rag.confidence import ConfidenceGate
from src.rag.generator import Generator
from src.rag.intent import classify_query_intent, should_refuse_upstream
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


def _is_usable_answer_text(answer_text: str) -> bool:
    t = (answer_text or "").strip()
    if len(t) < 40:
        return False
    if re.search(r"[a-zA-Z]", t) is None:
        return False
    low = t.lower()
    # If the entire message is a boilerplate refusal, keep existing override behavior.
    boilerplate = {
        "i don't have enough information in the python standard library documentation to answer that.",
        "i do not have enough information in the python standard library documentation to answer that.",
        "i don't have enough information from the provided documentation.",
        "i do not have enough information from the provided documentation.",
    }
    if low in boilerplate:
        return False
    if low.startswith("i don't have enough information") or low.startswith("i do not have enough information"):
        return False
    return True


def _is_strong_stdlib_coherence(chunks: List[Any]) -> bool:
    if not chunks:
        return False
    top = chunks[:3]
    modules = [c.module for c in top if getattr(c, "module", None)]
    if not modules:
        return False

    counts = Counter(modules)
    same_module = len(counts) == 1
    concentrated = counts.most_common(1)[0][1] >= 2

    stdlib_paths = 0
    for c in top:
        src = (getattr(c, "source_path", "") or "").lower().replace("\\", "/")
        if "python_stdlib/" in src:
            stdlib_paths += 1

    return same_module or (concentrated and stdlib_paths >= 2)


def _extract_explicit_symbol_mentions(query: str) -> set[str]:
    q = (query or "").lower()
    return set(re.findall(r"\b([a-z_][\w]*\.[a-z_][\w]*)\b", q))


def _evidence_blob(chunks: List[Any], top_n: int = 3) -> str:
    parts: List[str] = []
    for c in chunks[:top_n]:
        parts.append((getattr(c, "text", "") or "").lower())
        parts.append((getattr(c, "heading", "") or "").lower())
        parts.append((getattr(c, "module", "") or "").lower())
    return "\n".join(parts)


def _has_grounded_stdlib_signal(
    answer_text: str,
    chunks: List[Any],
    citation_ids: List[int],
    *,
    top_score: float,
    gate: ConfidenceGate,
    module_hints: set[str],
    use_target: str | None,
    symbol_hints: set[str],
) -> bool:
    # Primary signal: explicit citation linkage.
    if citation_ids:
        return True

    # Step-4b narrow fallback: if citations are missing, allow rescue only for
    # strong in-domain anchored queries with coherent evidence.
    if not (module_hints or use_target or symbol_hints):
        return False
    if top_score < (gate.th_high + 0.05):
        return False
    if not _is_strong_stdlib_coherence(chunks):
        return False
    if symbol_hints:
        evidence = _evidence_blob(chunks, top_n=3)
        if not any(sym in evidence for sym in symbol_hints):
            return False
    return True


def _should_block_post_generation_refusal_override(
    *,
    query: str,
    answer_text: str,
    gate_decision: str,
    mismatch_subtype: str,
    used_chunks: List[Any],
    citation_ids: List[int],
    gate: ConfidenceGate,
) -> tuple[bool, List[str]]:
    reasons: List[str] = []

    if gate_decision != "answer":
        return False, reasons
    if mismatch_subtype in {"hard", "recoverable"}:
        return False, reasons
    if not _is_usable_answer_text(answer_text):
        return False, reasons
    if not _is_strong_stdlib_coherence(used_chunks):
        return False, reasons
    module_hints = gate._extract_module_hints(query)  # noqa: SLF001
    use_target = gate._extract_use_target(query)  # noqa: SLF001
    symbol_hints = _extract_explicit_symbol_mentions(query)
    if not _has_grounded_stdlib_signal(
        answer_text,
        used_chunks,
        citation_ids,
        top_score=float(used_chunks[0].score) if used_chunks else 0.0,
        gate=gate,
        module_hints=module_hints,
        use_target=use_target,
        symbol_hints=symbol_hints,
    ):
        return False, reasons

    has_plausible_anchor = gate._has_plausible_stdlib_anchor(  # noqa: SLF001
        query,
        used_chunks,
        module_hints=module_hints,
        use_target=use_target,
    )
    if not has_plausible_anchor:
        return False, reasons

    if gate._explicit_out_of_domain_signals(query):  # noqa: SLF001
        return False, reasons

    if gate._python_general_concept_signals(query):  # noqa: SLF001
        return False, reasons

    if gate._python_general_out_of_scope_signals(query):  # noqa: SLF001
        return False, reasons

    reasons.append("code: PHASE2_POSTGEN_NARROW_RESCUE_KEEP_ANSWER")
    reasons.append("guard: gate_decision=answer")
    reasons.append("guard: mismatch not hard/recoverable")
    reasons.append("guard: strong stdlib coherence")
    reasons.append("guard: grounded evidence signal present (citations or strong anchored coherence with symbol evidence)")
    reasons.append("guard: plausible in-domain stdlib/module/function anchor")
    reasons.append("guard: no python-general out-of-scope intent signal")
    return True, reasons


class RAGPipeline:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.cfg, _ = load_app_config(repo_root)
        self.retriever = Retriever(repo_root)
        self.reranker = CrossEncoderReranker(repo_root)
        self.gate = ConfidenceGate(repo_root)
        self.generator = Generator(repo_root)

    def run(self, query: str, request_id: str | None = None) -> Dict[str, Any]:
        request_id = request_id or str(uuid.uuid4())

        t0 = time.perf_counter()

        intent = classify_query_intent(query)
        if should_refuse_upstream(intent):
            t1 = time.perf_counter()
            return {
                "type": "refuse",
                "answer": "I do not have enough information in the Python standard library documentation to answer that.",
                "confidence": 0.0,
                "sources": [],
                "citations": [],
                "meta": {
                    "top_score": 0.0,
                    "second_score": 0.0,
                    "score_margin": 0.0,
                    "gate_decision": "refuse",
                    "gate_rationale": "Step-3 upstream intent route refused query before retrieval.",
                    "gate_tie_breaker_fired": False,
                    "intent_label": intent.label,
                    "intent_confidence": float(intent.confidence),
                    "intent_rationale": intent.rationale,
                    "intent_signals": intent.signals,
                    "intent_routed_refuse": True,
                    "postgen_refusal_override_triggered": False,
                    "postgen_refusal_override_blocked": False,
                    "postgen_refusal_override_block_reasons": [],
                    "postgen_refusal_override_rescue_code": "",
                    "retrieved_k": 0,
                    "latency_ms_total": (t1 - t0) * 1000,
                    "latency_ms_retrieval": 0.0,
                    "latency_ms_generation": 0.0,
                    "request_id": request_id,
                },
            }

        # --- Retrieval ---
        t_retrieval_start = time.perf_counter()
        retrieval_k = self.reranker.candidate_k if self.reranker.enabled else None
        hits = self.retriever.retrieve(query, top_k=retrieval_k)
        hits = self.reranker.rerank(query, hits, top_k=int(self.cfg.retrieval.top_k))
        t_retrieval_end = time.perf_counter()

        decision = self.gate.decide(hits, query=query)

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

            source_mapping: List[Dict[str, Any]] = []
            citation_ids: List[int] = []
            parsed_citations: List[Dict[str, Any]] = []
            try:
                _, source_mapping = format_retrieved_chunks(decision.used_chunks)
                citation_ids = _extract_citation_ids(answer_text)
                parsed_citations = _build_citations_from_ids(citation_ids, source_mapping)
            except Exception:
                source_mapping = []
                citation_ids = []
                parsed_citations = []

            override_triggered = _is_refusal_text(answer_text)
            override_blocked = False
            override_block_reasons: List[str] = []

            # 🔒 Post-generation refusal override
            if override_triggered:
                mismatch_subtype, _ = self.gate._classify_mismatch(query, decision.used_chunks)  # noqa: SLF001

                override_blocked, override_block_reasons = _should_block_post_generation_refusal_override(
                    query=query,
                    answer_text=answer_text,
                    gate_decision=decision.decision,
                    mismatch_subtype=mismatch_subtype,
                    used_chunks=decision.used_chunks,
                    citation_ids=citation_ids,
                    gate=self.gate,
                )

                if override_blocked:
                    result_type = "answer"
                    citations = parsed_citations
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
                else:
                    result_type = "refuse"
                    sources = []
                    citations = []
            else:
                result_type = "answer"
                citations = parsed_citations

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
                "second_score": float(decision.second_score),
                "score_margin": float(decision.margin),
                "gate_decision": decision.decision,
                "gate_rationale": decision.rationale,
                "gate_tie_breaker_fired": (
                    "PHASE1B_INTENT_TIEBREAKER_REFUSE" in (decision.rationale or "")
                ),
                "intent_label": intent.label,
                "intent_confidence": float(intent.confidence),
                "intent_rationale": intent.rationale,
                "intent_signals": intent.signals,
                "intent_routed_refuse": False,
                "postgen_refusal_override_triggered": (
                    bool(override_triggered)
                    if decision.decision == "answer"
                    else False
                ),
                "postgen_refusal_override_blocked": (
                    bool(override_blocked)
                    if decision.decision == "answer"
                    else False
                ),
                "postgen_refusal_override_block_reasons": (
                    override_block_reasons
                    if decision.decision == "answer"
                    else []
                ),
                "postgen_refusal_override_rescue_code": (
                    "PHASE2_POSTGEN_NARROW_RESCUE_KEEP_ANSWER"
                    if decision.decision == "answer" and bool(override_blocked)
                    else ""
                ),
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
