from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Literal, Optional, Tuple

from src.config import load_app_config
from src.retrieval.retriever import RetrievedChunk

Decision = Literal["answer", "clarify", "refuse"]
MismatchSubtype = Literal["none", "recoverable", "hard", "not_applicable"]


@dataclass(frozen=True)
class ConfidenceResult:
    decision: Decision
    confidence: float
    top_score: float
    second_score: float
    margin: float
    rationale: str
    used_chunks: List[RetrievedChunk]


class ConfidenceGate:
    """
    Confidence gating turns retrieval scores into a safe decision:

    - refuse: not enough evidence in corpus
    - clarify: some evidence but not strong / ambiguous
    - answer: strong evidence

        Step-1 (honest baseline) scope:
            1) top score thresholds (primary signal)
            2) topic consistency (same doc/module between top hits)
            3) support_count (how many hits exceed high threshold)
            4) module/symbol mismatch between query framing and evidence

        Deliberately out of scope for this gate:
            - OOD / python-general intent detection
            - lexical intent overrides
            - broad ambiguity phrasing rules
    """

    def __init__(self, repo_root: Path, *, config_path: Optional[Path] = None):
        self.repo_root = repo_root
        cfg, self.config_path = load_app_config(repo_root, config_path)

        self.th_high = float(cfg.confidence.threshold_high)
        self.th_low = float(cfg.confidence.threshold_low)
        # margin_min is only applied when top hits disagree in topic
        self.margin_min = float(cfg.confidence.margin_min)

        self.max_chunks = int(cfg.retrieval.top_k)

        if not (self.th_low < self.th_high):
            raise ValueError("confidence.threshold_low must be < confidence.threshold_high")

    @staticmethod
    def _extract_module_hints(query: str) -> set[str]:
        q = query.lower()
        hints: set[str] = set()
        generic_hints = {
            "python",
            "stdlib",
            "standard",
            "library",
            "python standard library",
        }

        for m in re.findall(r"\bin\s+([a-zA-Z_][\w\.]*)", q):
            hints.add(m.strip())
        for m in re.findall(r"\b([a-zA-Z_][\w\.]*)\s+module\b", q):
            hints.add(m.strip())

        return {h for h in hints if h and h not in generic_hints}

    @staticmethod
    def _extract_use_target(query: str) -> Optional[str]:
        q = query.lower()
        m = re.search(r"\buse\s+([a-zA-Z_][\w\.]*)", q)
        if not m:
            return None
        token = m.group(1).strip()
        if token in {"to", "a", "an", "the", "for", "in", "on", "with"}:
            return None
        return token or None

    @staticmethod
    def _evidence_text(hits_sorted: List[RetrievedChunk], top_n: int = 3) -> str:
        parts: List[str] = []
        for h in hits_sorted[:top_n]:
            parts.append((h.text or "").lower())
            if h.heading:
                parts.append(h.heading.lower())
            parts.append((h.module or "").lower())
        return "\n".join(parts)

    @staticmethod
    def _strong_symbol_evidence(symbol: str, evidence_text: str) -> bool:
        escaped = re.escape(symbol)
        strong_symbol_pattern = re.compile(rf"(function::\s+{escaped}\b|\b{escaped}\s*\()")
        return strong_symbol_pattern.search(evidence_text) is not None

    @staticmethod
    def _mismatch_applicable(module_hints: set[str], use_target: Optional[str]) -> bool:
        # Apply mismatch analysis only to explicit module/symbol framing.
        # This avoids over-penalizing conceptual or broad explanatory queries.
        return bool(module_hints) or bool(use_target)

    # Compatibility shims for downstream components that still call these methods.
    # Intent classification is intentionally excluded from gate decisions in Step-1.
    @staticmethod
    def _python_general_out_of_scope_signals(query: str) -> List[str]:
        return []

    @staticmethod
    def _python_general_concept_signals(query: str) -> List[str]:
        return []

    @staticmethod
    def _explicit_out_of_domain_signals(query: str) -> List[str]:
        return []

    @staticmethod
    def _has_plausible_stdlib_anchor(
        query: str,
        hits_sorted: List[RetrievedChunk],
        *,
        module_hints: set[str],
        use_target: Optional[str],
    ) -> bool:
        if module_hints or use_target:
            return True

        q = query.lower()
        top_modules = [h.module.lower() for h in hits_sorted[:3] if h.module]
        for module in top_modules:
            mod_leaf = module.split(".")[-1]
            if re.search(rf"\b{re.escape(module)}\b", q) or re.search(rf"\b{re.escape(mod_leaf)}\b", q):
                return True

        return False

    def _classify_mismatch(
        self,
        query: str,
        hits_sorted: List[RetrievedChunk],
    ) -> Tuple[MismatchSubtype, List[str]]:
        hard_reasons: List[str] = []
        recoverable_reasons: List[str] = []
        if not query.strip() or not hits_sorted:
            return "none", []

        module_hints = self._extract_module_hints(query)
        use_target = self._extract_use_target(query)

        if not self._mismatch_applicable(module_hints, use_target):
            return "not_applicable", [
                "mismatch checks not applicable: query lacks explicit module/symbol intent"
            ]

        top_modules = {h.module.lower() for h in hits_sorted[:3] if h.module}
        evidence_text = self._evidence_text(hits_sorted, top_n=3)

        # Signal 1: explicit module hints conflict with retrieved modules.
        module_conflict = bool(module_hints and not (module_hints & top_modules))
        if module_conflict:
            reason = (
                f"query module hints {sorted(module_hints)} not present in top modules {sorted(top_modules)}"
            )
            if use_target:
                hard_reasons.append(reason)
            else:
                recoverable_reasons.append(reason)

        # Signal 2: explicit 'use <symbol>' target is missing/weak in top evidence.
        if use_target:
            if use_target not in evidence_text:
                reason = f"requested symbol '{use_target}' not found in top evidence text"
                recoverable_reasons.append(reason)
            else:
                if not self._strong_symbol_evidence(use_target, evidence_text):
                    reason = (
                        f"requested symbol '{use_target}' appears only weakly; no callable/function-style evidence found"
                    )
                    recoverable_reasons.append(reason)

        # Signal 3: strong-score but fragmented evidence topics in top chunks.
        strong_top = [h for h in hits_sorted[:3] if float(h.score) >= self.th_low]
        strong_modules = {h.module for h in strong_top if h.module}
        if len(strong_top) >= 3 and len(strong_modules) >= 3:
            recoverable_reasons.append("top evidence spans multiple modules without strong topical consistency")

        if hard_reasons:
            unique = list(dict.fromkeys(hard_reasons + recoverable_reasons))
            return "hard", unique
        if recoverable_reasons:
            unique = list(dict.fromkeys(recoverable_reasons))
            return "recoverable", unique

        return "none", []

    def decide(self, hits: List[RetrievedChunk], *, query: str = "") -> ConfidenceResult:
        if not hits:
            return ConfidenceResult(
                decision="refuse",
                confidence=0.0,
                top_score=0.0,
                second_score=0.0,
                margin=0.0,
                rationale="No retrieved evidence.",
                used_chunks=[],
            )

        # Enforce ordering & cap
        hits_sorted = sorted(hits, key=lambda x: float(x.score), reverse=True)[: self.max_chunks]

        top = hits_sorted[0]
        second = hits_sorted[1] if len(hits_sorted) > 1 else None

        s1 = float(top.score)
        s2 = float(second.score) if second else 0.0
        margin = s1 - s2
        mismatch_subtype, mismatch_reasons = self._classify_mismatch(query, hits_sorted)

        # Rule 1 — refuse if too weak
        if s1 < self.th_low:
            return ConfidenceResult(
                decision="refuse",
                confidence=s1,
                top_score=s1,
                second_score=s2,
                margin=margin,
                rationale=f"Top score {s1:.3f} < low threshold {self.th_low:.3f}.",
                used_chunks=[],
            )

        # Topic consistency: if top-1 and top-2 come from same doc/module,
        # a small margin usually means "adjacent/supporting evidence", not ambiguity.
        same_topic = False
        if second is not None:
            same_topic = (top.doc_id == second.doc_id) or (top.module == second.module)

        # Support count: more than one strong chunk is reinforcing evidence
        support_count = sum(1 for h in hits_sorted if float(h.score) >= self.th_high)

        # Hard mismatch can veto answers independent of score band.
        if mismatch_subtype == "hard":
            return ConfidenceResult(
                decision="refuse",
                confidence=s1,
                top_score=s1,
                second_score=s2,
                margin=margin,
                rationale=(
                    "Strong score but hard mismatch signals detected: "
                    + "; ".join(mismatch_reasons)
                ),
                used_chunks=[],
            )

        # Rule 2 — strong evidence region
        if s1 >= self.th_high:
            if mismatch_subtype == "recoverable":
                return ConfidenceResult(
                    decision="clarify",
                    confidence=s1,
                    top_score=s1,
                    second_score=s2,
                    margin=margin,
                    rationale=(
                        "Strong score but recoverable mismatch/inconsistency signals detected: "
                        + "; ".join(mismatch_reasons)
                    ),
                    used_chunks=hits_sorted,
                )

            # Strong score but conflicting top evidence should clarify.
            if second is not None and (not same_topic) and margin < self.margin_min and support_count < 2:
                return ConfidenceResult(
                    decision="clarify",
                    confidence=s1,
                    top_score=s1,
                    second_score=s2,
                    margin=margin,
                    rationale=(
                        f"Strong top score {s1:.3f} but competing topics with small margin "
                        f"{margin:.3f} < {self.margin_min:.3f}; clarification needed."
                    ),
                    used_chunks=hits_sorted,
                )

            return ConfidenceResult(
                decision="answer",
                confidence=s1,
                top_score=s1,
                second_score=s2,
                margin=margin,
                rationale=(
                    f"Strong evidence: top score {s1:.3f} >= {self.th_high:.3f} "
                    f"and topic support is consistent (same_topic={same_topic}, support_count={support_count})."
                ),
                used_chunks=hits_sorted,
            )

        # Rule 3 — middle zone: some evidence but not strong enough
        if mismatch_subtype == "recoverable":
            return ConfidenceResult(
                decision="clarify",
                confidence=s1,
                top_score=s1,
                second_score=s2,
                margin=margin,
                rationale=(
                    "Middle score band with recoverable mismatch signals: "
                    + "; ".join(mismatch_reasons)
                ),
                used_chunks=hits_sorted,
            )

        return ConfidenceResult(
            decision="clarify",
            confidence=s1,
            top_score=s1,
            second_score=s2,
            margin=margin,
            rationale=(
                f"Top score {s1:.3f} is between thresholds "
                f"{self.th_low:.3f} and {self.th_high:.3f}; clarification recommended."
            ),
            used_chunks=hits_sorted,
        )
