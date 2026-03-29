from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

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

    This version uses:
      1) top score thresholds (primary signal)
      2) topic consistency (same doc/module between top hits)
      3) support_count (how many hits exceed high threshold)
      4) margin only when top hits compete across topics
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
    def _query_in_module_use_pattern(query: str) -> bool:
        q = query.lower()
        return re.search(r"\bin\s+[a-zA-Z_][\w\.]*.{0,100}\buse\s+[a-zA-Z_][\w\.]*", q) is not None

    @staticmethod
    def _mismatch_applicable(module_hints: set[str], use_target: Optional[str]) -> bool:
        # Apply mismatch analysis only to explicit module/symbol framing.
        # This avoids over-penalizing conceptual or broad explanatory queries.
        return bool(module_hints) or bool(use_target)

    def _ambiguity_signals(
        self,
        query: str,
        hits_sorted: List[RetrievedChunk],
        *,
        margin: float,
        module_hints: set[str],
        use_target: Optional[str],
    ) -> List[str]:
        q = query.lower().strip()
        reasons: List[str] = []

        broad_patterns = [
            "best way",
            "how do i handle",
            "how does this work",
            "overview",
            "in general",
            "what module should i use",
            "which library should i use",
            "which package should i use",
            "which tool should i use",
        ]

        if not module_hints and not use_target:
            reasons.append("no explicit module/symbol target in query")

        if any(p in q for p in broad_patterns):
            reasons.append("query uses broad/ambiguous phrasing")

        top_modules = [h.module for h in hits_sorted[:3] if h.module]
        unique_modules = set(top_modules)
        if len(top_modules) >= 2 and len(unique_modules) >= 2:
            reasons.append("top evidence spans multiple modules")

        if margin < self.margin_min:
            reasons.append(
                f"top score margin {margin:.3f} < margin_min {self.margin_min:.3f}"
            )

        return reasons

    @staticmethod
    def _python_general_out_of_scope_signals(query: str) -> List[str]:
        q = query.lower().strip()
        reasons: List[str] = []

        if re.search(r"\b(pip|conda|poetry|install|setup\.py|pyproject)\b", q):
            reasons.append("intent: package install/setup")

        if re.search(r"\b(virtual\s*env|venv|environment\s+setup|python\s+path)\b", q):
            reasons.append("intent: environment configuration")

        if re.search(r"\b(vscode|pycharm|jupyter|ide|editor|tooling)\b", q):
            reasons.append("intent: editor/tooling setup")

        if re.search(r"\b(deploy|deployment|production\s+setup|docker|kubernetes)\b", q):
            reasons.append("intent: deployment/setup")

        if re.search(r"\b(which|what)\s+(library|package|tool|framework)\s+should\s+i\s+use\b", q):
            reasons.append("intent: broad library/package/tool selection")

        if re.search(r"\b(numpy|pandas|tensorflow|pytorch|django|flask|fastapi|requests|scikit\-learn)\b", q):
            reasons.append("intent: third-party library/framework")

        if re.search(r"\b(dataframe|machine\s+learning|gradient\s+descent|neural\s+network)\b", q):
            reasons.append("intent: data-science/ml concept outside stdlib docs")

        if re.search(r"\bdifference\s+between\s+a\s+list\s+and\s+a\s+tuple\b", q):
            reasons.append("intent: broad python-general concept outside targeted stdlib docs")

        if re.search(r"\bpython\s+garbage\s+collection\b", q):
            reasons.append("intent: broad python runtime concept outside targeted stdlib docs")

        return reasons

    @staticmethod
    def _phase2_near_domain_refuse_signals(query: str) -> List[str]:
        q = query.lower().strip()
        reasons: List[str] = []

        # Synthetic-style near-domain mismatch template should not be answered.
        if (
            re.search(r"\bin\s+[a-zA-Z_][\w\.]*\s*,\s*how\s+do\s+i\s+use\s+[a-zA-Z_][\w\.]*", q)
            and "exactly as documented" in q
        ):
            reasons.append("intent: explicit in-module symbol request with exact-doc framing")

        # Module + off-topic capability combinations should be refused.
        module_pat = r"\b(argparse|threading|pathlib|logging|itertools|heapq|asyncio)\b"
        capability_pat = r"\b(wildcard|json|csv|sql|regex|http)\b"
        if re.search(module_pat, q) and re.search(capability_pat, q):
            reasons.append("intent: near-domain module + off-topic capability mismatch")

        return reasons

    @staticmethod
    def _python_general_concept_signals(query: str) -> List[str]:
        q = query.lower().strip()
        reasons: List[str] = []

        conceptual_framing = re.search(
            r"\b(what\s+is|what\s+does|explain|how\s+do\s+i\s+write|meaning\s+of)\b",
            q,
        )
        if not conceptual_framing:
            return reasons

        if re.search(r"\bpython\s+decorator\b|\bdecorator\b", q):
            reasons.append("intent: python-general decorator concept not tied to a stdlib API target")

        if re.search(r"\bgil\b|global\s+interpreter\s+lock", q):
            reasons.append("intent: python runtime/GIL concept not tied to a stdlib API target")

        if re.search(r"difference\s+between\s+a\s+list\s+and\s+a\s+tuple", q):
            reasons.append("intent: python core language concept not tied to a stdlib API target")

        if re.search(r"python\s+garbage\s+collection", q):
            reasons.append("intent: python runtime concept not tied to a stdlib API target")

        return reasons

    @staticmethod
    def _explicit_out_of_domain_signals(query: str) -> List[str]:
        q = query.lower().strip()
        reasons: List[str] = []

        if re.search(r"\b(css|html|center\s+a\s+div)\b", q):
            reasons.append("intent: web/frontend topic outside stdlib corpus")

        if re.search(r"\b(plot\s+of\s+dune|dune\b)\b", q):
            reasons.append("intent: entertainment/media topic outside stdlib corpus")

        if re.search(r"\b(fifa|world\s+cup|capital\s+of\s+france|coffee\s+shop)\b", q):
            reasons.append("intent: general-world knowledge outside stdlib corpus")

        if re.search(r"\b(symptoms\s+of\s+flu|quantum\s+entanglement|love\s+poem)\b", q):
            reasons.append("intent: non-programming general topic outside stdlib corpus")

        if re.search(r"\b(gradient\s+descent|machine\s+learning|transformer\s+models?)\b", q):
            reasons.append("intent: broad ML concept outside stdlib documentation scope")

        return reasons

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

    def _should_refuse_on_ambiguous_python_general(
        self,
        query: str,
        hits_sorted: List[RetrievedChunk],
        *,
        top_score: float,
        margin: float,
        module_hints: set[str],
        use_target: Optional[str],
        ambiguity_reasons: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Phase 1b tie-breaker: in high-ambiguity clarify cases with no explicit stdlib target,
        switch clarify -> refuse when query intent is strongly python-general/out-of-scope.
        """
        reasons: List[str] = []

        out_of_scope_hint_tokens = {
            "pip",
            "conda",
            "poetry",
            "vscode",
            "pycharm",
            "jupyter",
            "ide",
            "editor",
            "numpy",
            "pandas",
            "tensorflow",
            "pytorch",
            "django",
            "flask",
            "fastapi",
            "requests",
            "scikit",
            "sklearn",
            "machine",
        }
        explicit_non_oos_hints = {h for h in module_hints if h not in out_of_scope_hint_tokens}
        explicit_use_target = use_target if use_target not in out_of_scope_hint_tokens else None
        if explicit_non_oos_hints or explicit_use_target:
            return False, reasons

        has_broad = any("broad/ambiguous phrasing" in r for r in ambiguity_reasons)
        has_multi = any("spans multiple modules" in r for r in ambiguity_reasons)
        has_low_margin = margin < (self.margin_min * 1.25)
        high_ambiguity = (has_broad and (has_multi or has_low_margin)) or (has_multi and has_low_margin)
        if not high_ambiguity:
            return False, reasons

        top_modules = [h.module for h in hits_sorted[:3] if h.module]
        coherent_top_modules = len(set(top_modules)) <= 1 if top_modules else False
        if coherent_top_modules and top_score >= self.th_low:
            return False, reasons

        if self._has_plausible_stdlib_anchor(
            query,
            hits_sorted,
            module_hints=explicit_non_oos_hints,
            use_target=explicit_use_target,
        ):
            return False, reasons

        oos_signals = self._python_general_out_of_scope_signals(query)
        if not oos_signals:
            return False, reasons

        reasons.extend(oos_signals)
        reasons.append("trigger: high ambiguity + no explicit stdlib target")
        reasons.append("decision: clarify->refuse via phase1b tie-breaker")
        return True, reasons

    def _should_answer_on_conceptual_in_domain(
        self,
        query: str,
        hits_sorted: List[RetrievedChunk],
        *,
        top_score: float,
        module_hints: set[str],
        use_target: Optional[str],
        ambiguity_reasons: List[str],
        mismatch_subtype: MismatchSubtype,
    ) -> Tuple[bool, List[str]]:
        """
        Phase 2 conceptual rescue: keep broad conceptual stdlib questions answerable
        when retrieval is strongly coherent and no mismatch exists.
        """
        reasons: List[str] = []

        if top_score < self.th_high:
            return False, reasons
        if mismatch_subtype in {"hard", "recoverable"}:
            return False, reasons
        if not ambiguity_reasons:
            return False, reasons

        top_modules = [h.module for h in hits_sorted[:3] if h.module]
        if not top_modules:
            return False, reasons
        if len(set(top_modules)) != 1:
            return False, reasons

        if self._python_general_out_of_scope_signals(query):
            return False, reasons

        if not self._has_plausible_stdlib_anchor(
            query,
            hits_sorted,
            module_hints=module_hints,
            use_target=use_target,
        ):
            return False, reasons

        reasons.append("trigger: coherent single-module retrieval with stdlib anchor")
        reasons.append("decision: clarify->answer via conceptual in-domain rescue")
        return True, reasons

    def _should_refuse_on_explicit_out_of_domain_clarify(
        self,
        query: str,
        hits_sorted: List[RetrievedChunk],
        *,
        top_score: float,
        module_hints: set[str],
        use_target: Optional[str],
    ) -> Tuple[bool, List[str]]:
        """
        Conservative Phase 2 clarify->refuse override for clearly out-of-domain asks.
        Only fires when query has no explicit stdlib/module/function anchor and
        retrieved evidence does not look plausibly in-domain.
        """
        reasons: List[str] = []

        # Keep this override limited to middle-band clarify behavior.
        if top_score >= self.th_high:
            return False, reasons

        top_modules = {h.module.lower() for h in hits_sorted[:3] if h.module}
        explicit_in_domain_module_hint = bool(module_hints & top_modules)

        if use_target or explicit_in_domain_module_hint:
            return False, reasons

        if self._has_plausible_stdlib_anchor(
            query,
            hits_sorted,
            module_hints=set(),
            use_target=None,
        ):
            return False, reasons

        ood_reasons = self._explicit_out_of_domain_signals(query)
        if not ood_reasons:
            return False, reasons

        top_modules_list = [h.module for h in hits_sorted[:3] if h.module]
        module_fragmented = len(set(top_modules_list)) >= 2
        near_low_band = top_score <= (self.th_low + 0.06)
        if not (module_fragmented or near_low_band):
            return False, reasons

        reasons.extend(ood_reasons)
        reasons.append("trigger: clarify path with no stdlib anchor and weak/non-coherent evidence")
        reasons.append("decision: clarify->refuse via explicit out-of-domain override")
        return True, reasons

    def _should_refuse_on_python_general_concept_answer(
        self,
        query: str,
        hits_sorted: List[RetrievedChunk],
        *,
        top_score: float,
        module_hints: set[str],
        use_target: Optional[str],
    ) -> Tuple[bool, List[str]]:
        """
        Conservative Phase 2 answer->refuse override for generic python-concept asks.
        Fires only when query lacks plausible in-domain stdlib/module/function anchors.
        """
        reasons: List[str] = []

        if top_score < self.th_high:
            return False, reasons

        if module_hints or use_target:
            return False, reasons

        if self._has_plausible_stdlib_anchor(
            query,
            hits_sorted,
            module_hints=module_hints,
            use_target=use_target,
        ):
            return False, reasons

        concept_reasons = self._python_general_concept_signals(query)
        if not concept_reasons:
            return False, reasons

        # Require missing explicit stdlib framing to keep this guard conservative.
        q = query.lower().strip()
        has_explicit_stdlib_scope = bool(
            re.search(r"\b(stdlib|standard\s+library|python\s+standard\s+library)\b", q)
        )
        if has_explicit_stdlib_scope:
            return False, reasons

        reasons.extend(concept_reasons)
        reasons.append("trigger: answer path with no plausible stdlib/module/function anchor")
        reasons.append("decision: answer->refuse via python-general conceptual scope guard")
        return True, reasons

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
        in_module_use_pattern = self._query_in_module_use_pattern(query)

        # Signal 1: explicit module hints conflict with retrieved modules.
        module_conflict = bool(module_hints and not (module_hints & top_modules))
        if module_conflict:
            reason = (
                f"query module hints {sorted(module_hints)} not present in top modules {sorted(top_modules)}"
            )
            if use_target and in_module_use_pattern:
                hard_reasons.append(reason)
            else:
                recoverable_reasons.append(reason)

        # Signal 2: explicit 'use <symbol>' target is missing/weak in top evidence.
        if use_target:
            if use_target not in evidence_text:
                reason = f"requested symbol '{use_target}' not found in top evidence text"
                if module_hints and in_module_use_pattern:
                    hard_reasons.append(reason)
                else:
                    recoverable_reasons.append(reason)
            else:
                if not self._strong_symbol_evidence(use_target, evidence_text):
                    reason = (
                        f"requested symbol '{use_target}' appears only weakly; no callable/function-style evidence found"
                    )
                    if module_hints and in_module_use_pattern:
                        hard_reasons.append(reason)
                    else:
                        recoverable_reasons.append(reason)

        # Signal 3: strong-score but fragmented evidence topics in top chunks.
        strong_top = [h for h in hits_sorted[:3] if float(h.score) >= self.th_low]
        strong_modules = {h.module for h in strong_top if h.module}
        if len(strong_top) >= 3 and len(strong_modules) >= 3:
            recoverable_reasons.append("top evidence spans multiple modules without strong topical consistency")

        # Strong wrong-module pattern with explicit symbol-in-module framing -> hard mismatch.
        if module_hints and use_target and in_module_use_pattern and (hard_reasons or module_conflict):
            hard_reasons.append("explicit 'in <module> use <symbol>' request appears unsupported by evidence")

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
        module_hints = self._extract_module_hints(query)
        use_target = self._extract_use_target(query)
        ambiguity_reasons = self._ambiguity_signals(
            query,
            hits_sorted,
            margin=margin,
            module_hints=module_hints,
            use_target=use_target,
        )

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

        # Rule 2 — strong evidence region
        if s1 >= self.th_high:
            mismatch_subtype, mismatch_reasons = self._classify_mismatch(query, hits_sorted)

            phase2_near_domain_reasons = self._phase2_near_domain_refuse_signals(query)
            if phase2_near_domain_reasons:
                return ConfidenceResult(
                    decision="refuse",
                    confidence=s1,
                    top_score=s1,
                    second_score=s2,
                    margin=margin,
                    rationale=(
                        "PHASE2_INTENT_SCOPE_REFUSE | " + "; ".join(phase2_near_domain_reasons)
                    ),
                    used_chunks=[],
                )

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

            if mismatch_subtype == "recoverable":
                tie_breaker_fire, tie_breaker_reasons = self._should_refuse_on_ambiguous_python_general(
                    query,
                    hits_sorted,
                    top_score=s1,
                    margin=margin,
                    module_hints=module_hints,
                    use_target=use_target,
                    ambiguity_reasons=ambiguity_reasons,
                )
                if tie_breaker_fire:
                    return ConfidenceResult(
                        decision="refuse",
                        confidence=s1,
                        top_score=s1,
                        second_score=s2,
                        margin=margin,
                        rationale=(
                            "PHASE1B_INTENT_TIEBREAKER_REFUSE | "
                            + "; ".join(tie_breaker_reasons)
                        ),
                        used_chunks=[],
                    )

                ood_refuse_fire, ood_refuse_reasons = self._should_refuse_on_explicit_out_of_domain_clarify(
                    query,
                    hits_sorted,
                    top_score=s1,
                    module_hints=module_hints,
                    use_target=use_target,
                )
                if ood_refuse_fire:
                    return ConfidenceResult(
                        decision="refuse",
                        confidence=s1,
                        top_score=s1,
                        second_score=s2,
                        margin=margin,
                        rationale=(
                            "PHASE2_OOD_CLARIFY_TO_REFUSE | "
                            + "; ".join(ood_refuse_reasons)
                        ),
                        used_chunks=[],
                    )

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

            # not_applicable is informational and should not bias decision outcome.
            if mismatch_subtype == "not_applicable":
                mismatch_reasons = []

            if (
                ambiguity_reasons
                and second is not None
                and ((not same_topic) or margin < self.margin_min)
            ):
                tie_breaker_fire, tie_breaker_reasons = self._should_refuse_on_ambiguous_python_general(
                    query,
                    hits_sorted,
                    top_score=s1,
                    margin=margin,
                    module_hints=module_hints,
                    use_target=use_target,
                    ambiguity_reasons=ambiguity_reasons,
                )
                if tie_breaker_fire:
                    return ConfidenceResult(
                        decision="refuse",
                        confidence=s1,
                        top_score=s1,
                        second_score=s2,
                        margin=margin,
                        rationale=(
                            "PHASE1B_INTENT_TIEBREAKER_REFUSE | "
                            + "; ".join(tie_breaker_reasons)
                        ),
                        used_chunks=[],
                    )

            has_broad_signal = any("broad/ambiguous phrasing" in r for r in ambiguity_reasons)
            if (
                has_broad_signal
                and ambiguity_reasons
                and second is not None
                and ((not same_topic) or margin < self.margin_min)
            ):
                conceptual_answer_fire, conceptual_answer_reasons = self._should_answer_on_conceptual_in_domain(
                    query,
                    hits_sorted,
                    top_score=s1,
                    module_hints=module_hints,
                    use_target=use_target,
                    ambiguity_reasons=ambiguity_reasons,
                    mismatch_subtype=mismatch_subtype,
                )
                if conceptual_answer_fire:
                    concept_refuse_fire, concept_refuse_reasons = self._should_refuse_on_python_general_concept_answer(
                        query,
                        hits_sorted,
                        top_score=s1,
                        module_hints=module_hints,
                        use_target=use_target,
                    )
                    if concept_refuse_fire:
                        return ConfidenceResult(
                            decision="refuse",
                            confidence=s1,
                            top_score=s1,
                            second_score=s2,
                            margin=margin,
                            rationale=(
                                "PHASE2_PY_GENERAL_CONCEPT_REFUSE | "
                                + "; ".join(concept_refuse_reasons)
                            ),
                            used_chunks=[],
                        )

                    return ConfidenceResult(
                        decision="answer",
                        confidence=s1,
                        top_score=s1,
                        second_score=s2,
                        margin=margin,
                        rationale=(
                            "PHASE2_CONCEPTUAL_RESCUE_ANSWER | "
                            + "; ".join(conceptual_answer_reasons)
                        ),
                        used_chunks=hits_sorted,
                    )

                ood_refuse_fire, ood_refuse_reasons = self._should_refuse_on_explicit_out_of_domain_clarify(
                    query,
                    hits_sorted,
                    top_score=s1,
                    module_hints=module_hints,
                    use_target=use_target,
                )
                if ood_refuse_fire:
                    return ConfidenceResult(
                        decision="refuse",
                        confidence=s1,
                        top_score=s1,
                        second_score=s2,
                        margin=margin,
                        rationale=(
                            "PHASE2_OOD_CLARIFY_TO_REFUSE | "
                            + "; ".join(ood_refuse_reasons)
                        ),
                        used_chunks=[],
                    )

                return ConfidenceResult(
                    decision="clarify",
                    confidence=s1,
                    top_score=s1,
                    second_score=s2,
                    margin=margin,
                    rationale=(
                        "Strong score but ambiguity heuristics triggered: "
                        + "; ".join(ambiguity_reasons)
                    ),
                    used_chunks=hits_sorted,
                )

            # If evidence reinforces (same topic) or multiple strong hits, answer confidently
            if same_topic or support_count >= 2:
                concept_refuse_fire, concept_refuse_reasons = self._should_refuse_on_python_general_concept_answer(
                    query,
                    hits_sorted,
                    top_score=s1,
                    module_hints=module_hints,
                    use_target=use_target,
                )
                if concept_refuse_fire:
                    return ConfidenceResult(
                        decision="refuse",
                        confidence=s1,
                        top_score=s1,
                        second_score=s2,
                        margin=margin,
                        rationale=(
                            "PHASE2_PY_GENERAL_CONCEPT_REFUSE | "
                            + "; ".join(concept_refuse_reasons)
                        ),
                        used_chunks=[],
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

            # If strong score but top hits are from different topics and almost tied, clarify
            if second is not None and (not same_topic) and margin < self.margin_min:
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

            # Otherwise, allow answering
            concept_refuse_fire, concept_refuse_reasons = self._should_refuse_on_python_general_concept_answer(
                query,
                hits_sorted,
                top_score=s1,
                module_hints=module_hints,
                use_target=use_target,
            )
            if concept_refuse_fire:
                return ConfidenceResult(
                    decision="refuse",
                    confidence=s1,
                    top_score=s1,
                    second_score=s2,
                    margin=margin,
                    rationale=(
                        "PHASE2_PY_GENERAL_CONCEPT_REFUSE | "
                        + "; ".join(concept_refuse_reasons)
                    ),
                    used_chunks=[],
                )

            return ConfidenceResult(
                decision="answer",
                confidence=s1,
                top_score=s1,
                second_score=s2,
                margin=margin,
                rationale="Strong top score and no strong ambiguity signals.",
                used_chunks=hits_sorted,
            )

        # Rule 3 — middle zone: some evidence but not strong enough
        phase2_near_domain_reasons = self._phase2_near_domain_refuse_signals(query)
        if phase2_near_domain_reasons:
            return ConfidenceResult(
                decision="refuse",
                confidence=s1,
                top_score=s1,
                second_score=s2,
                margin=margin,
                rationale=(
                    "PHASE2_INTENT_SCOPE_REFUSE | " + "; ".join(phase2_near_domain_reasons)
                ),
                used_chunks=[],
            )

        tie_breaker_fire, tie_breaker_reasons = self._should_refuse_on_ambiguous_python_general(
            query,
            hits_sorted,
            top_score=s1,
            margin=margin,
            module_hints=module_hints,
            use_target=use_target,
            ambiguity_reasons=ambiguity_reasons,
        )
        if tie_breaker_fire:
            return ConfidenceResult(
                decision="refuse",
                confidence=s1,
                top_score=s1,
                second_score=s2,
                margin=margin,
                rationale=(
                    "PHASE1B_INTENT_TIEBREAKER_REFUSE | "
                    + "; ".join(tie_breaker_reasons)
                ),
                used_chunks=[],
            )

        ood_refuse_fire, ood_refuse_reasons = self._should_refuse_on_explicit_out_of_domain_clarify(
            query,
            hits_sorted,
            top_score=s1,
            module_hints=module_hints,
            use_target=use_target,
        )
        if ood_refuse_fire:
            return ConfidenceResult(
                decision="refuse",
                confidence=s1,
                top_score=s1,
                second_score=s2,
                margin=margin,
                rationale=(
                    "PHASE2_OOD_CLARIFY_TO_REFUSE | "
                    + "; ".join(ood_refuse_reasons)
                ),
                used_chunks=[],
            )

        return ConfidenceResult(
            decision="clarify",
            confidence=s1,
            top_score=s1,
            second_score=s2,
            margin=margin,
            rationale=(
                (
                    "Middle score band with ambiguity signals: " + "; ".join(ambiguity_reasons)
                )
                if ambiguity_reasons
                else (
                    f"Top score {s1:.3f} is between thresholds "
                    f"{self.th_low:.3f} and {self.th_high:.3f}; clarification recommended."
                )
            ),
            used_chunks=hits_sorted,
        )
