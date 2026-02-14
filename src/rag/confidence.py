from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml

from src.retrieval.retriever import RetrievedChunk

Decision = Literal["answer", "clarify", "refuse"]


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
        self.config_path = config_path or (repo_root / "config.yaml")

        with self.config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        conf = cfg.get("confidence", {})
        self.th_high = float(conf.get("threshold_high", 0.40))
        self.th_low = float(conf.get("threshold_low", 0.25))
        # margin_min is only applied when top hits disagree in topic
        self.margin_min = float(conf.get("margin_min", 0.03))

        self.max_chunks = int(cfg.get("retrieval", {}).get("top_k", 5))

        if not (self.th_low < self.th_high):
            raise ValueError("confidence.threshold_low must be < confidence.threshold_high")

    def decide(self, hits: List[RetrievedChunk]) -> ConfidenceResult:
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
            # If evidence reinforces (same topic) or multiple strong hits, answer confidently
            if same_topic or support_count >= 2:
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
