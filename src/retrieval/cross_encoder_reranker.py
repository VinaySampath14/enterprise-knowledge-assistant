from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence

import numpy as np

from src.config import load_app_config
from src.retrieval.retriever import RetrievedChunk


class CrossEncoderReranker:
    """
    Reorders retrieved chunks using a cross-encoder relevance model.
    Keeps original retrieval scores unchanged so confidence thresholds
    continue to operate on the same scale.
    """

    def __init__(self, repo_root: Path, *, model: Any | None = None):
        cfg, _ = load_app_config(repo_root)

        self.enabled = bool(cfg.reranker.enabled)
        self.model_name = str(cfg.reranker.model_name)
        self.candidate_k = max(1, int(cfg.reranker.candidate_k))
        self.max_length = max(32, int(cfg.reranker.max_length))

        self._model = model
        if self.enabled and self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=self.max_length)

    @staticmethod
    def _pair_text(hit: RetrievedChunk) -> str:
        heading = (hit.heading or "").strip()
        module = (hit.module or "").strip()
        text = (hit.text or "").strip()
        if heading:
            return f"{module} {heading}\n{text}".strip()
        return f"{module}\n{text}".strip()

    def rerank(self, query: str, hits: Sequence[RetrievedChunk], *, top_k: int) -> List[RetrievedChunk]:
        if top_k <= 0:
            return []
        if not self.enabled or len(hits) <= 1 or self._model is None:
            return list(hits)[:top_k]

        candidates = list(hits)[: self.candidate_k]
        pairs = [(query, self._pair_text(h)) for h in candidates]
        scores = self._model.predict(pairs)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        rank_idx = np.argsort(scores)[::-1]
        return [candidates[int(i)] for i in rank_idx[:top_k]]
