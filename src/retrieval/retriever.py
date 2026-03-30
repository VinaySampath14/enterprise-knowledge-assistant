from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import numpy as np
import faiss

from src.config import load_app_config
from src.embeddings.embedder import Embedder
from src.utils.jsonl import iter_jsonl


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    module: str
    score: float
    text: str
    source_path: Optional[str]
    heading: Optional[str]
    meta: Dict[str, Any]
    start_char: int
    end_char: int
    chunk_index: int
    vector_id: int


class FaissMetaStore:
    """
    Loads meta.jsonl into a list so that meta[vector_id] is O(1).
    Assumes meta.jsonl is written in vector_id order (0..N-1).
    """

    def __init__(self, meta_path: Path):
        self.meta: List[Dict[str, Any]] = []
        for rec in iter_jsonl(meta_path):
            self.meta.append(rec)

    def __len__(self) -> int:
        return len(self.meta)

    def get(self, vector_id: int) -> Dict[str, Any]:
        return self.meta[vector_id]


class Retriever:
    def __init__(self, repo_root: Path, *, config_path: Optional[Path] = None):
        self.repo_root = repo_root
        self.cfg, self.config_path = load_app_config(repo_root, config_path)

        index_path = repo_root / self.cfg.index.index_path
        meta_path = repo_root / self.cfg.index.meta_path

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load meta aligned to vector ids
        self.meta_store = FaissMetaStore(meta_path)

        # Basic alignment check (fast)
        if self.index.ntotal != len(self.meta_store):
            raise ValueError(
                f"Index/meta mismatch: index.ntotal={self.index.ntotal} meta_rows={len(self.meta_store)}"
            )

        self.mode = str(self.cfg.retrieval.mode).strip().lower()
        if self.mode not in {"dense", "bm25", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {self.mode}")

        # Embedder is only required for dense/hybrid retrieval.
        self.embedder = Embedder(self.config_path) if self.mode in {"dense", "hybrid"} else None

        # BM25 is required for bm25/hybrid retrieval.
        self._bm25 = None
        if self.mode in {"bm25", "hybrid"}:
            try:
                from rank_bm25 import BM25Okapi  # type: ignore
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "rank-bm25 is required for retrieval.mode=bm25|hybrid. Install dependencies from requirements.txt."
                ) from e

            tokenized_corpus = [
                self._tokenize_for_bm25(str(rec.get("text", "")))
                for rec in self.meta_store.meta
            ]
            self._bm25 = BM25Okapi(tokenized_corpus)

        # Retrieval parameters from config.
        self.top_k = int(self.cfg.retrieval.top_k)
        self.hybrid_rrf_k = int(self.cfg.retrieval.hybrid_rrf_k)
        self.hybrid_dense_weight = float(self.cfg.retrieval.hybrid_dense_weight)
        self.hybrid_bm25_weight = float(self.cfg.retrieval.hybrid_bm25_weight)

    @staticmethod
    def _extract_symbol_mentions(query: str) -> List[str]:
        q = (query or "").lower()
        return re.findall(r"\b([a-z_][\w]*\.[a-z_][\w]*)\b", q)

    @staticmethod
    def _tokenize_for_bm25(text: str) -> List[str]:
        return re.findall(r"[a-zA-Z_][\w\.]*", (text or "").lower())

    @staticmethod
    def _symbol_rerank(results: List[RetrievedChunk], query: str) -> List[RetrievedChunk]:
        symbols = Retriever._extract_symbol_mentions(query)
        if not symbols:
            return results

        def bonus(hit: RetrievedChunk) -> float:
            text = (hit.text or "").lower()
            heading = (hit.heading or "").lower()
            module = (hit.module or "").lower()

            b = 0.0
            for sym in symbols:
                mod_prefix = sym.split(".", 1)[0]
                if sym in text or sym in heading:
                    b += 0.08
                if module == mod_prefix:
                    b += 0.03
            return b

        return sorted(results, key=lambda h: float(h.score) + bonus(h), reverse=True)

    @staticmethod
    def _clone_with_score(hit: RetrievedChunk, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            module=hit.module,
            score=float(score),
            text=hit.text,
            source_path=hit.source_path,
            heading=hit.heading,
            meta=hit.meta,
            start_char=hit.start_char,
            end_char=hit.end_char,
            chunk_index=hit.chunk_index,
            vector_id=hit.vector_id,
        )

    @staticmethod
    def _rrf_fuse(
        dense_hits: List[RetrievedChunk],
        bm25_hits: List[RetrievedChunk],
        *,
        rrf_k: int,
        dense_weight: float,
        bm25_weight: float,
    ) -> List[RetrievedChunk]:
        dense_rank = {h.chunk_id: i + 1 for i, h in enumerate(dense_hits)}
        bm25_rank = {h.chunk_id: i + 1 for i, h in enumerate(bm25_hits)}
        dense_score = {h.chunk_id: float(h.score) for h in dense_hits}

        by_chunk_id: Dict[str, RetrievedChunk] = {}
        for h in dense_hits + bm25_hits:
            if h.chunk_id not in by_chunk_id:
                by_chunk_id[h.chunk_id] = h

        fused: List[RetrievedChunk] = []
        for cid, h in by_chunk_id.items():
            s = 0.0
            if cid in dense_rank:
                s += dense_weight * (1.0 / (rrf_k + dense_rank[cid]))
            if cid in bm25_rank:
                s += bm25_weight * (1.0 / (rrf_k + bm25_rank[cid]))
            fused.append(Retriever._clone_with_score(h, s))

        fused = sorted(fused, key=lambda x: float(x.score), reverse=True)

        # Keep hybrid ranking from RRF, but present confidence-facing scores on
        # the same scale as dense retrieval so gate thresholds remain meaningful.
        out: List[RetrievedChunk] = []
        for h in fused:
            calibrated = dense_score.get(h.chunk_id, 0.0)
            out.append(Retriever._clone_with_score(h, calibrated))
        return out

    def _to_retrieved_chunk(self, *, score: float, vector_id: int) -> RetrievedChunk:
        rec = self.meta_store.get(int(vector_id))
        rec_meta = rec.get("meta") or {}
        return RetrievedChunk(
            chunk_id=rec["chunk_id"],
            doc_id=rec["doc_id"],
            module=rec["module"],
            score=float(score),
            text=rec["text"],
            source_path=rec_meta.get("source_path"),
            heading=rec_meta.get("heading"),
            meta=rec_meta,
            start_char=int(rec["start_char"]),
            end_char=int(rec["end_char"]),
            chunk_index=int(rec["chunk_index"]),
            vector_id=int(rec["vector_id"]),
        )

    def _dense_search(self, query: str, *, fetch_k: int) -> List[RetrievedChunk]:
        if self.embedder is None:
            return []

        q_vec = self.embedder.encode([query])
        if q_vec.shape[1] != self.index.d:
            raise ValueError(f"Query dim {q_vec.shape[1]} != index dim {self.index.d}")

        scores, ids = self.index.search(q_vec, fetch_k)
        scores = scores[0]
        ids = ids[0]

        results: List[RetrievedChunk] = []
        for score, vid in zip(scores, ids):
            if vid < 0:
                continue
            results.append(self._to_retrieved_chunk(score=float(score), vector_id=int(vid)))
        return results

    def _bm25_search(self, query: str, *, fetch_k: int) -> List[RetrievedChunk]:
        if self._bm25 is None:
            return []

        q_tokens = self._tokenize_for_bm25(query)
        if not q_tokens:
            return []

        scores = np.asarray(self._bm25.get_scores(q_tokens), dtype=np.float32)
        if scores.size == 0:
            return []

        top_idx = np.argsort(scores)[::-1][:fetch_k]
        results: List[RetrievedChunk] = []
        for vid in top_idx:
            score = float(scores[int(vid)])
            if score <= 0.0 and results:
                # Keep deterministic useful hits; stop once scores turn non-positive
                # after at least one candidate has been collected.
                continue
            results.append(self._to_retrieved_chunk(score=score, vector_id=int(vid)))
        return results

    def retrieve(self, query: str, *, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        k = int(top_k or self.top_k)
        if k <= 0:
            return []

        symbol_mentions = self._extract_symbol_mentions(query)
        fetch_k = k
        if symbol_mentions:
            fetch_k = min(max(k * 3, 10), int(self.index.ntotal))

        if self.mode == "dense":
            results = self._dense_search(query, fetch_k=fetch_k)
        elif self.mode == "bm25":
            results = self._bm25_search(query, fetch_k=fetch_k)
        else:
            dense_hits = self._dense_search(query, fetch_k=fetch_k)
            bm25_hits = self._bm25_search(query, fetch_k=fetch_k)
            results = self._rrf_fuse(
                dense_hits,
                bm25_hits,
                rrf_k=self.hybrid_rrf_k,
                dense_weight=self.hybrid_dense_weight,
                bm25_weight=self.hybrid_bm25_weight,
            )

        if symbol_mentions:
            results = self._symbol_rerank(results, query)

        return results[:k]
