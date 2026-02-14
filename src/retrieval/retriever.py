from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
import faiss

from src.embeddings.embedder import Embedder
from src.utils.jsonl import iter_jsonl


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    module: str
    score: float
    text: str
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
        self.config_path = config_path or (repo_root / "config.yaml")

        with self.config_path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        index_path = repo_root / self.cfg["index"]["index_path"]
        meta_path = repo_root / self.cfg["index"]["meta_path"]

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

        # Embedder (loads sentence-transformers model)
        self.embedder = Embedder(self.config_path)

        # top_k from config
        self.top_k = int(self.cfg["retrieval"].get("top_k", 5))

    def retrieve(self, query: str, *, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        k = int(top_k or self.top_k)
        if k <= 0:
            return []

        # Embed query (shape: (1, dim))
        q_vec = self.embedder.encode([query])

        if q_vec.shape[1] != self.index.d:
            raise ValueError(f"Query dim {q_vec.shape[1]} != index dim {self.index.d}")


        # Search
        scores, ids = self.index.search(q_vec, k)  # scores: (1,k), ids: (1,k)
        scores = scores[0]
        ids = ids[0]

        results: List[RetrievedChunk] = []
        for score, vid in zip(scores, ids):
            if vid < 0:
                continue  # FAISS may return -1 if not enough entries

            rec = self.meta_store.get(int(vid))
            results.append(
                RetrievedChunk(
                    chunk_id=rec["chunk_id"],
                    doc_id=rec["doc_id"],
                    module=rec["module"],
                    score=float(score),
                    text=rec["text"],
                    meta=rec.get("meta", {}),
                    start_char=int(rec["start_char"]),
                    end_char=int(rec["end_char"]),
                    chunk_index=int(rec["chunk_index"]),
                    vector_id=int(rec["vector_id"]),
                )
            )

        return results
