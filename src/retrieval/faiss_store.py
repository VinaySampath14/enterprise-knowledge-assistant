from __future__ import annotations

import faiss
import numpy as np
from pathlib import Path


class FaissStore:
    def __init__(self, dim: int):
        # Inner product index (works as cosine if normalized)
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray):
        self.index.add(vectors)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @staticmethod
    def load(path: Path):
        return faiss.read_index(str(path))
