from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
import yaml
from pathlib import Path


class Embedder:
    def __init__(self, config_path: Path):
        with config_path.open("r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        model_name = self.config["embeddings"]["model_name"]
        self.normalize = self.config["embeddings"].get("normalize", True)
        self.batch_size = self.config["embeddings"].get("batch_size", 64)

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, a_min=1e-12, a_max=None)

        return embeddings.astype("float32")
