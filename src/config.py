from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel
import yaml


# ================================
# Section Models
# ================================

class ChunkingConfig(BaseModel):
    chunk_size: int
    overlap: int


class EmbeddingsConfig(BaseModel):
    model_name: str
    normalize: bool = True
    batch_size: int = 64


class IndexConfig(BaseModel):
    index_path: str
    meta_path: str


class RetrievalConfig(BaseModel):
    top_k: int = 5


class ConfidenceConfig(BaseModel):
    threshold_high: float = 0.40
    threshold_low: float = 0.25
    margin_min: float = 0.03


class LoggingConfig(BaseModel):
    enabled: bool = True
    path: str = "logs/queries.jsonl"


class GenerationConfig(BaseModel):
    enabled: bool = True
    model: str = "gpt-4o-mini"
    temperature: float = 0.0


class AppConfig(BaseModel):
    chunking: ChunkingConfig
    embeddings: EmbeddingsConfig
    index: IndexConfig
    retrieval: RetrievalConfig
    confidence: ConfidenceConfig
    logging: LoggingConfig
    generation: GenerationConfig


# ================================
# Loader Function
# ================================

def load_config(config_path: Path | str = "config.yaml") -> AppConfig:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return AppConfig(**data)


def load_app_config(
    repo_root: Path,
    config_path: Optional[Path] = None,
) -> Tuple[AppConfig, Path]:
    """
    Load app config from a repo root + optional explicit config path.
    Returns both parsed config and resolved config path.
    """
    resolved = config_path or (repo_root / "config.yaml")
    return load_config(resolved), resolved

