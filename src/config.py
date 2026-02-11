from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings
import yaml


# ================================
# Section Models
# ================================

class ChunkingConfig(BaseModel):
    chunk_size: int
    overlap: int


class EmbeddingsConfig(BaseModel):
    model_name: str


class RetrievalConfig(BaseModel):
    top_k: int


class ConfidenceConfig(BaseModel):
    threshold_high: float
    threshold_low: float


class AppConfig(BaseModel):
    chunking: ChunkingConfig
    embeddings: EmbeddingsConfig
    retrieval: RetrievalConfig
    confidence: ConfidenceConfig


# ================================
# Loader Function
# ================================

def load_config(config_path: str = "config.yaml") -> AppConfig:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return AppConfig(**data)


# ================================
# Global Config Instance
# ================================

config = load_config()
