import os
from pathlib import Path

import pytest

from src.rag.pipeline import RAGPipeline

@pytest.mark.skipif(os.getenv("OPENAI_API_KEY") is None, reason="OPENAI_API_KEY not set")
def test_pipeline_smoke_openai():
    repo_root = Path(__file__).resolve().parents[1]
    p = RAGPipeline(repo_root)
    out = p.run("How do I open a sqlite3 connection?")
    assert out["type"] == "answer"
    assert "sqlite3" in out["answer"].lower()
