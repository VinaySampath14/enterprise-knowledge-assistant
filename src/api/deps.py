from __future__ import annotations

from pathlib import Path
from fastapi import Request

from src.rag.pipeline import RAGPipeline


def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline
