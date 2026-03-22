from __future__ import annotations

from fastapi import HTTPException, Request

from src.rag.pipeline import RAGPipeline


def get_pipeline(request: Request) -> RAGPipeline:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        startup_errors = getattr(request.app.state, "startup_errors", [])
        detail = "Service unavailable: pipeline is not initialized"
        if startup_errors:
            detail = f"{detail}. startup_errors={startup_errors}"
        raise HTTPException(status_code=503, detail=detail)
    return pipeline
