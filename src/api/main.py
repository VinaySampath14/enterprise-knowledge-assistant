from __future__ import annotations

from pathlib import Path
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from src.api.schemas import QueryRequest, QueryResponse
from src.api.deps import get_pipeline
from src.rag.pipeline import RAGPipeline


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise Knowledge Assistant", version="0.1.0")

    @app.on_event("startup")
    def _startup() -> None:
        repo_root = Path(__file__).resolve().parents[2]
        app.state.pipeline = RAGPipeline(repo_root)

    @app.get("/health", response_model=dict)
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/query", response_model=QueryResponse)
    def query_endpoint(
        payload: QueryRequest,
        pipeline: RAGPipeline = Depends(get_pipeline),
        x_request_id: str | None = Header(default=None),
    ):
        request_id = x_request_id or str(uuid.uuid4())

        try:
            out = pipeline.run(payload.query, request_id=request_id)
            return JSONResponse(content=out, headers={"X-Request-ID": request_id})
        except Exception as e:
            raise HTTPException(status_code=500, detail="Internal server error") from e

    return app


app = create_app()
