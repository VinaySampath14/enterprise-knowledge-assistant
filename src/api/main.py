from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from src.api.schemas import QueryRequest, QueryResponse
from src.config import load_app_config
from src.api.deps import get_pipeline
from src.monitoring.stats import compute_stats_from_query_log, default_stats_summary
from src.rag.pipeline import RAGPipeline
from src.utils.query_logger import QueryLogger


def _validate_runtime_dependencies(repo_root: Path, cfg: Any) -> List[str]:
    errors: List[str] = []

    index_path = repo_root / cfg.index.index_path
    meta_path = repo_root / cfg.index.meta_path

    if not index_path.exists():
        errors.append(f"index file not found: {index_path}")
    if not meta_path.exists():
        errors.append(f"meta file not found: {meta_path}")

    if bool(cfg.generation.enabled) and not os.getenv("OPENAI_API_KEY"):
        errors.append("OPENAI_API_KEY is required when generation is enabled")

    if bool(cfg.logging.enabled):
        log_path = repo_root / cfg.logging.path
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            errors.append(f"invalid logging path: {log_path} ({e})")

    return errors


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise Knowledge Assistant", version="0.1.0")

    @app.on_event("startup")
    def _startup() -> None:
        repo_root = Path(__file__).resolve().parents[2]
        load_dotenv(repo_root / ".env")
        app.state.startup_errors = []

        try:
            cfg, _ = load_app_config(repo_root)
        except Exception as e:
            app.state.pipeline = None
            app.state.startup_errors = [f"config load failed: {e}"]
            return

        startup_errors = _validate_runtime_dependencies(repo_root, cfg)

        app.state.repo_root = repo_root
        app.state.logging_enabled = bool(cfg.logging.enabled)
        app.state.query_log_path = repo_root / cfg.logging.path
        app.state.query_logger = QueryLogger(app.state.query_log_path) if app.state.logging_enabled else None
        app.state.startup_errors = startup_errors

        if startup_errors:
            app.state.pipeline = None
            return

        try:
            app.state.pipeline = RAGPipeline(repo_root)
        except Exception as e:
            app.state.pipeline = None
            app.state.startup_errors = [f"pipeline init failed: {e}"]

    @app.get("/health", response_model=dict)
    def health() -> dict:
        pipeline = getattr(app.state, "pipeline", None)
        pipeline_loaded = pipeline is not None

        dependencies = {
            "retriever_loaded": bool(getattr(pipeline, "retriever", None)) if pipeline_loaded else False,
            "gate_loaded": bool(getattr(pipeline, "gate", None)) if pipeline_loaded else False,
            "generator_loaded": bool(getattr(pipeline, "generator", None)) if pipeline_loaded else False,
        }

        return {
            "status": "ok" if pipeline_loaded else "degraded",
            "service": "enterprise-knowledge-assistant",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pipeline_loaded": pipeline_loaded,
            "dependencies": dependencies,
            "startup_errors": getattr(app.state, "startup_errors", []),
        }

    @app.get("/stats", response_model=dict)
    def stats() -> dict:
        logging_enabled = bool(getattr(app.state, "logging_enabled", True))
        log_path = getattr(app.state, "query_log_path", Path("logs/queries.jsonl"))
        summary = (
            compute_stats_from_query_log(log_path)
            if logging_enabled
            else default_stats_summary()
        )

        return {
            "service": "enterprise-knowledge-assistant",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "log_path": str(log_path),
            "logging_enabled": logging_enabled,
            **summary,
        }

    @app.post("/query", response_model=QueryResponse)
    def query_endpoint(
        payload: QueryRequest,
        pipeline: RAGPipeline = Depends(get_pipeline),
        x_request_id: str | None = Header(default=None),
    ):
        request_id = x_request_id or str(uuid.uuid4())

        if not payload.query.strip():
            raise HTTPException(status_code=400, detail="query must not be empty or whitespace")

        try:
            out = pipeline.run(payload.query, request_id=request_id)

            query_logger = getattr(app.state, "query_logger", None)
            if query_logger is not None:
                try:
                    sources = out.get("sources", []) if isinstance(out, dict) else []
                    query_logger.log(
                        {
                            "query": payload.query,
                            "type": out.get("type") if isinstance(out, dict) else None,
                            "confidence": out.get("confidence") if isinstance(out, dict) else None,
                            "meta": out.get("meta") if isinstance(out, dict) else {},
                            "sources": sources if isinstance(sources, list) else [],
                            "num_sources": len(sources) if isinstance(sources, list) else 0,
                        },
                        request_id=request_id,
                    )
                except Exception:
                    # Logging failures should not fail API queries.
                    pass

            return JSONResponse(content=out, headers={"X-Request-ID": request_id})
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail="Upstream generation provider error") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail="Internal server error") from e

    return app


app = create_app()
