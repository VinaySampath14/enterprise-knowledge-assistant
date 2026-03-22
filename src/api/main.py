from __future__ import annotations

import json
import os
from pathlib import Path
from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from src.api.schemas import QueryRequest, QueryResponse
from src.config import load_app_config
from src.api.deps import get_pipeline
from src.rag.pipeline import RAGPipeline


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _default_stats_summary() -> Dict[str, Any]:
    return {
        "total_queries": 0,
        "type_counts": {"answer": 0, "clarify": 0, "refuse": 0},
        "avg_confidence": 0.0,
        "avg_top_score": 0.0,
        "avg_latency_ms_total": 0.0,
        "avg_latency_ms_retrieval": 0.0,
        "avg_latency_ms_generation": 0.0,
        "avg_num_sources": 0.0,
        "avg_groundedness_overlap": 0.0,
        "answer_only_avg_groundedness_overlap": 0.0,
    }


def _compute_stats_from_query_log(log_path: Path) -> Dict[str, Any]:
    summary = _default_stats_summary()
    type_counts = summary["type_counts"]

    confidences: List[float] = []
    top_scores: List[float] = []
    lat_total: List[float] = []
    lat_retrieval: List[float] = []
    lat_generation: List[float] = []
    num_sources: List[float] = []
    groundedness: List[float] = []
    groundedness_answer_only: List[float] = []

    total = 0

    if log_path.exists() and log_path.is_file():
        try:
            with log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(rec, dict):
                        continue

                    total += 1

                    rec_type = rec.get("type")
                    if rec_type in type_counts:
                        type_counts[rec_type] += 1

                    meta = rec.get("meta")
                    if not isinstance(meta, dict):
                        meta = {}

                    confidences.append(_safe_float(rec.get("confidence"), 0.0))
                    top_scores.append(_safe_float(meta.get("top_score"), 0.0))
                    lat_total.append(_safe_float(meta.get("latency_ms_total"), 0.0))
                    lat_retrieval.append(_safe_float(meta.get("latency_ms_retrieval"), 0.0))
                    lat_generation.append(_safe_float(meta.get("latency_ms_generation"), 0.0))

                    if "num_sources" in rec:
                        nsrc = _safe_float(rec.get("num_sources"), 0.0)
                    else:
                        srcs = rec.get("sources")
                        nsrc = float(len(srcs)) if isinstance(srcs, list) else 0.0
                    num_sources.append(nsrc)

                    g = rec.get("groundedness_overlap")
                    if g is not None:
                        g_val = _safe_float(g, 0.0)
                        groundedness.append(g_val)
                        if rec_type == "answer":
                            groundedness_answer_only.append(g_val)
        except OSError:
            return summary

    return {
        "total_queries": total,
        "type_counts": type_counts,
        "avg_confidence": _mean(confidences),
        "avg_top_score": _mean(top_scores),
        "avg_latency_ms_total": _mean(lat_total),
        "avg_latency_ms_retrieval": _mean(lat_retrieval),
        "avg_latency_ms_generation": _mean(lat_generation),
        "avg_num_sources": _mean(num_sources),
        "avg_groundedness_overlap": _mean(groundedness),
        "answer_only_avg_groundedness_overlap": _mean(groundedness_answer_only),
    }


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
        app.state.startup_errors = []

        try:
            cfg, _ = load_app_config(repo_root)
        except Exception as e:
            app.state.pipeline = None
            app.state.startup_errors = [f"config load failed: {e}"]
            return

        startup_errors = _validate_runtime_dependencies(repo_root, cfg)

        app.state.repo_root = repo_root
        app.state.query_log_path = repo_root / cfg.logging.path
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
        log_path = getattr(app.state, "query_log_path", Path("logs/queries.jsonl"))
        summary = _compute_stats_from_query_log(log_path)

        return {
            "service": "enterprise-knowledge-assistant",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "log_path": str(log_path),
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
            return JSONResponse(content=out, headers={"X-Request-ID": request_id})
        except RuntimeError as e:
            raise HTTPException(status_code=502, detail="Upstream generation provider error") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail="Internal server error") from e

    return app


app = create_app()
