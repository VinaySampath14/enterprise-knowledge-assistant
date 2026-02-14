from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from src.retrieval.retriever import Retriever
from src.rag.confidence import ConfidenceGate
from src.rag.generator import Generator
from src.utils.timing import Timer
from src.utils.query_logger import QueryLogger
from src.api.schemas import QueryResponse, ResponseMeta, SourceItem


class RAGPipeline:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.retriever = Retriever(repo_root)
        self.gate = ConfidenceGate(repo_root)
        self.generator = Generator(repo_root)

        cfg_path = repo_root / "config.yaml"
        with cfg_path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        log_cfg = self.cfg.get("logging", {})
        self.logging_enabled = bool(log_cfg.get("enabled", True))
        log_path = repo_root / log_cfg.get("path", "logs/queries.jsonl")
        self.query_logger = QueryLogger(log_path)

        print("LOGGING ENABLED:", self.logging_enabled)
        print("LOG PATH:", log_path)


    def run(self, query: str, *, request_id: Optional[str] = None) -> Dict[str, Any]:
        t_total = Timer.begin()

        # Retrieval
        t_ret = Timer.begin()
        hits = self.retriever.retrieve(query)
        ret_ms = t_ret.ms()

        # Confidence
        conf = self.gate.decide(hits)

        # Generation
        t_gen = Timer.begin()
        gen_out = self.generator.generate(query, conf)
        gen_ms = t_gen.ms()

        # Sources (attach what was used, not just what was retrieved)
        sources = []
        for c in conf.used_chunks:
            sources.append(
                SourceItem(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    module=c.module,
                    score=float(c.score),
                    source_path=c.meta.get("source_path"),
                )
            )

        meta = ResponseMeta(
            top_score=float(conf.top_score),
            retrieved_k=len(hits),
            latency_ms_total=t_total.ms(),
            latency_ms_retrieval=ret_ms,
            latency_ms_generation=gen_ms,
        )

        resp = QueryResponse(
            type=gen_out["type"],
            answer=gen_out["answer"],
            confidence=float(gen_out["confidence"]),
            sources=sources,
            meta=meta,
        )

        out = resp.model_dump()

        # Structured log record (store summary + sources)
        if self.logging_enabled:
            log_record = {
                "query": query,
                "type": out["type"],
                "confidence": out["confidence"],
                "meta": out["meta"],
                "sources": out.get("sources", []),
            }
            rid = self.query_logger.log(log_record, request_id=request_id)
            out["meta"]["request_id"] = rid

        return out
