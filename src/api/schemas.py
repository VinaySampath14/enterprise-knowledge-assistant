from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

ResponseType = Literal["answer", "clarify", "refuse"]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class SourceItem(BaseModel):
    chunk_id: str
    doc_id: Optional[str] = None
    module: str
    score: float
    source_path: Optional[str] = None


class ResponseMeta(BaseModel):
    top_score: float = 0.0
    retrieved_k: int = 0
    latency_ms_total: float = 0.0
    latency_ms_retrieval: float = 0.0
    latency_ms_generation: float = 0.0
    request_id: Optional[str] = None


class QueryResponse(BaseModel):
    type: ResponseType
    answer: str
    confidence: float
    sources: List[SourceItem] = Field(default_factory=list)
    meta: ResponseMeta
