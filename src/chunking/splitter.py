from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SplitConfig:
    chunk_size: int = 800
    overlap: int = 150


def split_text_with_offsets(text: str, cfg: SplitConfig) -> List[Dict]:
    """
    Split text into overlapping character chunks.
    Returns list of dicts: {text, start_char, end_char, chunk_index}
    """
    if cfg.chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if cfg.overlap < 0:
        raise ValueError("overlap must be >= 0")
    if cfg.overlap >= cfg.chunk_size:
        raise ValueError("overlap must be < chunk_size")

    n = len(text)
    if n == 0:
        return []

    step = cfg.chunk_size - cfg.overlap
    chunks: List[Dict] = []
    start = 0
    idx = 0

    while start < n:
        end = min(start + cfg.chunk_size, n)
        chunk_text = text[start:end]

        # Skip pure-whitespace chunks (optional but usually good)
        if chunk_text.strip():
            chunks.append(
                {
                    "text": chunk_text,
                    "start_char": start,
                    "end_char": end,
                    "chunk_index": idx,
                }
            )
            idx += 1

        # Move forward by step
        if end == n:
            break
        start += step

    return chunks


def chunk_doc_record(doc: Dict, cfg: SplitConfig) -> List[Dict]:
    """
    Convert one docs.jsonl record -> list of chunks.jsonl records (schema-aligned).
    """
    doc_id = doc["id"]
    module = doc["module"]
    source_path = doc["source"]["path"]
    text = doc["text"]

    created_at = _utc_now_iso()

    base_chunks = split_text_with_offsets(text, cfg)

    out: List[Dict] = []
    for c in base_chunks:
        chunk_index = c["chunk_index"]
        chunk_id = f"{doc_id}#c{chunk_index:04d}"

        out.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "module": module,
                "text": c["text"],
                "start_char": c["start_char"],
                "end_char": c["end_char"],
                "chunk_index": chunk_index,
                "created_at": created_at,
                "meta": {
                    "source_path": source_path,
                    "heading": None,  # optional; we can add heading extraction later
                },
            }
        )

    return out
