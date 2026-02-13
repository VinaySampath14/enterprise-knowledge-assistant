from __future__ import annotations

from pathlib import Path
import yaml

from src.utils.jsonl import iter_jsonl, write_jsonl
from src.embeddings.embedder import Embedder
from src.retrieval.faiss_store import FaissStore


def main():
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "config.yaml"

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    chunks_path = repo_root / "data" / "processed" / "chunks.jsonl"
    index_path = repo_root / config["index"]["index_path"]
    meta_path = repo_root / config["index"]["meta_path"]

    # Load chunks
    chunks = list(iter_jsonl(chunks_path))
    print(f"[INFO] Loaded {len(chunks)} chunks")

    # Initialize embedder
    embedder = Embedder(config_path)

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts)

    print(f"[INFO] Embeddings shape: {embeddings.shape}")

    # Build FAISS index
    dim = embeddings.shape[1]
    store = FaissStore(dim)
    store.add(embeddings)
    store.save(index_path)

    print(f"[OK] Saved FAISS index to {index_path}")

    # Write metadata aligned by vector row
    def meta_records():
        for i, c in enumerate(chunks):
            yield {
                "vector_id": i,
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "module": c["module"],
                "text": c["text"],
                "meta": c["meta"],
                "start_char": c["start_char"],
                "end_char": c["end_char"],
                "chunk_index": c["chunk_index"],
            }

    n = write_jsonl(meta_path, meta_records(), append=False)
    print(f"[OK] Wrote {n} metadata rows to {meta_path}")


if __name__ == "__main__":
    main()
