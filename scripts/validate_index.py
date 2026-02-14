from __future__ import annotations

from pathlib import Path
import yaml
import faiss

from src.utils.jsonl import iter_jsonl


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    chunks_path = repo_root / "data" / "processed" / "chunks.jsonl"
    index_path = repo_root / cfg["index"]["index_path"]
    meta_path = repo_root / cfg["index"]["meta_path"]

    assert chunks_path.exists(), f"Missing: {chunks_path}"
    assert index_path.exists(), f"Missing: {index_path}"
    assert meta_path.exists(), f"Missing: {meta_path}"

    # Count chunks
    n_chunks = sum(1 for _ in iter_jsonl(chunks_path))

    # Load FAISS index
    index = faiss.read_index(str(index_path))
    n_vec = index.ntotal
    dim = index.d

    # Count meta + validate vector_id sequence
    n_meta = 0
    expected_vid = 0
    for rec in iter_jsonl(meta_path):
        assert rec["vector_id"] == expected_vid, f"vector_id mismatch: got {rec['vector_id']} expected {expected_vid}"
        expected_vid += 1
        n_meta += 1

    assert n_chunks == n_meta, f"chunks ({n_chunks}) != meta ({n_meta})"
    assert n_vec == n_meta, f"faiss vectors ({n_vec}) != meta ({n_meta})"
    assert dim > 0, "FAISS dimension must be > 0"

    print(f"[OK] index validated: chunks={n_chunks}, vectors={n_vec}, dim={dim}")


if __name__ == "__main__":
    main()

