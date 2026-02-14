from pathlib import Path
import yaml
import faiss

from src.utils.jsonl import iter_jsonl


def test_index_and_meta_alignment():
    repo_root = Path(__file__).resolve().parents[1]

    with (repo_root / "config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    chunks_path = repo_root / "data" / "processed" / "chunks.jsonl"
    index_path = repo_root / cfg["index"]["index_path"]
    meta_path = repo_root / cfg["index"]["meta_path"]

    assert chunks_path.exists()
    assert index_path.exists()
    assert meta_path.exists()

    n_chunks = sum(1 for _ in iter_jsonl(chunks_path))

    index = faiss.read_index(str(index_path))
    n_vec = index.ntotal

    n_meta = 0
    expected = 0
    for rec in iter_jsonl(meta_path):
        assert rec["vector_id"] == expected
        expected += 1
        n_meta += 1

    assert n_chunks == n_meta
    assert n_vec == n_meta
    assert index.d > 0
