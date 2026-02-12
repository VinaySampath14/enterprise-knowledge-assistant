from pathlib import Path

from src.ingest.load_raw import load_raw_docs


REQUIRED_KEYS = {"id", "module", "source", "text", "sha256", "created_at"}
REQUIRED_SOURCE_KEYS = {"path", "type"}


def test_ingestion_schema_and_uniqueness():
    repo_root = Path(__file__).resolve().parents[1]
    raw_dir = repo_root / "data" / "raw" / "python_stdlib"

    docs = load_raw_docs(raw_dir)

    assert len(docs) > 0

    ids = set()
    for d in docs:
        assert REQUIRED_KEYS.issubset(d.keys())
        assert isinstance(d["id"], str) and d["id"]
        assert isinstance(d["module"], str) and d["module"]
        assert isinstance(d["text"], str) and d["text"].strip()
        assert isinstance(d["sha256"], str) and len(d["sha256"]) == 64

        assert isinstance(d["source"], dict)
        assert REQUIRED_SOURCE_KEYS.issubset(d["source"].keys())
        assert d["source"]["type"] == "rst"

        assert d["id"] not in ids
        ids.add(d["id"])
