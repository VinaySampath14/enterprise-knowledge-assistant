from pathlib import Path
from src.utils.jsonl import iter_jsonl

REQUIRED_KEYS = {
    "chunk_id",
    "doc_id",
    "module",
    "text",
    "start_char",
    "end_char",
    "chunk_index",
    "created_at",
    "meta",
}

def main():
    path = Path("data/processed/chunks.jsonl")
    assert path.exists(), "chunks.jsonl does not exist"

    ids = set()
    count = 0

    for c in iter_jsonl(path):
        count += 1

        # schema
        assert REQUIRED_KEYS.issubset(c.keys())

        # unique id
        assert c["chunk_id"] not in ids
        ids.add(c["chunk_id"])

        # offset consistency
        assert len(c["text"]) == c["end_char"] - c["start_char"]

        # non-empty text
        assert c["text"].strip()

    print(f"[OK] Validated {count} chunks")

if __name__ == "__main__":
    main()
