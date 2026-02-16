from pathlib import Path
from src.utils.jsonl import iter_jsonl

REQUIRED = {"id", "query", "expected_type"}

def main():
    repo_root = Path(__file__).resolve().parents[1]
    p = repo_root / "eval" / "questions.jsonl"
    ids = set()
    n = 0
    for obj in iter_jsonl(p):
        n += 1
        missing = REQUIRED - set(obj.keys())
        assert not missing, f"Missing keys {missing} in record: {obj}"
        assert obj["expected_type"] in ("answer", "refuse"), f"Invalid expected_type: {obj['expected_type']}"
        assert obj["id"] not in ids, f"Duplicate id: {obj['id']}"
        ids.add(obj["id"])
    print(f"[OK] {n} eval questions valid.")

if __name__ == "__main__":
    main()
