from pathlib import Path
import json
import sys
from src.rag.pipeline import RAGPipeline


def _run_and_print(pipeline: RAGPipeline, query: str) -> None:
    result = pipeline.run(query)

    print(f"\nQUERY: {query}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = RAGPipeline(repo_root)

    args = sys.argv[1:]
    if args and args[0] == "--demo-citations":
        _run_and_print(pipeline, "How do I join paths in Python?")
        _run_and_print(pipeline, "Who won the FIFA World Cup in 2018?")
        return

    query = " ".join(args) or "How do I open a sqlite3 connection?"
    result = pipeline.run(query)

    print("\nRESULT:")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
