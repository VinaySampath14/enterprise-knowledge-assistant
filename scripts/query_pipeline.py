from pathlib import Path
import sys
from src.rag.pipeline import RAGPipeline


def main():
    repo_root = Path(__file__).resolve().parents[1]
    pipeline = RAGPipeline(repo_root)

    query = " ".join(sys.argv[1:]) or "How do I open a sqlite3 connection?"
    result = pipeline.run(query)

    print("\nRESULT:")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
