from pathlib import Path
import sys

# Ensure repo root is on sys.path when this file is executed directly.
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.eval_runner.run_eval import main

if __name__ == "__main__":
    main()
