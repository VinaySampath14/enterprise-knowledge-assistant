from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _module_name_from_path(p: Path) -> str:
    # pathlib.rst -> pathlib
    # os.path.rst -> os.path
    return p.stem


def iter_rst_files(root_dir: Path) -> Iterator[Path]:
    if not root_dir.exists():
        raise FileNotFoundError(f"Raw docs directory not found: {root_dir}")
    yield from sorted(root_dir.rglob("*.rst"))


def load_raw_docs(
    root_dir: Path,
    *,
    id_prefix: str = "py-stdlib",
    include_empty: bool = False,
) -> List[Dict]:
    """
    Load all .rst files under root_dir.
    Output records are ready to write to data/processed/docs.jsonl.
    """
    records: List[Dict] = []

    for fp in iter_rst_files(root_dir):
        text = fp.read_text(encoding="utf-8", errors="replace")
        if (not text.strip()) and (not include_empty):
            continue

        rel = fp.relative_to(root_dir).as_posix()  # stable relative path
        doc_id = f"{id_prefix}:{rel}"
        module = _module_name_from_path(fp)

        records.append(
            {
                "id": doc_id,
                "module": module,
                "source_path": fp.as_posix(),
                "text": text,
                "sha256": _sha256_text(text),
                "created_at": _utc_now_iso(),
            }
        )

    return records
