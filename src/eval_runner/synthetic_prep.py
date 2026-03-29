from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re

from src.utils.jsonl import iter_jsonl, write_jsonl


FUNCTION_RE = re.compile(r"\.\.\s+function::\s+([\w\.]+)")


def _detect_ragas_available() -> bool:
    try:
        import ragas  # type: ignore  # noqa: F401

        return True
    except Exception:
        return False


def _load_chunk_candidates(chunks_path: Path, per_module_limit: int = 5) -> List[Dict[str, Any]]:
    """
    Load a bounded set of chunks to keep synthetic scaffold deterministic and lightweight.
    """
    by_module_count: Dict[str, int] = defaultdict(int)
    out: List[Dict[str, Any]] = []

    for rec in iter_jsonl(chunks_path):
        module = str(rec.get("module", "")).strip()
        if not module:
            continue

        if by_module_count[module] >= per_module_limit:
            continue

        text = str(rec.get("text", ""))
        if not text.strip():
            continue

        func_match = FUNCTION_RE.search(text)
        function_name = func_match.group(1) if func_match else None

        meta = rec.get("meta") if isinstance(rec.get("meta"), dict) else {}
        source_path = meta.get("source_path")

        out.append(
            {
                "chunk_id": rec.get("chunk_id"),
                "doc_id": rec.get("doc_id"),
                "module": module,
                "source_path": source_path,
                "text": text,
                "function_name": function_name,
            }
        )
        by_module_count[module] += 1

    return out


def _make_in_domain_items(candidates: List[Dict[str, Any]], target_n: int = 8) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    for c in candidates:
        if len(items) >= target_n:
            break

        module = c["module"]
        fn = c.get("function_name")

        if fn:
            query = f"How do I use {fn} in the {module} module?"
            note = "Function-targeted in-domain question generated from chunk function directive."
        else:
            query = f"What does the {module} module provide in Python standard library?"
            note = "Module-level in-domain question generated from chunk text."

        items.append(
            {
                "category": "in_domain_answerable",
                "expected_type": "answer",
                "query": query,
                "difficulty": "easy",
                "reference": {
                    "support_chunk_ids": [c["chunk_id"]],
                    "support_doc_ids": [c["doc_id"]],
                    "notes": note,
                },
                "generation_meta": {
                    "strategy": "chunk_template",
                    "module": module,
                    "function_name": fn,
                    "source_path": c.get("source_path"),
                },
            }
        )

    return items


def _make_adversarial_items(candidates: List[Dict[str, Any]], target_n: int = 7) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    funcs = [c for c in candidates if c.get("function_name")]
    if not funcs:
        funcs = candidates

    for i, c in enumerate(candidates):
        if len(items) >= target_n:
            break

        other = funcs[(i + 3) % len(funcs)] if funcs else c
        module_a = c["module"]
        fn_b = other.get("function_name") or "nonexistent_feature"
        module_b = other["module"]

        query = f"In {module_a}, how do I use {fn_b} exactly as documented in {module_a}?"

        items.append(
            {
                "category": "adversarial_near_domain",
                "expected_type": "clarify",
                "query": query,
                "difficulty": "medium",
                "reference": {
                    "support_chunk_ids": [c["chunk_id"]],
                    "support_doc_ids": [c["doc_id"]],
                    "notes": (
                        "Near-domain adversarial mismatch: function likely belongs to a different context. "
                        f"Function source module candidate={module_b}."
                    ),
                },
                "generation_meta": {
                    "strategy": "cross_module_mismatch",
                    "module_prompted": module_a,
                    "function_injected": fn_b,
                    "function_source_module": module_b,
                    "source_path": c.get("source_path"),
                },
            }
        )

    return items


def _make_out_of_domain_items(target_n: int = 7) -> List[Dict[str, Any]]:
    pool = [
        "Who won the FIFA World Cup in 2018?",
        "What is the capital of France?",
        "Write a short love poem.",
        "What are the symptoms of flu?",
        "What is the best budget laptop in 2026?",
        "How do I optimize Facebook ads?",
        "Explain quantum entanglement in simple terms.",
        "Give me a 7-day keto diet plan.",
        "What stock should I buy this week?",
        "How do I tune a guitar by ear?",
    ]

    items: List[Dict[str, Any]] = []
    for q in pool[:target_n]:
        items.append(
            {
                "category": "out_of_domain_unanswerable",
                "expected_type": "refuse",
                "query": q,
                "difficulty": "easy",
                "reference": {
                    "support_chunk_ids": [],
                    "support_doc_ids": [],
                    "notes": "Deliberately out-of-corpus query.",
                },
                "generation_meta": {
                    "strategy": "handcrafted_out_of_domain",
                },
            }
        )

    return items


def _attach_ids_and_ragas_fields(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, it in enumerate(items, start=1):
        item = dict(it)
        item_id = f"synv2-{i:04d}"
        item["id"] = item_id

        # RAGAS-style scaffold fields for future integration.
        item["ragas"] = {
            "question": item["query"],
            "ground_truth": None,
            "contexts": [],
            "answer": None,
            "metadata": {
                "expected_type": item["expected_type"],
                "category": item["category"],
            },
        }
        out.append(item)

    return out


def build_synthetic_scaffold(
    repo_root: Path,
    *,
    in_domain_n: int = 8,
    adversarial_n: int = 7,
    out_of_domain_n: int = 7,
) -> Dict[str, Any]:
    chunks_path = repo_root / "data" / "processed" / "chunks.jsonl"
    out_dataset = repo_root / "eval_v2" / "synthetic_scaffold_dataset.jsonl"
    out_summary = repo_root / "eval_v2" / "synthetic_scaffold_summary.json"

    candidates = _load_chunk_candidates(chunks_path)

    in_domain = _make_in_domain_items(candidates, target_n=in_domain_n)
    adversarial = _make_adversarial_items(candidates, target_n=adversarial_n)
    out_of_domain = _make_out_of_domain_items(target_n=out_of_domain_n)

    items = _attach_ids_and_ragas_fields(in_domain + adversarial + out_of_domain)

    write_jsonl(out_dataset, items, append=False)

    by_category = Counter(x["category"] for x in items)
    by_expected = Counter(x["expected_type"] for x in items)

    summary: Dict[str, Any] = {
        "total_items": len(items),
        "counts_by_category": dict(by_category),
        "counts_by_expected_type": dict(by_expected),
        "ragas_available": _detect_ragas_available(),
        "ragas_scaffold": {
            "ready": True,
            "required_fields": ["question", "ground_truth", "contexts", "answer"],
            "source_dataset": str(out_dataset),
        },
        "notes": [
            "This is scaffolding only: no threshold tuning or core RAG logic changes.",
            "Synthetic dataset is separate from manual benchmark outputs in eval/.",
        ],
    }

    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "dataset_path": str(out_dataset),
        "summary_path": str(out_summary),
        "summary": summary,
    }
