from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import json
import re
import sys

# Ensure repo root is on sys.path when this file is executed directly.
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.utils.jsonl import iter_jsonl, write_jsonl


TYPO_HINT_RE = re.compile(r"\b(squilite|sqllite|argparze|asycio|jsoon|pythin)\b", re.IGNORECASE)


def _expected_distribution(items: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    dist: Dict[str, Counter[str]] = defaultdict(Counter)
    for item in items:
        cat = str(item.get("refined_category", "unknown"))
        expected = str(item.get("expected_type_refined", item.get("expected_type", "unknown")))
        dist[cat][expected] += 1
    return {cat: dict(counter) for cat, counter in sorted(dist.items())}


def _apply_refine_policy(item: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Returns (refined_category, expected_type_refined, policy_reason).
    """
    category = str(item.get("category", "")).strip()
    query = str(item.get("query", "")).strip()

    # Policy branch 1: explicit near-domain cross-module/library mismatch -> refuse.
    if category == "adversarial_near_domain":
        strategy = str(item.get("generation_meta", {}).get("strategy", ""))
        fn_src = str(item.get("generation_meta", {}).get("function_source_module", ""))
        prompted = str(item.get("generation_meta", {}).get("module_prompted", ""))

        if strategy == "cross_module_mismatch" or (fn_src and prompted and fn_src != prompted):
            return (
                "near_domain_should_refuse",
                "refuse",
                "Cross-module/library mismatch is near-domain but not answerable from intended corpus context.",
            )

        # Defensive fallback for any other adversarial near-domain items.
        return (
            "near_domain_should_refuse",
            "refuse",
            "Adversarial near-domain item defaults to refuse when answerability is not supported by corpus evidence.",
        )

    # Policy branch 2: recoverable ambiguity/typo -> clarify.
    if category == "borderline_weak_retrieval":
        if TYPO_HINT_RE.search(query) or "how to" in query.lower() or "?" in query:
            return (
                "recoverable_should_clarify",
                "clarify",
                "Likely recoverable intent (typo/ambiguous phrasing) should trigger clarification.",
            )

        return (
            "recoverable_should_clarify",
            "clarify",
            "Borderline weak retrieval is treated as recoverable clarification case.",
        )

    # Non-target categories stay unchanged.
    expected = str(item.get("expected_type", "unknown"))
    return (category, expected, "Category outside near-domain relabel scope; kept unchanged.")


def _refine_items(items: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in items:
        refined_category, expected_refined, reason = _apply_refine_policy(item)

        new_item = dict(item)
        new_item["source_benchmark"] = source_name
        new_item["original_category"] = item.get("category")
        new_item["original_expected_type"] = item.get("expected_type")
        new_item["refined_category"] = refined_category
        new_item["expected_type_refined"] = expected_refined
        new_item["label_policy_reason"] = reason

        out.append(new_item)

    return out


def _sample_examples(
    items: List[Dict[str, Any]],
    refined_category: str,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    selected = [x for x in items if x.get("refined_category") == refined_category][:limit]
    return [
        {
            "id": x.get("id"),
            "query": x.get("query"),
            "original_category": x.get("original_category"),
            "refined_category": x.get("refined_category"),
            "original_expected_type": x.get("original_expected_type"),
            "expected_type_refined": x.get("expected_type_refined"),
        }
        for x in selected
    ]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    src_eval_v2 = repo_root / "eval_v2" / "synthetic_scaffold_dataset.jsonl"
    src_diag = repo_root / "eval" / "diagnostic_citation_signal_questions.jsonl"

    out_eval_v2 = repo_root / "eval_v2" / "synthetic_scaffold_dataset_refined.jsonl"
    out_diag = repo_root / "eval" / "diagnostic_citation_signal_questions_refined.jsonl"
    out_summary = repo_root / "eval_v2" / "refined_label_policy_summary.json"

    eval_v2_items = list(iter_jsonl(src_eval_v2))
    diag_items = list(iter_jsonl(src_diag))

    refined_eval_v2 = _refine_items(eval_v2_items, "eval_v2.synthetic_scaffold")
    refined_diag = _refine_items(diag_items, "eval.diagnostic_citation_signal")

    write_jsonl(out_eval_v2, refined_eval_v2, append=False)
    write_jsonl(out_diag, refined_diag, append=False)

    combined = refined_eval_v2 + refined_diag

    summary: Dict[str, Any] = {
        "policy": {
            "near_domain_should_refuse": (
                "Queries close to domain but not answerable from corpus context, "
                "including wrong-module/wrong-library/concept mismatches."
            ),
            "recoverable_should_clarify": (
                "Queries likely recoverable via clarification, such as typos or ambiguous phrasing "
                "that may map to in-corpus intent."
            ),
        },
        "artifacts": {
            "eval_v2_refined_dataset": str(out_eval_v2),
            "diagnostic_refined_dataset": str(out_diag),
        },
        "counts_per_refined_category": dict(
            sorted(Counter(str(x.get("refined_category", "unknown")) for x in combined).items())
        ),
        "expected_type_distribution_per_refined_category": _expected_distribution(combined),
        "source_breakdown": {
            "eval_v2": {
                "counts_per_refined_category": dict(
                    sorted(Counter(str(x.get("refined_category", "unknown")) for x in refined_eval_v2).items())
                ),
                "expected_type_distribution_per_refined_category": _expected_distribution(refined_eval_v2),
            },
            "diagnostic": {
                "counts_per_refined_category": dict(
                    sorted(Counter(str(x.get("refined_category", "unknown")) for x in refined_diag).items())
                ),
                "expected_type_distribution_per_refined_category": _expected_distribution(refined_diag),
            },
        },
        "examples": {
            "near_domain_should_refuse": _sample_examples(combined, "near_domain_should_refuse", limit=3),
            "recoverable_should_clarify": _sample_examples(combined, "recoverable_should_clarify", limit=3),
        },
        "notes": [
            "Original benchmark artifacts are preserved; refined versions were written separately.",
            "This step updates evaluation labels/reporting only; no core RAG behavior changes.",
        ],
    }

    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Wrote refined eval_v2 dataset: {out_eval_v2}")
    print(f"[OK] Wrote refined diagnostic dataset: {out_diag}")
    print(f"[OK] Wrote policy summary: {out_summary}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()