from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.utils.jsonl import iter_jsonl


TARGET_BUCKETS = {
    "clarify vs refuse confusion",
    "python-general leakage",
    "conceptual in-domain false refusals",
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bucket_counts(summary_json: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in summary_json.get("bucket_ranking", []):
        counts[str(row.get("bucket", ""))] = int(row.get("count", 0))
    return counts


def _changed_decisions(
    baseline_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    base_by_id = {str(r.get("id", "")): r for r in baseline_rows}
    out: List[Dict[str, Any]] = []

    for cur in candidate_rows:
        rid = str(cur.get("id", ""))
        if not rid or rid not in base_by_id:
            continue
        before = str(base_by_id[rid].get("predicted_type", ""))
        after = str(cur.get("predicted_type", ""))
        if before == after:
            continue
        out.append(
            {
                "id": rid,
                "query": cur.get("query"),
                "expected_type": cur.get("expected_type"),
                "before_predicted_type": before,
                "after_predicted_type": after,
                "tie_breaker_fired": bool(cur.get("gate_tie_breaker_fired", False)),
                "gate_rationale": cur.get("gate_rationale", ""),
            }
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Phase 1b before/after deltas.")
    parser.add_argument("--comparison-json", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--baseline-buckets", required=True)
    parser.add_argument("--candidate-buckets", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]

    comparison = _load_json(root / args.comparison_json)
    baseline_bucket_summary = _load_json(root / args.baseline_buckets)
    candidate_bucket_summary = _load_json(root / args.candidate_buckets)

    base_rows = list(iter_jsonl(root / args.baseline_predictions))
    cand_rows = list(iter_jsonl(root / args.candidate_predictions))

    base_counts = _bucket_counts(baseline_bucket_summary)
    cand_counts = _bucket_counts(candidate_bucket_summary)

    bucket_deltas: Dict[str, Dict[str, int]] = {}
    for b in sorted(TARGET_BUCKETS):
        before = int(base_counts.get(b, 0))
        after = int(cand_counts.get(b, 0))
        bucket_deltas[b] = {"before": before, "after": after, "delta": after - before}

    changed = _changed_decisions(base_rows, cand_rows)
    tie_breaker_fired_count = sum(1 for r in cand_rows if bool(r.get("gate_tie_breaker_fired", False)))

    summary = {
        "overall_accuracy_delta": float(
            comparison.get("core_deltas", {})
            .get("overall_accuracy", {})
            .get("delta", 0.0)
        ),
        "bucket_deltas": bucket_deltas,
        "tie_breaker_fired_count": tie_breaker_fired_count,
        "changed_decision_count": len(changed),
        "changed_decisions": changed,
    }

    out_json = root / args.output_json
    out_md = root / args.output_md
    out_json.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Phase 1b Before/After Summary\n\n")
    lines.append(f"- overall_accuracy_delta: {summary['overall_accuracy_delta']:+.4f}\n")
    lines.append(f"- tie_breaker_fired_count: {tie_breaker_fired_count}\n")
    lines.append(f"- changed_decision_count: {len(changed)}\n\n")
    lines.append("## Target Bucket Deltas\n\n")
    for name, vals in bucket_deltas.items():
        lines.append(
            f"- {name}: before={vals['before']}, after={vals['after']}, delta={vals['delta']:+d}\n"
        )
    lines.append("\n## Changed Decisions\n\n")
    if changed:
        for row in changed:
            lines.append(
                f"- {row['id']}: {row['before_predicted_type']} -> {row['after_predicted_type']} "
                f"(expected={row['expected_type']}, tie_breaker_fired={row['tie_breaker_fired']})\n"
            )
    else:
        lines.append("- none\n")

    out_md.write_text("".join(lines), encoding="utf-8")

    print(f"[OK] Wrote summary JSON: {out_json}")
    print(f"[OK] Wrote summary Markdown: {out_md}")


if __name__ == "__main__":
    main()
