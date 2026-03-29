from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.rag.confidence import ConfidenceGate
from src.retrieval.retriever import Retriever
from src.utils.jsonl import iter_jsonl, write_jsonl


@dataclass(frozen=True)
class FailureRecord:
    query_id: str
    query: str
    expected_type: str
    predicted_type: str
    gate_decision: str
    top_score: float
    second_score: float
    margin: float
    mismatch_status: str
    mismatch_reasons: List[str]
    ambiguity_flags: List[str]
    top_retrieved_modules: List[str]
    decision_trigger_primary: str
    decision_trigger_path: List[str]
    category: str
    likely_bucket: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "expected_type": self.expected_type,
            "predicted_type": self.predicted_type,
            "gate_decision": self.gate_decision,
            "top_score": self.top_score,
            "second_score": self.second_score,
            "score_margin": self.margin,
            "mismatch_status": self.mismatch_status,
            "mismatch_reasons": self.mismatch_reasons,
            "ambiguity_flags": self.ambiguity_flags,
            "top_retrieved_modules": self.top_retrieved_modules,
            "decision_trigger_primary": self.decision_trigger_primary,
            "decision_trigger_path": self.decision_trigger_path,
            "category": self.category,
            "failure_bucket": self.likely_bucket,
        }


def _read_predictions(path: Path) -> List[Dict[str, Any]]:
    rows = list(iter_jsonl(path))
    if not rows:
        raise ValueError(f"No records found in predictions file: {path}")
    return rows


def _top_modules(hits: Iterable[Any], top_n: int = 5) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for h in hits:
        module = (getattr(h, "module", None) or "").strip()
        if not module or module in seen:
            continue
        seen.add(module)
        ordered.append(module)
        if len(ordered) >= top_n:
            break
    return ordered


def _classify_trigger(
    *,
    gate_decision: str,
    final_predicted_type: str,
    top_score: float,
    th_low: float,
    mismatch_status: str,
    ambiguity_flags: List[str],
) -> Tuple[str, List[str]]:
    path: List[str] = []

    if top_score < th_low:
        path.append("threshold_low")

    if mismatch_status in {"hard", "recoverable"}:
        path.append(f"mismatch_{mismatch_status}")

    if ambiguity_flags:
        path.append("ambiguity")

    if not path:
        path.append("other_rule_path")

    if final_predicted_type == "refuse" and gate_decision != "refuse":
        path.append("post_generation_refusal_override")

    primary = path[0]
    return primary, path


def _bucket_failure(
    *,
    expected_type: str,
    predicted_type: str,
    category: str,
    trigger_primary: str,
) -> str:
    category_lc = (category or "").lower()

    if expected_type == "answer" and predicted_type == "refuse":
        if "in_domain" in category_lc:
            return "conceptual in-domain false refusals"
        return "false refusals on answerable questions"

    if "python_general" in category_lc and predicted_type != expected_type:
        return "python-general leakage"

    if (
        expected_type == "clarify"
        and predicted_type == "refuse"
    ) or (
        expected_type == "refuse"
        and predicted_type == "clarify"
    ):
        return "clarify vs refuse confusion"

    return f"other repeated bucket ({expected_type}->{predicted_type}, {trigger_primary})"


def _fixability_label(bucket: str, trigger_counts: Dict[str, int]) -> str:
    if trigger_counts.get("mismatch_recoverable", 0) > 0 or trigger_counts.get("ambiguity", 0) > 0:
        return "high"
    if trigger_counts.get("threshold_low", 0) > 0 or trigger_counts.get("mismatch_hard", 0) > 0:
        return "medium"
    if "leakage" in bucket:
        return "medium"
    return "low"


def _fixability_score(label: str) -> int:
    return {"high": 3, "medium": 2, "low": 1}[label]


def _recommend_next_patch(ranked_buckets: List[Dict[str, Any]]) -> Dict[str, str]:
    if not ranked_buckets:
        return {
            "recommended_intervention": "No intervention needed.",
            "rationale": "No failures remain in the selected predictions set.",
        }

    top = ranked_buckets[0]
    bucket = top["bucket"]
    triggers = top.get("trigger_counts", {})

    if triggers.get("ambiguity", 0) > 0:
        return {
            "recommended_intervention": (
                "Add a narrow ambiguity tie-breaker for broad conceptual queries: when "
                "top modules disagree and score margin is near boundary, prefer clarify with "
                "a module-disambiguating follow-up prompt template."
            ),
            "rationale": (
                f"Highest-volume bucket is '{bucket}' and is ambiguity-driven, which is usually "
                "fixable with a localized rule change and low regression risk."
            ),
        }

    if triggers.get("mismatch_recoverable", 0) > 0:
        return {
            "recommended_intervention": (
                "Add a recoverable-mismatch escalation rule: for explicit symbol requests with "
                "weak symbol evidence, force clarify and ask for module confirmation."
            ),
            "rationale": (
                f"Highest-volume bucket is '{bucket}' and shows recoverable mismatch signals that "
                "can be targeted without threshold or architecture changes."
            ),
        }

    return {
        "recommended_intervention": (
            "Add a small false-refusal backoff rule in the low-score band for explicitly "
            "answerable in-domain questions with consistent top-module evidence."
        ),
        "rationale": (
            f"Highest-volume bucket is '{bucket}' and is not primarily ambiguity-driven; "
            "a constrained backoff rule is the most likely isolated ROI patch."
        ),
    }


def run(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    predictions_path = repo_root / args.predictions_jsonl
    rows = _read_predictions(predictions_path)

    retriever = Retriever(repo_root, config_path=(repo_root / args.config) if args.config else None)
    gate = ConfidenceGate(repo_root, config_path=(repo_root / args.config) if args.config else None)

    failures: List[FailureRecord] = []

    for row in rows:
        query = str(row.get("query", "")).strip()
        if not query:
            continue

        expected_type = str(row.get("expected_type", ""))
        predicted_type = str(row.get("predicted_type", row.get("pred_type", "")))
        if not expected_type or not predicted_type:
            continue

        if expected_type == predicted_type:
            continue

        hits = retriever.retrieve(query)
        decision = gate.decide(hits, query=query)
        mismatch_status, mismatch_reasons = gate._classify_mismatch(query, hits[: gate.max_chunks])  # noqa: SLF001
        ambiguity_flags = gate._ambiguity_signals(  # noqa: SLF001
            query,
            hits[: gate.max_chunks],
            margin=decision.margin,
            module_hints=gate._extract_module_hints(query),  # noqa: SLF001
            use_target=gate._extract_use_target(query),  # noqa: SLF001
        )

        primary_trigger, trigger_path = _classify_trigger(
            gate_decision=decision.decision,
            final_predicted_type=predicted_type,
            top_score=decision.top_score,
            th_low=gate.th_low,
            mismatch_status=mismatch_status,
            ambiguity_flags=ambiguity_flags,
        )

        category = str(row.get(args.category_field, row.get("category", "unknown")))
        failure_bucket = _bucket_failure(
            expected_type=expected_type,
            predicted_type=predicted_type,
            category=category,
            trigger_primary=primary_trigger,
        )

        failures.append(
            FailureRecord(
                query_id=str(row.get(args.id_field, row.get("id", "unknown"))),
                query=query,
                expected_type=expected_type,
                predicted_type=predicted_type,
                gate_decision=decision.decision,
                top_score=decision.top_score,
                second_score=decision.second_score,
                margin=decision.margin,
                mismatch_status=mismatch_status,
                mismatch_reasons=mismatch_reasons,
                ambiguity_flags=ambiguity_flags,
                top_retrieved_modules=_top_modules(hits),
                decision_trigger_primary=primary_trigger,
                decision_trigger_path=trigger_path,
                category=category,
                likely_bucket=failure_bucket,
            )
        )

    run_id = args.run_id or predictions_path.parent.name
    output_dir = repo_root / "artifacts" / "experiments" / "diagnostics" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    failures_jsonl = output_dir / "failure_diagnostics.jsonl"
    failures_json = output_dir / "failure_diagnostics.json"
    failures_md = output_dir / "failure_diagnostics.md"
    buckets_json = output_dir / "failure_bucket_summary.json"
    buckets_md = output_dir / "failure_bucket_summary.md"

    failure_rows = [f.to_dict() for f in failures]
    write_jsonl(failures_jsonl, failure_rows, append=False)
    failures_json.write_text(json.dumps(failure_rows, indent=2), encoding="utf-8")

    bucket_to_records: Dict[str, List[FailureRecord]] = defaultdict(list)
    for f in failures:
        bucket_to_records[f.likely_bucket].append(f)

    ranked: List[Dict[str, Any]] = []
    for bucket, bucket_records in bucket_to_records.items():
        trigger_counter = Counter(r.decision_trigger_primary for r in bucket_records)
        fixability = _fixability_label(bucket, dict(trigger_counter))
        ranked.append(
            {
                "bucket": bucket,
                "count": len(bucket_records),
                "likely_fixability": fixability,
                "trigger_counts": dict(trigger_counter),
                "example_query_ids": [r.query_id for r in bucket_records[:5]],
            }
        )

    ranked.sort(key=lambda x: (x["count"], _fixability_score(x["likely_fixability"])), reverse=True)
    recommendation = _recommend_next_patch(ranked)

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "predictions_path": str(predictions_path),
        "total_predictions": len(rows),
        "total_failures": len(failures),
        "failure_rate": (len(failures) / len(rows)) if rows else 0.0,
        "bucket_ranking": ranked,
        "recommended_next_patch": recommendation,
    }
    buckets_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines: List[str] = []
    md_lines.append(f"# Failure Diagnostics - {run_id}\n\n")
    md_lines.append(f"- Predictions file: `{predictions_path}`\n")
    md_lines.append(f"- Total predictions: **{len(rows)}**\n")
    md_lines.append(f"- Total failures: **{len(failures)}**\n")
    md_lines.append(f"- Failure rate: **{summary['failure_rate']:.3f}**\n\n")
    md_lines.append("## Ranked Failure Buckets\n\n")
    if ranked:
        for i, b in enumerate(ranked, start=1):
            md_lines.append(
                f"{i}. **{b['bucket']}** - count={b['count']}, likely_fixability={b['likely_fixability']}, "
                f"trigger_counts={b['trigger_counts']}\n"
            )
    else:
        md_lines.append("No failures found.\n")

    md_lines.append("\n## Recommended Next Isolated Intervention\n\n")
    md_lines.append(f"- Recommendation: {recommendation['recommended_intervention']}\n")
    md_lines.append(f"- Rationale: {recommendation['rationale']}\n")

    md_lines.append("\n## Per-Failure Detail\n\n")
    if failures:
        for f in failures:
            md_lines.append(
                f"- id={f.query_id} | expected={f.expected_type} | predicted={f.predicted_type} | "
                f"top={f.top_score:.3f} | second={f.second_score:.3f} | margin={f.margin:.3f} | "
                f"mismatch={f.mismatch_status} | trigger={f.decision_trigger_primary} | "
                f"bucket={f.likely_bucket} | modules={f.top_retrieved_modules}\n"
            )
    else:
        md_lines.append("No failed queries to report.\n")

    failures_md.write_text("".join(md_lines), encoding="utf-8")

    bucket_md_lines: List[str] = []
    bucket_md_lines.append(f"# Failure Bucket Summary - {run_id}\n\n")
    for i, b in enumerate(ranked, start=1):
        bucket_md_lines.append(f"{i}. bucket: **{b['bucket']}**\n")
        bucket_md_lines.append(f"   count: {b['count']}\n")
        bucket_md_lines.append(f"   likely_fixability: {b['likely_fixability']}\n")
        bucket_md_lines.append(f"   trigger_counts: {b['trigger_counts']}\n")
        bucket_md_lines.append(f"   example_query_ids: {b['example_query_ids']}\n\n")
    if not ranked:
        bucket_md_lines.append("No failures found.\n")
    buckets_md.write_text("".join(bucket_md_lines), encoding="utf-8")

    print(f"[OK] Wrote per-failure JSONL: {failures_jsonl}")
    print(f"[OK] Wrote per-failure JSON: {failures_json}")
    print(f"[OK] Wrote per-failure Markdown: {failures_md}")
    print(f"[OK] Wrote bucket summary JSON: {buckets_json}")
    print(f"[OK] Wrote bucket summary Markdown: {buckets_md}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explain remaining evaluation failures without changing behavior.")
    parser.add_argument(
        "--predictions-jsonl",
        required=True,
        help="Path to predictions.jsonl relative to repo root (e.g. artifacts/experiments/phase0/<run_id>/predictions.jsonl)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config path relative to repo root.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional diagnostics output run id (defaults to parent directory of predictions file).",
    )
    parser.add_argument(
        "--id-field",
        default="id",
        help="ID field in predictions rows.",
    )
    parser.add_argument(
        "--category-field",
        default="category",
        help="Category field in predictions rows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
