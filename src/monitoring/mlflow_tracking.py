from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional


def _slug(value: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s.lower()[:120] if s else "default"


def _to_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def resolve_tracking_uri(repo_root: Path) -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if uri:
        return uri
    return (repo_root / "artifacts" / "mlflow").resolve().as_uri()


def _setup_mlflow(repo_root: Path, experiment_name: str):
    try:
        import mlflow
    except ImportError:
        return None, "", "mlflow package not installed; skipping tracking"

    tracking_uri = resolve_tracking_uri(repo_root)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow, tracking_uri, None


def _log_artifact_if_exists(mlflow_mod: Any, path: Path) -> None:
    if path.exists() and path.is_file():
        mlflow_mod.log_artifact(str(path))


def log_eval_tracking(
    *,
    repo_root: Path,
    experiment_name: str,
    run_name: str,
    params: Dict[str, Any],
    tags: Dict[str, Any],
    summary: Dict[str, Any],
    summary_path: Path,
    results_path: Path,
) -> Optional[str]:
    mlflow_mod, tracking_uri, err = _setup_mlflow(repo_root, experiment_name)
    if err:
        print(f"[WARN] {err}")
        return None

    with mlflow_mod.start_run(run_name=run_name):
        for k, v in params.items():
            mlflow_mod.log_param(k, str(v))

        mlflow_mod.set_tags({k: str(v) for k, v in tags.items()})

        top_level_metrics = {
            "overall_accuracy": summary.get("overall_accuracy"),
            "total_queries": summary.get("total_queries"),
            "total_answer_predictions": summary.get("total_answer_predictions"),
            "avg_faithfulness": summary.get("avg_faithfulness"),
            "avg_answer_relevancy": summary.get("avg_answer_relevancy"),
            "avg_faithfulness_answer_only": summary.get("avg_faithfulness_answer_only"),
            "avg_answer_relevancy_answer_only": summary.get("avg_answer_relevancy_answer_only"),
            "avg_faithfulness_all_predictions": summary.get("avg_faithfulness_all_predictions"),
            "avg_answer_relevancy_all_predictions": summary.get("avg_answer_relevancy_all_predictions"),
        }
        for k, v in top_level_metrics.items():
            num = _to_number(v)
            if num is not None:
                mlflow_mod.log_metric(k, num)

        confusion = summary.get("confusion_summary", {}) or {}
        confusion_metrics = {
            "expected_answer_predicted_refuse": confusion.get("expected_answer_predicted_refuse"),
            "expected_refuse_predicted_answer": confusion.get("expected_refuse_predicted_answer"),
            "expected_refuse_predicted_clarify": confusion.get("expected_refuse_predicted_clarify"),
            "expected_clarify_predicted_refuse": confusion.get("expected_clarify_predicted_refuse"),
        }
        for k, v in confusion_metrics.items():
            num = _to_number(v)
            if num is not None:
                mlflow_mod.log_metric(k, num)

        category_summary = summary.get("category_summary", {}) or {}
        for cat, obj in category_summary.items():
            if not isinstance(obj, dict):
                continue
            acc = _to_number(obj.get("accuracy"))
            if acc is not None:
                mlflow_mod.log_metric(f"category_accuracy__{_slug(cat)}", acc)

        _log_artifact_if_exists(mlflow_mod, summary_path)
        _log_artifact_if_exists(mlflow_mod, results_path)

    return tracking_uri


def log_phase_gate_tracking(
    *,
    repo_root: Path,
    experiment_name: str,
    run_name: str,
    params: Dict[str, Any],
    tags: Dict[str, Any],
    comparison_json: Optional[Path],
    comparison_report: Optional[Path],
    guardrails: Dict[str, Any],
    core_deltas: Dict[str, Any],
) -> Optional[str]:
    mlflow_mod, tracking_uri, err = _setup_mlflow(repo_root, experiment_name)
    if err:
        print(f"[WARN] {err}")
        return None

    with mlflow_mod.start_run(run_name=run_name):
        for k, v in params.items():
            mlflow_mod.log_param(k, str(v))

        mlflow_mod.set_tags({k: str(v) for k, v in tags.items()})

        overall_guard = _to_number(guardrails.get("overall_passed"))
        if overall_guard is not None:
            mlflow_mod.log_metric("guardrail_overall_passed", overall_guard)

        for key in ("false_refusal_guardrail", "false_answer_guardrail"):
            obj = guardrails.get(key, {}) or {}
            for field in ("delta", "max_allowed", "passed"):
                num = _to_number(obj.get(field))
                if num is not None:
                    mlflow_mod.log_metric(f"{key}__{field}", num)

        for name, obj in (core_deltas or {}).items():
            if not isinstance(obj, dict):
                continue
            for field in ("baseline", "current", "delta"):
                num = _to_number(obj.get(field))
                if num is not None:
                    mlflow_mod.log_metric(f"{name}__{field}", num)

        if comparison_json is not None:
            _log_artifact_if_exists(mlflow_mod, comparison_json)
        if comparison_report is not None:
            _log_artifact_if_exists(mlflow_mod, comparison_report)

    return tracking_uri
