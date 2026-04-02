from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_float(v: Any, default: float = 0.0) -> float:
	try:
		if v is None:
			return default
		return float(v)
	except (TypeError, ValueError):
		return default


def _mean(values: List[float]) -> float:
	return sum(values) / len(values) if values else 0.0


def default_stats_summary() -> Dict[str, Any]:
	return {
		"total_queries": 0,
		"type_counts": {"answer": 0, "clarify": 0, "refuse": 0},
		"avg_confidence": 0.0,
		"avg_top_score": 0.0,
		"avg_latency_ms_total": 0.0,
		"avg_latency_ms_retrieval": 0.0,
		"avg_latency_ms_generation": 0.0,
		"avg_num_sources": 0.0,
		"avg_groundedness_overlap": 0.0,
		"answer_only_avg_groundedness_overlap": 0.0,
	}


def compute_stats_from_query_log(log_path: Path) -> Dict[str, Any]:
	summary = default_stats_summary()
	type_counts = summary["type_counts"]

	confidences: List[float] = []
	top_scores: List[float] = []
	lat_total: List[float] = []
	lat_retrieval: List[float] = []
	lat_generation: List[float] = []
	num_sources: List[float] = []
	groundedness: List[float] = []
	groundedness_answer_only: List[float] = []

	total = 0

	if log_path.exists() and log_path.is_file():
		try:
			with log_path.open("r", encoding="utf-8") as f:
				for line in f:
					line = line.strip()
					if not line:
						continue

					try:
						rec = json.loads(line)
					except json.JSONDecodeError:
						continue

					if not isinstance(rec, dict):
						continue

					total += 1

					rec_type = rec.get("type")
					if rec_type in type_counts:
						type_counts[rec_type] += 1

					meta = rec.get("meta")
					if not isinstance(meta, dict):
						meta = {}

					confidences.append(_safe_float(rec.get("confidence"), 0.0))
					top_scores.append(_safe_float(meta.get("top_score"), 0.0))
					lat_total.append(_safe_float(meta.get("latency_ms_total"), 0.0))
					lat_retrieval.append(_safe_float(meta.get("latency_ms_retrieval"), 0.0))
					lat_generation.append(_safe_float(meta.get("latency_ms_generation"), 0.0))

					if "num_sources" in rec:
						nsrc = _safe_float(rec.get("num_sources"), 0.0)
					else:
						srcs = rec.get("sources")
						nsrc = float(len(srcs)) if isinstance(srcs, list) else 0.0
					num_sources.append(nsrc)

					g = rec.get("groundedness_overlap")
					if g is not None:
						g_val = _safe_float(g, 0.0)
						groundedness.append(g_val)
						if rec_type == "answer":
							groundedness_answer_only.append(g_val)
		except OSError:
			return summary

	return {
		"total_queries": total,
		"type_counts": type_counts,
		"avg_confidence": _mean(confidences),
		"avg_top_score": _mean(top_scores),
		"avg_latency_ms_total": _mean(lat_total),
		"avg_latency_ms_retrieval": _mean(lat_retrieval),
		"avg_latency_ms_generation": _mean(lat_generation),
		"avg_num_sources": _mean(num_sources),
		"avg_groundedness_overlap": _mean(groundedness),
		"answer_only_avg_groundedness_overlap": _mean(groundedness_answer_only),
	}
