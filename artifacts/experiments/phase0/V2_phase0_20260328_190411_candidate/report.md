# Phase 0 Baseline Report

- run_id: V2_phase0_20260328_190411_candidate
- dataset: eval/manual.jsonl
- dataset_sha256: 42f41bd3af0f1e2c8e4a95764c540f8b8d7d9534fb63e3585a7e0c3660a8c207
- total_queries: 33
- overall_accuracy: 0.8485

## Core Decision Metrics
- clarify_rate: 0.1212
- false_refusal_rate: 0.2000
- false_answer_rate: 0.0000
- predicted_type_counts: {'answer': 9, 'refuse': 20, 'clarify': 4}

## Reproducibility
- comparable_rows: 33
- matched_rows: 33
- match_rate: 1.0000
- diff_count: 0

## Per-Category Accuracy
- in_domain_answerable: accuracy=0.8000, count=10
- near_domain_should_refuse: accuracy=0.8571, count=7
- out_of_domain_unanswerable: accuracy=1.0000, count=5
- python_general_out_of_scope: accuracy=1.0000, count=6
- recoverable_should_clarify: accuracy=0.6000, count=5

## Phase 0 Gate
- required_artifacts_present: True
- reproducibility_passed: True
- final_decision: GO
