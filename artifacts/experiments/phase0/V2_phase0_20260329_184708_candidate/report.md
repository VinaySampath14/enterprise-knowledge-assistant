# Phase 0 Baseline Report

- run_id: V2_phase0_20260329_184708_candidate
- dataset: eval/manual.jsonl
- dataset_sha256: 42f41bd3af0f1e2c8e4a95764c540f8b8d7d9534fb63e3585a7e0c3660a8c207
- total_queries: 33
- overall_accuracy: 0.6970

## Core Decision Metrics
- clarify_rate: 0.2121
- false_refusal_rate: 0.2000
- false_answer_rate: 0.0000
- predicted_type_counts: {'answer': 9, 'refuse': 17, 'clarify': 7}

## Reproducibility
- comparable_rows: 33
- matched_rows: 33
- match_rate: 1.0000
- diff_count: 0

## Per-Category Accuracy
- in_domain_answerable: accuracy=0.8000, count=10
- near_domain_should_refuse: accuracy=0.8571, count=7
- out_of_domain_unanswerable: accuracy=0.6000, count=5
- python_general_out_of_scope: accuracy=0.6667, count=6
- recoverable_should_clarify: accuracy=0.4000, count=5

## Phase 0 Gate
- required_artifacts_present: True
- reproducibility_passed: True
- final_decision: GO
