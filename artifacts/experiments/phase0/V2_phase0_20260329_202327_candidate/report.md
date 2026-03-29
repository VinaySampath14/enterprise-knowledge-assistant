# Phase 0 Baseline Report

- run_id: V2_phase0_20260329_202327_candidate
- dataset: eval_v2/synthetic_scaffold_dataset_refined.jsonl
- dataset_sha256: f2c43dcd04540be585dc6a44d72e0fd7178ca40b400eddebc68c45847c45b2fc
- total_queries: 22
- overall_accuracy: 0.9091

## Core Decision Metrics
- clarify_rate: 0.0909
- false_refusal_rate: 0.0000
- false_answer_rate: 0.0000
- predicted_type_counts: {'answer': 8, 'clarify': 2, 'refuse': 12}

## Reproducibility
- comparable_rows: 22
- matched_rows: 22
- match_rate: 1.0000
- diff_count: 0

## Per-Category Accuracy
- in_domain_answerable: accuracy=1.0000, count=8
- near_domain_should_refuse: accuracy=0.7143, count=7
- out_of_domain_unanswerable: accuracy=1.0000, count=7

## Phase 0 Gate
- required_artifacts_present: True
- reproducibility_passed: True
- final_decision: GO
