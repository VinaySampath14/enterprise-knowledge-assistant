# Phase 0 Baseline Report

- run_id: V2_phase0_20260329_184943_candidate
- dataset: eval/holdout_paraphrases.jsonl
- dataset_sha256: 379c0309109bac2fadf084f2df115de22b552017b13f45bc8129cf8473cd4832
- total_queries: 20
- overall_accuracy: 0.7000

## Core Decision Metrics
- clarify_rate: 0.2500
- false_refusal_rate: 0.2500
- false_answer_rate: 0.0000
- predicted_type_counts: {'answer': 3, 'refuse': 12, 'clarify': 5}

## Reproducibility
- comparable_rows: 20
- matched_rows: 20
- match_rate: 1.0000
- diff_count: 0

## Per-Category Accuracy
- in_domain_answerable: accuracy=0.7500, count=4
- near_domain_should_refuse: accuracy=0.7500, count=4
- out_of_domain_unanswerable: accuracy=0.7500, count=4
- python_general_out_of_scope: accuracy=0.7500, count=4
- recoverable_should_clarify: accuracy=0.5000, count=4

## Phase 0 Gate
- required_artifacts_present: True
- reproducibility_passed: True
- final_decision: GO
