# Ablation Comparison Report

- baseline_run_dir: C:\Users\Vinay\OneDrive\Desktop\enterprise-knowledge-assistant\artifacts\experiments\phase0\V1_phase0_20260329_163011_baseline
- current_run_dir: C:\Users\Vinay\OneDrive\Desktop\enterprise-knowledge-assistant\artifacts\experiments\phase0\V2_phase0_20260329_163011_candidate
- generated_utc: 2026-03-29T16:31:57.205161+00:00
- baseline_run_id: V1_phase0_20260329_163011_baseline
- current_run_id: V2_phase0_20260329_163011_candidate

## Core Metrics
- overall_accuracy: base=0.7000, current=0.7000, delta=+0.0000
- clarify_rate: base=0.2500, current=0.2500, delta=+0.0000
- false_refusal_rate: base=0.2500, current=0.2500, delta=+0.0000
- false_answer_rate: base=0.0000, current=0.0000, delta=+0.0000

## Predicted Type Counts
- answer: base=3, current=3, delta=+0
- clarify: base=5, current=5, delta=+0
- refuse: base=12, current=12, delta=+0

## Per-Category Accuracy
| Category | Baseline Count | Current Count | Baseline Acc | Current Acc | Delta |
|---|---:|---:|---:|---:|---:|
| in_domain_answerable | 4 | 4 | 0.7500 | 0.7500 | +0.0000 |
| near_domain_should_refuse | 4 | 4 | 0.7500 | 0.7500 | +0.0000 |
| out_of_domain_unanswerable | 4 | 4 | 0.7500 | 0.7500 | +0.0000 |
| python_general_out_of_scope | 4 | 4 | 0.7500 | 0.7500 | +0.0000 |
| recoverable_should_clarify | 4 | 4 | 0.5000 | 0.5000 | +0.0000 |

## Guardrail Check
- false_refusal_guardrail: passed=True, delta=+0.0000, max_allowed=+0.0500
- false_answer_guardrail: passed=True, delta=+0.0000, max_allowed=+0.0200
- overall_guardrail_status: PASS

## Final Decision
- promotion_recommendation: GO
