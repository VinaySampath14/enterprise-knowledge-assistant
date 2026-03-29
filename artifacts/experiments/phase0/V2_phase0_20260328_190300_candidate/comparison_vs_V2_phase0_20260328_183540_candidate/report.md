# Ablation Comparison Report

- baseline_run_dir: C:\Users\Vinay\OneDrive\Desktop\enterprise-knowledge-assistant\artifacts\experiments\phase0\V2_phase0_20260328_183540_candidate
- current_run_dir: C:\Users\Vinay\OneDrive\Desktop\enterprise-knowledge-assistant\artifacts\experiments\phase0\V2_phase0_20260328_190300_candidate
- generated_utc: 2026-03-28T19:04:01.953902+00:00
- baseline_run_id: V2_phase0_20260328_183540_candidate
- current_run_id: V2_phase0_20260328_190300_candidate

## Core Metrics
- overall_accuracy: base=1.0000, current=1.0000, delta=+0.0000
- clarify_rate: base=0.0000, current=0.0000, delta=+0.0000
- false_refusal_rate: base=0.0000, current=0.0000, delta=+0.0000
- false_answer_rate: base=0.0000, current=0.0000, delta=+0.0000

## Predicted Type Counts
- answer: base=8, current=8, delta=+0
- refuse: base=14, current=14, delta=+0

## Per-Category Accuracy
| Category | Baseline Count | Current Count | Baseline Acc | Current Acc | Delta |
|---|---:|---:|---:|---:|---:|
| in_domain_answerable | 8 | 8 | 1.0000 | 1.0000 | +0.0000 |
| near_domain_should_refuse | 7 | 7 | 1.0000 | 1.0000 | +0.0000 |
| out_of_domain_unanswerable | 7 | 7 | 1.0000 | 1.0000 | +0.0000 |

## Guardrail Check
- false_refusal_guardrail: passed=True, delta=+0.0000, max_allowed=+0.0500
- false_answer_guardrail: passed=True, delta=+0.0000, max_allowed=+0.0200
- overall_guardrail_status: PASS

## Final Decision
- promotion_recommendation: GO
