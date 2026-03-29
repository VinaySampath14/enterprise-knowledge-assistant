# Ablation Comparison Report

- baseline_run_dir: C:\Users\Vinay\OneDrive\Desktop\enterprise-knowledge-assistant\artifacts\experiments\phase0\V2_phase0_20260328_190300_candidate
- current_run_dir: C:\Users\Vinay\OneDrive\Desktop\enterprise-knowledge-assistant\artifacts\experiments\phase0\V2_phase0_20260329_204558_candidate
- generated_utc: 2026-03-29T20:47:07.900222+00:00
- baseline_run_id: V2_phase0_20260328_190300_candidate
- current_run_id: V2_phase0_20260329_204558_candidate

## Core Metrics
- overall_accuracy: base=1.0000, current=0.9091, delta=-0.0909
- clarify_rate: base=0.0000, current=0.0909, delta=+0.0909
- false_refusal_rate: base=0.0000, current=0.0000, delta=+0.0000
- false_answer_rate: base=0.0000, current=0.0000, delta=+0.0000

## Predicted Type Counts
- answer: base=8, current=8, delta=+0
- clarify: base=0, current=2, delta=+2
- refuse: base=14, current=12, delta=-2

## Per-Category Accuracy
| Category | Baseline Count | Current Count | Baseline Acc | Current Acc | Delta |
|---|---:|---:|---:|---:|---:|
| in_domain_answerable | 8 | 8 | 1.0000 | 1.0000 | +0.0000 |
| near_domain_should_refuse | 7 | 7 | 1.0000 | 0.7143 | -0.2857 |
| out_of_domain_unanswerable | 7 | 7 | 1.0000 | 1.0000 | +0.0000 |

## Guardrail Check
- false_refusal_guardrail: passed=True, delta=+0.0000, max_allowed=+0.0500
- false_answer_guardrail: passed=True, delta=+0.0000, max_allowed=+0.0200
- overall_guardrail_status: PASS

## Final Decision
- promotion_recommendation: GO
