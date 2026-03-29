# Ablation Comparison Report

- baseline_run_dir: C:\Users\Vinay\OneDrive\Desktop\enterprise-knowledge-assistant\artifacts\experiments\phase0\V2_phase0_20260328_190411_candidate
- current_run_dir: C:\Users\Vinay\OneDrive\Desktop\enterprise-knowledge-assistant\artifacts\experiments\phase0\V2_phase0_20260329_184708_candidate
- generated_utc: 2026-03-29T18:48:38.706789+00:00
- baseline_run_id: V2_phase0_20260328_190411_candidate
- current_run_id: V2_phase0_20260329_184708_candidate

## Core Metrics
- overall_accuracy: base=0.8485, current=0.6970, delta=-0.1515
- clarify_rate: base=0.1212, current=0.2121, delta=+0.0909
- false_refusal_rate: base=0.2000, current=0.2000, delta=+0.0000
- false_answer_rate: base=0.0000, current=0.0000, delta=+0.0000

## Predicted Type Counts
- answer: base=9, current=9, delta=+0
- clarify: base=4, current=7, delta=+3
- refuse: base=20, current=17, delta=-3

## Per-Category Accuracy
| Category | Baseline Count | Current Count | Baseline Acc | Current Acc | Delta |
|---|---:|---:|---:|---:|---:|
| in_domain_answerable | 10 | 10 | 0.8000 | 0.8000 | +0.0000 |
| near_domain_should_refuse | 7 | 7 | 0.8571 | 0.8571 | +0.0000 |
| out_of_domain_unanswerable | 5 | 5 | 1.0000 | 0.6000 | -0.4000 |
| python_general_out_of_scope | 6 | 6 | 1.0000 | 0.6667 | -0.3333 |
| recoverable_should_clarify | 5 | 5 | 0.6000 | 0.4000 | -0.2000 |

## Guardrail Check
- false_refusal_guardrail: passed=True, delta=+0.0000, max_allowed=+0.0500
- false_answer_guardrail: passed=True, delta=+0.0000, max_allowed=+0.0200
- overall_guardrail_status: PASS

## Final Decision
- promotion_recommendation: GO
