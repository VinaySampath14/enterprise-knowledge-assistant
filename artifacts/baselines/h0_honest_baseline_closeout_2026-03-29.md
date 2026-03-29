# H0 Honest Baseline Closeout (2026-03-29)

## Purpose
Record Step 1 H0 as a formal diagnostic reference baseline after confidence-gate simplification.

## Scope
- Confidence gate simplified to retrieval-confidence-first logic.
- Intent-style query rules removed from gate decisioning.
- This artifact is reference-only and does not update promoted baselines.

## H0 Candidate Runs
- Manual H0 candidate: `V2_phase0_20260329_140124_candidate`
- Synthetic H0 candidate: `V2_phase0_20260329_140133_candidate`

## Baseline Comparisons
### Manual (vs promoted manual baseline `V2_phase0_20260328_190411_candidate`)
- overall_accuracy: `0.8485 -> 0.6061` (delta `-0.2424`)
- clarify_rate: `0.1212 -> 0.2424` (delta `+0.1212`)
- false_refusal_rate: `0.2000 -> 0.2000` (delta `+0.0000`)
- false_answer_rate: `0.0000 -> 0.1111` (delta `+0.1111`)
- guardrail result: `NO-GO` (false_answer guardrail failed)

### Synthetic (vs promoted synthetic baseline `V2_phase0_20260328_190300_candidate`)
- overall_accuracy: `1.0000 -> 1.0000` (delta `+0.0000`)
- clarify_rate: `0.0000 -> 0.0000` (delta `+0.0000`)
- false_refusal_rate: `0.0000 -> 0.0000` (delta `+0.0000`)
- false_answer_rate: `0.0000 -> 0.0000` (delta `+0.0000`)
- guardrail result: `GO`

## Step 1 Decision
- H0 is accepted as a truthful diagnostic baseline.
- H0 is **not** promoted for manual dataset due to guardrail regression.
- Promoted baseline pointers remain unchanged.

## Next Step
Proceed to Step 2: paraphrased holdout generalization check against H0.
