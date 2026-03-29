## Executive Summary
- Best manual accuracy: step6_h4_symbol_retrieval (0.7273).
- Best holdout accuracy: step3_h1_synthetic_refresh (0.7000).
- Safety status uses false-answer and false-refusal thresholds configured in this script run.

## Legend
- Manual/Synth/Holdout: Overall accuracy on each evaluation slice.
- FalseAns: expected_refuse_predicted_answer / total_queries.
- FalseRef: expected_answer_predicted_refuse / total_queries.
- DeltaPrev(M/S/H): Manual, Synthetic, and Holdout deltas versus previous version row.
- Safety: pass/warn/fail based on configured false-answer and false-refusal limits.
- Faithfulness: Grounding quality score (placeholder until computed/logged).
- Decision and Run ID can be auto-filled from phase-gate MLflow runs by ablation version.
- Rows with excluded prefixes (default: dryrun_, tmp_) are hidden unless explicitly included.

## Comparison Table

Version | What Changed | Manual | Synth | Holdout | FalseAns | FalseRef | DeltaPrev(M/S/H) | Safety | Faith | Decision | Run ID | Notes
--------|--------------|--------|-------|---------|----------|----------|------------------|--------|-------|----------|--------|------
baseline | Baseline after confidence-gate cleanup | 0.6061 | 0.9091 | 0.6500 | 0.0606 | 0.0606 | -/-/- | fail | - |  |  | Reference baseline row
v1 (Intent layer (upstream conservative routing)) | Intent layer (upstream conservative routing) | 0.6970 | 0.9091 | 0.7000 | 0.0000 | 0.0606 | +0.0909/+0.0000/+0.0500 | warn | - | GO | aefb342e232d435a8e7c8e30f3530192 | Strong quality lift vs baseline; false-refusal still slightly high
v2 (Broad post-gen refusal soften to clarify) | Broad post-gen refusal soften to clarify | 0.5455 | 0.7727 | 0.6000 | 0.0000 | 0.0000 | -0.1515/-0.1364/-0.1000 | pass | - | GO | 6ed7986e02ab47488050100fb67ea447 | Rejected: large accuracy regression
v3 (Narrow grounded rescue fallback) | Narrow grounded rescue fallback | 0.6970 | 0.9091 | 0.7000 | 0.0000 | 0.0606 | +0.1515/+0.1364/+0.1000 | warn | - | GO | 7be66442c71f4c1e94cf33caf2ac110c | Recovered from step4 regression; similar to v1
v4 (Symbol-anchor rescue refinement) | Symbol-anchor rescue refinement | 0.6970 | 0.9091 | 0.7000 | 0.0000 | 0.0606 | +0.0000/+0.0000/+0.0000 | warn | - | GO | 7fb69b6f82794e69808e71b965bf2086 | No material change vs v3
v5 (Retrieval rerank for explicit dotted symbols) | Retrieval rerank for explicit dotted symbols | 0.7273 | 0.9091 | 0.7000 | 0.0000 | 0.0303 | +0.0303/+0.0000/+0.0000 | pass | - | GO | 77af7e05ffdf4f63ae07f371244ce378 | Current champion: improves manual and drops false-refusal to 0.0303
v6 (change) | - | 0.7273 | 0.9091 | 0.7000 | 0.0000 | 0.0303 | +0.0000/+0.0000/+0.0000 | pass | - | GO | 064255a47e0b410aa70bede7056138cd | -
