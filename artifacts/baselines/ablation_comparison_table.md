## Executive Summary
- Best manual accuracy: step10_reranker_cross_encoder (0.7576).
- Best holdout accuracy: step10_reranker_cross_encoder (0.7500).
- Safety status uses false-answer and false-refusal thresholds configured in this script run.

## Legend
- Decision Quality table is release-focused (accuracy, safety, and phase-gate outcome).
- Answer/Retrieval Quality table is diagnostic (evidence and performance signals).
- Manual/Synth/Holdout: Overall accuracy on each evaluation slice.
- FalseAns: expected_refuse_predicted_answer / total_queries.
- FalseRef: expected_answer_predicted_refuse / total_queries.
- DeltaPrev(M/S/H): Manual, Synthetic, and Holdout deltas versus previous version row.
- Safety: pass/warn/fail based on configured false-answer and false-refusal limits.
- Faithfulness: Grounding quality score (placeholder until computed/logged).
- Top(M/S/H): Weighted avg top_score from each dataset summary category breakdown.
- Cites(M/S/H): Weighted avg citations_count from each dataset summary category breakdown.
- LatMs(M/S/H): avg_latency_ms_total when present in summary/MLflow, else '-'.
- RAGAS notation in Notes: AR=answer_relevancy, CP=context_precision, Keep=answer_keep_rate.
- Decision and Run ID can be auto-filled from phase-gate MLflow runs by ablation version.
- Rows with excluded prefixes (default: dryrun_, tmp_) are hidden unless explicitly included.

## Decision Quality Table

Version | What Changed | Manual | Synth | Holdout | FalseAns | FalseRef | DeltaPrev(M/S/H) | Safety | Decision | Run ID | Notes
--------|--------------|--------|-------|---------|----------|----------|------------------|--------|----------|--------|------
baseline | Baseline after confidence-gate cleanup | 0.6061 | 0.9091 | 0.6500 | 0.0606 | 0.0606 | -/-/- | fail | REF |  | Reference baseline row
v1 | Intent layer (upstream conservative routing) | 0.6970 | 0.9091 | 0.7000 | 0.0000 | 0.0606 | +0.0909/+0.0000/+0.0500 | warn | GO | aefb342e232d435a8e7c8e30f3530192 | Strong quality lift vs baseline; false-refusal still slightly high
v2 | Broad post-gen refusal soften to clarify | 0.5455 | 0.7727 | 0.6000 | 0.0000 | 0.0000 | -0.1515/-0.1364/-0.1000 | pass | NO-GO | 6ed7986e02ab47488050100fb67ea447 | Rejected: large accuracy regression
v3 | Narrow grounded rescue fallback | 0.6970 | 0.9091 | 0.7000 | 0.0000 | 0.0606 | +0.1515/+0.1364/+0.1000 | warn | GO | 7be66442c71f4c1e94cf33caf2ac110c | Recovered from step4 regression; similar to v1
v4 | Symbol-anchor rescue refinement | 0.6970 | 0.9091 | 0.7000 | 0.0000 | 0.0606 | +0.0000/+0.0000/+0.0000 | warn | HOLD | 7fb69b6f82794e69808e71b965bf2086 | No material change vs v3
v5 | Retrieval rerank for explicit dotted symbols | 0.7273 | 0.9091 | 0.7000 | 0.0000 | 0.0303 | +0.0303/+0.0000/+0.0000 | pass | GO | 77af7e05ffdf4f63ae07f371244ce378 | Champion at v5: manual quality improved and false-refusal dropped to 0.0303
v6 | Citation-retry post-generation experiment | 0.7273 | 0.9091 | 0.7000 | 0.0000 | 0.0303 | +0.0000/+0.0000/+0.0000 | pass | HOLD | 064255a47e0b410aa70bede7056138cd | No material improvement vs champion; not retained
v7 | Hybrid retrieval (initial RRF scoring) | 0.5455 | 0.6364 | 0.6000 | 0.0000 | 0.3030 | -0.1818/-0.2727/-0.1000 | fail | NO-GO | 230bd81a3c964ef38ca2b9bdc0a2b561 | Rejected: RRF score scale mismatched confidence thresholds, causing near-universal refusals
v8 | Hybrid retrieval (RRF ranking + dense-scale confidence score) | 0.7273 | 0.9091 | 0.7000 | 0.0000 | 0.0303 | +0.1818/+0.2727/+0.1000 | pass | HOLD | fdf5fe3b045a48d4b247bafca0bc6ef3 | Validated fix and safety recovery to parity with champion; no net uplift vs best run, so not promoted
v9 | Cross-encoder reranker after retrieval (ms-marco-MiniLM-L-6-v2) | 0.7576 | 0.9091 | 0.7500 | 0.0000 | 0.0000 | +0.0303/+0.0000/+0.0500 | pass | GO | 8040794c517646a1b9ffec49d7114707 | Champion moved v5 -> v9: always-on reranking raised manual and holdout while maintaining safety pass
v10 | Conditional cross-encoder reranking enabled (low_margin_only, threshold 0.05); rerank applies only on low-confidence retrieval margins | 0.7273 | 0.9091 | 0.7000 | 0.0000 | 0.0000 | -0.0303/+0.0000/-0.0500 | pass | HOLD | 8f9e2728e44443beb3556f821e06063e | Validation run to compare selective reranking against always-on reranker baseline
v11 | Conditional cross-encoder reranking threshold sweep candidate (low_margin_only, threshold 0.15) | 0.7576 | 0.9091 | 0.7500 | 0.0000 | 0.0000 | +0.0303/+0.0000/+0.0500 | pass | GO | a18035ae23b74180b3f2c6c9c03b1c60 | Champion moved v9 -> v11: threshold 0.15 preserved quality and improved latency
v12 | Lock-in confirmation run for conditional reranking champion (low_margin_only, threshold 0.15) | 0.7576 | 0.9091 | 0.7500 | 0.0000 | 0.0000 | +0.0000/+0.0000/+0.0000 | pass |  |  | Final full-pass confirmation run before freezing this cycle


## Answer and Retrieval Quality Table

Version | Faith | AnswerRel | CtxPrec | KeepRate | LatMs(M/S/H)
--------|-------|-----------|---------|----------|------------------
v9 | - | - | - | - | 2006/1434/1464
v10 | - | - | - | - | 2012/1484/1370
v11 | 0.9141 | 0.9114 | 0.7571 | 0.4521 | 1608/1438/1273
v12 | 0.9141 | 0.9114 | 0.7571 | 0.4521 | 1602/1331/1213

