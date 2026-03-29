# Fallback Forensic Comparison (2026-03-28)

## Experiment Scope
- Manual baseline: V1_phase0_20260328_151928_baseline
- Manual fallback candidate: V2_phase0_20260328_160027_candidate
- Synthetic baseline: V1_phase0_20260328_151738_baseline
- Synthetic fallback candidate: V2_phase0_20260328_155824_candidate

## Corrected vs Regressed Cases
| id | cohort | expected | before | after | fallback_used | gate_decision | mismatch_subtype | py-general-signals | top_modules |
|---|---|---|---|---|---:|---|---|---:|---|
| man-003 | corrected | answer | clarify | answer | False | answer | not_applicable | 0 | collections, collections, collections |
| man-007 | corrected | answer | refuse | answer | True | answer | not_applicable | 0 | itertools, itertools, collections |
| man-009 | corrected | answer | refuse | answer | True | answer | not_applicable | 0 | asyncio, asyncio, sys |
| man-011 | regressed | refuse | refuse | answer | True | answer | not_applicable | 0 | argparse, argparse, argparse |
| man-013 | regressed | refuse | refuse | answer | True | answer | not_applicable | 0 | threading, threading, json |
| man-014 | regressed | refuse | refuse | answer | True | answer | none | 0 | csv, csv, csv |
| man-016 | regressed | refuse | refuse | answer | True | answer | none | 0 | logging, logging, logging |
| man-017 | regressed | refuse | refuse | answer | True | answer | not_applicable | 0 | re, re, re |
| man-019 | regressed | refuse | refuse | answer | True | answer | not_applicable | 0 | collections, collections, json |
| man-020 | regressed | refuse | refuse | answer | True | answer | not_applicable | 0 | sys, sys, pickle |
| man-024 | regressed | clarify | refuse | answer | True | answer | not_applicable | 0 | asyncio, asyncio, concurrent.futures |
| synv2-0012 | regressed | refuse | refuse | answer | True | answer | none | 0 | heapq, heapq, argparse |

## Contrastive Findings
### Aggregate Signals
- Corrected count: 3
- Regressed count: 9
- Python-general signal rate: corrected 0.0% vs regressed 0.0%
- Explicit symbol-request pattern rate: corrected 0.0% vs regressed 11.1%
- Explanatory framing rate: corrected 100.0% vs regressed 55.6%
- Same top-2 module retrieval rate: corrected 100.0% vs regressed 100.0%

### Strongest Differences
- Corrected cases are tightly in-domain explanatory questions with direct stdlib intent (not tool-install or cross-domain).
- Regressed cases include many near-domain symbol mismatch prompts and python-general questions where fallback converted safe refuses into unsupported answers.
- Same-module retrieval alone is not sufficient; regressed cases show that retrieval coherence can be misleading when user intent is out-of-scope or mismatch-driven.
- Fallback overfire is primarily an eligibility failure, not a construction-only failure.

## Strict Fallback Eligibility Proposal
### SHOULD be Eligible
- Only gate_decision == 'answer' and generator refusal-like output was produced
- Only in-domain explanatory queries that explicitly mention a stdlib module or symbol
- Top-3 retrieval must be strongly coherent: same module for top-2 and at least 2/3 from same stdlib module
- No python_general_out_of_scope_signals
- Mismatch subtype must be 'not_applicable'
- Must have evidence of conceptual intent (what/how/explain/overview) and no cross-domain/tooling terms
### MUST be Excluded
- Any explicit 'In <module>, how do I use <symbol>' near-domain symbol mismatch pattern
- Queries with python-general/tooling/third-party language (install, numpy, pandas, sql, http, dataframe)
- Broad, cross-cutting prompts without anchored stdlib target
- Cases where retrieval coherence exists but query intent is not stdlib-documentation anchored
- Any case where top evidence text does not contain a directly relevant symbol/topic match
### Required Retrieval/Gate Conditions
- gate_decision must be answer before any fallback consideration
- mismatch_subtype must remain not_applicable
- top_score should be in strong band (>= threshold_high) and support_count high
- top2 modules identical and top3 includes >=2 chunks from same module
### Required Metadata/Debug Checks
- Log and require explicit eligibility_reasons and exclusion_reasons
- Log query_intent_flags (conceptual/explanatory, explicit_symbol_request, tooling_terms)
- Log module_anchor_strength (top2_same, top3_majority_module)
- Abort fallback if any exclusion reason is present
### Minimum Evidence Signals
- Refusal-like generated answer
- Strong stdlib anchoring in query + retrieval
- No out-of-scope/python-general signals
- No near-domain symbol-mismatch template

## Final Recommendation
- Choice: B (fallback should continue only for a tiny conceptual in-domain slice)
- The mechanism can recover true in-domain misses (man-007, man-009), but broad eligibility causes major false-answer regressions.
- Regressed cases are dominated by near-domain symbol mismatch and python-general intent where same-module retrieval is misleading.
- A very narrow eligibility gate is required; otherwise risk outweighs gain.

## Notes
- No behavior changes were made. This is forensic analysis only.
- Citation ids are extracted from answer_preview where available; previews can be truncated.
- Retrieval top chunks and mismatch/out-of-scope diagnostics are computed post-hoc from current index/code for forensic contrast.