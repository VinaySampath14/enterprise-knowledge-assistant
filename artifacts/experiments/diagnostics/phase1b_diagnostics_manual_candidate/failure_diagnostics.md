# Failure Diagnostics - phase1b_diagnostics_manual_candidate

- Predictions file: `C:\Users\Vinay\OneDrive\Desktop\enterprise-knowledge-assistant\artifacts\experiments\phase0\V2_phase0_20260328_145726_candidate\predictions.jsonl`
- Total predictions: **33**
- Total failures: **12**
- Failure rate: **0.364**

## Ranked Failure Buckets

1. **clarify vs refuse confusion** - count=5, likely_fixability=high, trigger_counts={'mismatch_recoverable': 3, 'ambiguity': 2}
2. **python-general leakage** - count=3, likely_fixability=high, trigger_counts={'ambiguity': 2, 'mismatch_recoverable': 1}
3. **conceptual in-domain false refusals** - count=2, likely_fixability=high, trigger_counts={'ambiguity': 2}
4. **other repeated bucket (answer->clarify, ambiguity)** - count=1, likely_fixability=high, trigger_counts={'ambiguity': 1}
5. **other repeated bucket (clarify->answer, ambiguity)** - count=1, likely_fixability=high, trigger_counts={'ambiguity': 1}

## Recommended Next Isolated Intervention

- Recommendation: Add a narrow ambiguity tie-breaker for broad conceptual queries: when top modules disagree and score margin is near boundary, prefer clarify with a module-disambiguating follow-up prompt template.
- Rationale: Highest-volume bucket is 'clarify vs refuse confusion' and is ambiguity-driven, which is usually fixable with a localized rule change and low regression risk.

## Per-Failure Detail

- id=man-003 | expected=answer | predicted=clarify | top=0.612 | second=0.584 | margin=0.028 | mismatch=not_applicable | trigger=ambiguity | bucket=other repeated bucket (answer->clarify, ambiguity) | modules=['collections']
- id=man-007 | expected=answer | predicted=refuse | top=0.585 | second=0.489 | margin=0.095 | mismatch=not_applicable | trigger=ambiguity | bucket=conceptual in-domain false refusals | modules=['itertools', 'collections']
- id=man-009 | expected=answer | predicted=refuse | top=0.651 | second=0.616 | margin=0.035 | mismatch=not_applicable | trigger=ambiguity | bucket=conceptual in-domain false refusals | modules=['asyncio', 'sys']
- id=man-012 | expected=refuse | predicted=clarify | top=0.645 | second=0.610 | margin=0.035 | mismatch=recoverable | trigger=mismatch_recoverable | bucket=clarify vs refuse confusion | modules=['asyncio']
- id=man-021 | expected=refuse | predicted=answer | top=0.630 | second=0.611 | margin=0.019 | mismatch=not_applicable | trigger=ambiguity | bucket=python-general leakage | modules=['threading', 'concurrent.futures']
- id=man-022 | expected=refuse | predicted=clarify | top=0.327 | second=0.324 | margin=0.003 | mismatch=recoverable | trigger=mismatch_recoverable | bucket=python-general leakage | modules=['statistics', 're', 'math', 'itertools']
- id=man-023 | expected=refuse | predicted=answer | top=0.648 | second=0.582 | margin=0.066 | mismatch=not_applicable | trigger=ambiguity | bucket=python-general leakage | modules=['functools']
- id=man-024 | expected=clarify | predicted=refuse | top=0.615 | second=0.556 | margin=0.059 | mismatch=not_applicable | trigger=ambiguity | bucket=clarify vs refuse confusion | modules=['asyncio', 'concurrent.futures', 'multiprocessing']
- id=man-027 | expected=clarify | predicted=answer | top=0.643 | second=0.624 | margin=0.020 | mismatch=not_applicable | trigger=ambiguity | bucket=other repeated bucket (clarify->answer, ambiguity) | modules=['collections', 'functools']
- id=man-029 | expected=refuse | predicted=clarify | top=0.279 | second=0.226 | margin=0.054 | mismatch=recoverable | trigger=mismatch_recoverable | bucket=clarify vs refuse confusion | modules=['string', 'statistics']
- id=man-030 | expected=refuse | predicted=clarify | top=0.262 | second=0.206 | margin=0.056 | mismatch=recoverable | trigger=mismatch_recoverable | bucket=clarify vs refuse confusion | modules=['itertools', 'math', 'logging']
- id=man-032 | expected=refuse | predicted=clarify | top=0.284 | second=0.242 | margin=0.041 | mismatch=not_applicable | trigger=ambiguity | bucket=clarify vs refuse confusion | modules=['math', 'threading', 'pathlib', 'multiprocessing']
