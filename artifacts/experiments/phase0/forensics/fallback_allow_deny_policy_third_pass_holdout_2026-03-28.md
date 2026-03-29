# Third Forensic Pass: Holdout Robustness Check (2026-03-28)

## Scope
- Source policy: artifacts\experiments\phase0\forensics\fallback_allow_deny_policy_second_pass_2026-03-28.json
- Manual candidate run: V2_phase0_20260328_160027_candidate
- Synthetic candidate run: V2_phase0_20260328_155824_candidate
- Method: apply policy to holdout rows (excluding the 12 derivation cases)

## Summary
- Total rows evaluated: 55
- Training 12 rows: 12
- Holdout rows: 43
- Holdout eligible count: 0
- Holdout eligible expected_type counts: {}
- Holdout eligible category counts: {}

## Risk/Benefit Proxy
- Potential benefit (expected answer) count: 0
- Potential risk (expected != answer) count: 0
- Potential benefit ids: []
- Potential risk ids: []

## Eligible Holdout Cases
- None

## Recommendation
- abandon_fallback
- Holdout rows show non-answer eligible leakage or zero meaningful coverage.