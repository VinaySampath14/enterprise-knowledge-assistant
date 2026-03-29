# Phase 1b Before/After Summary

- overall_accuracy_delta: +0.0303
- tie_breaker_fired_count: 1
- changed_decision_count: 1

## Target Bucket Deltas

- clarify vs refuse confusion: before=5, after=5, delta=+0
- conceptual in-domain false refusals: before=2, after=2, delta=+0
- python-general leakage: before=4, after=3, delta=-1

## Changed Decisions

- man-018: clarify -> refuse (expected=refuse, tie_breaker_fired=True)
