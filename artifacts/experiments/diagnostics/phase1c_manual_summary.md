# Phase 1b Before/After Summary

- overall_accuracy_delta: +0.0606
- tie_breaker_fired_count: 2
- changed_decision_count: 2

## Target Bucket Deltas

- clarify vs refuse confusion: before=5, after=5, delta=+0
- conceptual in-domain false refusals: before=2, after=2, delta=+0
- python-general leakage: before=4, after=2, delta=-2

## Changed Decisions

- man-018: clarify -> refuse (expected=refuse, tie_breaker_fired=True)
- man-022: clarify -> refuse (expected=refuse, tie_breaker_fired=True)
