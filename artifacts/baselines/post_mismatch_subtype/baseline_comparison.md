# Post-Change Baseline Freeze

## Scope
- Evaluation/reporting only (no threshold or logic changes in this pass).
- Benchmarks: refined manual/diagnostic + refined synthetic.

## manual_refined
- overall_accuracy before=0.7500 after=0.6500 delta=-0.1000
- predicted_type_counts before={'answer': 7, 'refuse': 11, 'clarify': 2} after={'answer': 6, 'refuse': 9, 'clarify': 5}
- confusion_summary before={'expected_answer_predicted_refuse': 0, 'expected_answer_predicted_clarify': 0, 'expected_refuse_predicted_answer': 1, 'expected_refuse_predicted_clarify': 1, 'expected_clarify_predicted_answer': 0, 'expected_clarify_predicted_refuse': 3} after={'expected_answer_predicted_refuse': 0, 'expected_answer_predicted_clarify': 0, 'expected_refuse_predicted_answer': 0, 'expected_refuse_predicted_clarify': 4, 'expected_clarify_predicted_answer': 0, 'expected_clarify_predicted_refuse': 3}
- per-category accuracy:
  - in_scope_should_answer: before=1.0000 after=1.0000 delta=+0.0000
  - out_of_scope_should_refuse: before=1.0000 after=1.0000 delta=+0.0000
  - python_general_out_of_scope: before=0.6000 after=0.2000 delta=-0.4000
  - recoverable_should_clarify: before=0.2500 after=0.2500 delta=+0.0000

## synthetic_refined
- overall_accuracy before=1.0000 after=1.0000 delta=+0.0000
- predicted_type_counts before={'answer': 8, 'refuse': 14} after={'answer': 8, 'refuse': 14}
- confusion_summary before={'expected_answer_predicted_refuse': 0, 'expected_answer_predicted_clarify': 0, 'expected_refuse_predicted_answer': 0, 'expected_refuse_predicted_clarify': 0, 'expected_clarify_predicted_answer': 0, 'expected_clarify_predicted_refuse': 0} after={'expected_answer_predicted_refuse': 0, 'expected_answer_predicted_clarify': 0, 'expected_refuse_predicted_answer': 0, 'expected_refuse_predicted_clarify': 0, 'expected_clarify_predicted_answer': 0, 'expected_clarify_predicted_refuse': 0}
- per-category accuracy:
  - in_domain_answerable: before=1.0000 after=1.0000 delta=+0.0000
  - near_domain_should_refuse: before=1.0000 after=1.0000 delta=+0.0000
  - out_of_domain_unanswerable: before=1.0000 after=1.0000 delta=+0.0000

## Target Checks
- near_domain_should_refuse: before=1.0000, after=1.0000, improved=False
- recoverable_should_clarify: before=0.2500, after=0.2500, remained_correct=True
- in-domain answer behavior stable: True
- OOD refusal stable: True

## Conclusion
- Improved: no major gains detected versus archived baseline.
- Remaining: recoverable_should_clarify still under-target; python_general_out_of_scope still has clarify leakage

## Phase 2 Closeout (2026-03-29)

### Final promoted state
- manual overall_accuracy: 0.8485 (28/33)
- synthetic overall_accuracy: 1.0000 (22/22)
- promoted manual run: V2_phase0_20260328_190411_candidate
- promoted synthetic run: V2_phase0_20260328_190300_candidate

### Residual manual misses
- expected_answer_predicted_refuse remains 2
- IDs: man-007, man-009
- observed pattern: gate decision supports answer, but generation returns refusal-style text with zero citations

### Last intervention outcome (strict one-time retry path)
- focused tests passed
- guardrails showed no regressions
- behavioral movement: none
- manual delta vs promoted baseline: overall_accuracy 0.8485 -> 0.8485 (delta 0.0000)
- changed decision IDs: 0
- expected_answer_predicted_refuse: 2 -> 2

### Decision
- NO-GO for the strict retry patch (reverted).
- Phase 2 is considered complete enough; stop further Phase 2 patching.
- If revisited later, do forensic-only analysis first and design any new work as Phase 3 scope.
