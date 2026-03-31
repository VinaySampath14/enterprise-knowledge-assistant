# Hardening Signal Calibration

- ablation_version: step11_eval_hardening_metrics
- total_answer_predictions: 23
- bad_answer_count: 1
- uncited_answer_count: 0
- bad_or_uncited_count: 1

## Recommended Thresholds
- faithfulness_low_alert_threshold: 0.850489
- answer_relevancy_low_alert_threshold: 0.147059
- combined_rule: alert_if faithfulness <= faithfulness_low_alert_threshold OR answer_relevancy <= answer_relevancy_low_alert_threshold

## Fit Metrics
- faithfulness: precision=0.1429, recall=1.0000, f1=0.2500, alert_rate=0.3043
- answer_relevancy: precision=0.1429, recall=1.0000, f1=0.2500, alert_rate=0.3043

## Shadow Breakdown
- holdout: answers=4, alerts=3 (0.7500), precision_bad_or_uncited=0.0000, recall_bad_or_uncited=0.0000
- manual: answers=11, alerts=6 (0.5455), precision_bad_or_uncited=0.1667, recall_bad_or_uncited=1.0000
- synthetic: answers=8, alerts=4 (0.5000), precision_bad_or_uncited=0.0000, recall_bad_or_uncited=0.0000

## Notes
- Reporting-only output for Step 12 hardening.
- Promotion guardrails are intentionally unchanged.