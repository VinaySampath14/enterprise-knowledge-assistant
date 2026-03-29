# Phase 0 Run Checklist

## 1. Run Header
- Run ID:
- Date/Time (UTC):
- Owner:
- Parent baseline run ID: N/A (first baseline)
- Commit hash:
- Environment (OS/Python/GPU-CPU):

## 2. Dataset Lock Verification
- [ ] Fixed evaluation set selected
- [ ] Label corrections applied exactly once
- [ ] Dataset path recorded
- [ ] Dataset SHA256 recorded
- [ ] Dataset owner/approver recorded

Pass/Fail:
Notes:

## 3. Config Freeze Verification
- [ ] Active config snapshot captured
- [ ] Thresholds recorded
- [ ] Retrieval settings recorded
- [ ] Generation settings recorded

Pass/Fail:
Notes:

## 4. Baseline Execution - Run A
Command:

```
python scripts/experiments/run_phase0_baseline.py --dataset ... --category-field ... --expected-type-field ...
```

- [ ] Command saved to command.txt
- [ ] predictions_run_a.jsonl generated
- [ ] summary_run_a.json generated
- [ ] Exit code is 0

Pass/Fail:
Notes:

## 5. Reproducibility Execution - Run B
- [ ] Same inputs and same config rerun
- [ ] Row-level comparison completed
- [ ] Match rate computed
- [ ] Repro threshold met (default >= 0.95)

Pass/Fail:
Notes:

## 6. Artifact Bundle Completeness
Required files:
- [ ] config_snapshot.yaml
- [ ] predictions.jsonl
- [ ] summary.json
- [ ] report.md
- [ ] per_category_metrics.json
- [ ] failure_taxonomy.json
- [ ] gate_decision_breakdown.json
- [ ] metadata.json
- [ ] command.txt

Pass/Fail:
Notes:

## 7. Baseline Diagnostics
- [ ] Overall accuracy recorded
- [ ] Per-category metrics table recorded
- [ ] Clarify rate recorded
- [ ] False refusal rate recorded
- [ ] False answer rate recorded
- [ ] Representative failures captured per category

Pass/Fail:
Notes:

## 8. Phase Gate Decision
Go criteria:
- [ ] Required artifacts present
- [ ] Reproducibility threshold passed
- [ ] Per-category metrics present
- [ ] Failure taxonomy present

No-Go triggers:
- [ ] Missing artifact
- [ ] Reproducibility below threshold
- [ ] Missing per-category or failure outputs

Final Decision:
- [ ] GO
- [ ] NO-GO

Approver:
Date:
