# Ablation Report Template

## Header
- Run ID:
- Parent baseline Run ID:
- Phase:
- Objective:
- Date/Time (UTC):
- Commit hash:
- Config diff summary:

## Core Metrics
- Overall accuracy:
- Clarify rate:
- False refusal rate:
- False answer rate:
- Latency p50:
- Latency p95:

## Per-Category Metrics
| Category | Count | Accuracy | Delta vs Parent |
|---|---:|---:|---:|
| | | | |

## Decision Diagnostics
- Predicted type counts:
- Confusion summary:
- Reason code counts (if enabled):

### Top Changed Queries vs Parent
| Query ID | Category | Parent Decision | Current Decision | Expected | Notes |
|---|---|---|---|---|---|
| | | | | | |

## Retrieval Diagnostics (if applicable)
- Variant:
- Retrieval config:
- Attribution statement:

## Answer Quality Panel (if enabled)
- Answer-only lexical groundedness:
- Answer-only semantic groundedness:
- Faithfulness rate:
- Notable disagreement examples:

## Gate Evaluation
- Required artifacts present: pass/fail
- Global guardrails: pass/fail
- Phase-specific go criteria: pass/fail
- Phase-specific no-go triggers: pass/fail

## Final Decision
- Promote: GO/NO-GO
- Rollback recommendation:
- Next action:
