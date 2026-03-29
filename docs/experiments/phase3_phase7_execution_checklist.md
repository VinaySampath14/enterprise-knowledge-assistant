# Phase 3-7 Execution Checklist

## Goal
Close remaining quality gaps after Phase 2 with controlled, evidence-first changes while preserving synthetic stability and existing guardrails.

## Phase 3: Residual Forensics (No Code Changes)

### Objectives
- Re-validate root causes for residual manual errors, especially `man-007` and `man-009`.
- Produce an actionable failure taxonomy with explicit labels A-G.

### Inputs
- Promoted manual baseline summary and taxonomy artifacts.
- Promoted synthetic baseline summary artifacts.

### Outputs
- A forensic report with per-case fields:
  - id, expected_type, predicted_type
  - top_score, second_score, score_margin
  - gate_decision, gate_rationale
  - mismatch subtype and reasons
  - generation refusal signals, citation count, citation linkage status
- A documented taxonomy A-G (see schema below).

### Taxonomy A-G Schema
- A: Retrieval miss (relevant evidence not surfaced in top-k).
- B: Retrieval dilution (relevant evidence present but weak/co-mingled).
- C: Gate false refusal (gate should pass answer but blocks).
- D: Gate false answer (gate allows answer on unsupported query).
- E: Generator unsupported refusal (gate says answer, generator refuses).
- F: Generator unsupported answer (hallucinated or weakly grounded answer).
- G: Evaluation/label ambiguity (dataset case definition unclear or mixed).

### Exit Criteria
- Every residual miss assigned one primary A-G label and one secondary label (optional).
- Two highest-frequency labels identified with confidence level and evidence snippets.
- No code edits performed in this phase.

## Phase 4: Minimal Intervention Design

### Objectives
- Design at most 2 isolated interventions mapped directly to dominant Phase 3 labels.

### Required Design Artifacts
- Problem statement per intervention.
- Exact code surface (files/functions).
- Why this should move target bucket.
- Explicit non-goals (what will not be changed).

### Priority Missing Capability 1
- Hybrid BM25 + embedding retrieval design:
  - Candidate implementation path:
    - Add lexical index module under retrieval package.
    - Fuse FAISS dense score and BM25 lexical score with weighted normalization.
  - Required constraints:
    - Preserve existing retriever interface shape.
    - Keep top-k behavior configurable.

### Exit Criteria
- Intervention specs approved before implementation.
- Risk and rollback plan recorded.

## Phase 5: Controlled Ablation and Attribution

### Objectives
- Evaluate each intervention independently before any stacking.

### Execution Rules
- One intervention per candidate run.
- Compare against promoted baselines only.
- Track changed IDs and confusion bucket movement.

### Required Metrics
- overall_accuracy delta
- false_answer_rate delta
- false_refusal_rate delta
- clarify_rate delta
- Target confusion bucket deltas (especially expected_answer_predicted_refuse)

### Exit Criteria
- Promote only GO candidates.
- Revert NO-GO candidates immediately.

## Phase 6: Evaluation Hardening

### LLM-as-Judge (Recommended as Secondary Signal)
- Use an LLM judge to score answer quality dimensions that are hard to capture with lexical metrics alone.
- Keep it as a secondary diagnostic signal, not the only promotion gate.

### Judge Rubric (Per Example)
- Faithfulness to provided evidence.
- Answer relevancy to the user query.
- Citation correctness (claims backed by cited chunks).
- Helpfulness/coverage (did it answer the asked question sufficiently).

### Operating Rules
- Use fixed prompts and versioned judge configuration for reproducibility.
- Evaluate blind to candidate label (judge sees response/evidence, not run identity).
- Run pairwise on baseline vs candidate for changed IDs and hard cases.
- Keep a small human-audited set to calibrate judge drift.

### Promotion Use
- Do not promote on judge score alone.
- Require consistency with core guardrails and confusion-bucket movement.
- Use judge deltas as tie-breakers when top-line metrics are flat.

### Priority Missing Capability 2
- Full RAGAS faithfulness + answer relevancy integration:
  - Add evaluation runner support for both metrics as first-class outputs.
  - Persist per-example and aggregate metrics in comparison artifacts.
  - Keep a no-RAGAS fallback path when dependency is unavailable.

### Priority Missing Capability 3
- Semantic groundedness standardization:
  - Keep existing semantic groundedness metric.
  - Define one canonical aggregation in reports.

### Exit Criteria
- Re-runs show stable aggregate metrics within acceptable variance.
- RAGAS metrics are present in summary outputs for evaluation runs.
- LLM-judge outputs are reproducible and aligned with human-audited samples.

## Phase 7: Documentation and Release Readiness

### Objectives
- Convert validated outputs into publishable, repeatable project artifacts.

### Required Deliverables
- README updates including:
  - Ablation table with baseline vs candidate deltas.
  - Failure taxonomy A-G summary table.
- Experiment documentation updates:
  - Runbook commands.
  - Promotion criteria and rollback conditions.

### Exit Criteria
- Docs fully reflect promoted behavior.
- Known limitations and non-goals are explicit.

## Suggested Work Order
1. Complete Phase 3 forensics and finalize A-G labels.
2. Implement one intervention for the dominant label.
3. Run Phase 5 ablation and decide GO/NO-GO.
4. If GO, optionally test second intervention.
5. Integrate RAGAS faithfulness + answer relevancy and finalize docs.

## Stop Conditions
- Stop patching when target bucket does not move after a minimal intervention.
- Stop and reassess if synthetic accuracy or guardrails regress.
- Do not stack interventions until isolated attribution is complete.
