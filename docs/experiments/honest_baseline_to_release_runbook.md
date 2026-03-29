# Honest Baseline to Release Runbook

## Purpose
Capture the safest execution order so progress is attributable, reproducible, and less brittle.

## Operating Rules (Non-Negotiable)
- One major intervention per candidate run.
- If no movement or regressions occur, revert immediately.
- Keep promoted metrics as reference points, not architecture truth.
- Guardrails remain primary promotion criteria.

## Step 0.1: Pin MLflow Tracking URI (Required Once Per Shell)
- Set one fixed tracking URI before running any eval/phase scripts.
- Recommended repo-local location:
  - `MLFLOW_TRACKING_URI=file:./artifacts/mlflow`
- Why this matters:
  - MLflow defaults to `./mlruns` relative to current working directory.
  - Running scripts from different directories can split runs across multiple stores.
- PowerShell (from repo root):
  - `$env:MLFLOW_TRACKING_URI = "file:./artifacts/mlflow"`
- CMD (from repo root):
  - `set MLFLOW_TRACKING_URI=file:./artifacts/mlflow`
- Optional (persist for future Windows sessions):
  - `setx MLFLOW_TRACKING_URI "file:./artifacts/mlflow"`

## Step 0: Freeze Current State
- Keep currently promoted manual and synthetic numbers as reference only.
- Do not treat current confidence logic as final architecture.
- Goal: create a stable before/after comparison point.

## Step 1: Build Honest Baseline (H0)
- Refactor confidence gate to only evaluate:
  - evidence strength
  - evidence consistency
  - retrieval mismatch/coherence
- Remove intent-style query rules from confidence gate:
  - out-of-domain checks
  - python-general checks
  - broad ambiguity intent checks
- Keep gate retrieval-confidence only.
- Run full evaluation and save as Honest Baseline H0.
- Note: short-term score can drop; trustworthiness and modularity matter more here.

## Step 2: Add Generalization Check Before New Features
- Build a small paraphrased holdout set that does not mirror current manual wording.
- Evaluate H0 on:
  - manual
  - synthetic
  - holdout paraphrases
- Goal: verify H0 is less brittle than query-pattern logic.

## Step 3: Add Intent Classifier Upstream
- Add a separate intent classifier module before retrieval.
- Start with conservative labels:
  - in_domain
  - out_of_domain
  - python_general
  - ambiguous
- Only in_domain should flow to retrieval pipeline in v1.
- Re-evaluate against H0.
- Accept only if primary guardrails stay safe.

## Step 4: Add Hybrid Retrieval
- Add BM25 plus dense fusion in retriever.
- Do not modify intent classifier and confidence gate in the same run.
- Re-run evaluation and changed-ID attribution.
- Promote only if target errors improve without synthetic regressions.

## Step 5: Add Reranker
- Add cross-encoder reranker after retrieval.
- Keep this change isolated.
- Evaluate against latest promoted run.

## Step 6: Add Evaluation Hardening
- Add RAGAS faithfulness and answer relevancy.
- Add LLM-as-judge as a secondary signal only.
- Keep existing guardrails as primary promotion gate.

## Step 7: Finalize Docs and Release
- Update README with ablation table.
- Publish failure taxonomy A-G.
- Document known limits and rollback conditions.

## Why This Order
- First remove overfit logic and establish a truthful baseline.
- Then add modules one at a time so credit/blame is clear.
- Avoid multi-change runs that hide causal impact.

## Execution Checklist Per Candidate
- Define intervention scope and non-goals.
- Run full eval suite.
- Compare against promoted baseline.
- Inspect changed IDs and confusion buckets.
- Apply GO/NO-GO decision.
- Revert immediately on no movement or regressions.

## Artifact Hygiene (After Each Candidate Batch)
- Run dry-run cleanup first:
  - `python scripts/experiments/cleanup_phase_runs.py --phase-dir artifacts/experiments/phase0 --archive-dir artifacts/archive/experiments/phase0`
- Execute cleanup after reviewing plan output:
  - `python scripts/experiments/cleanup_phase_runs.py --phase-dir artifacts/experiments/phase0 --archive-dir artifacts/archive/experiments/phase0 --execute`
- Retention behavior:
  - promoted baseline runs are always protected
  - newest baseline/candidate runs are retained
  - older run folders are moved to archive, not deleted

## Comparison Table (README-Ready)
- Tag each eval/phase run with a stable version label:
  - `--ablation-version H0_clean_gate`
  - `--ablation-version v3_intent_clf`
- Generate a comparison table from summary files:
  - `python scripts/compare_ablation.py --source files`
- Generate a comparison table from MLflow runs (requires tagged runs):
  - `python scripts/compare_ablation.py --source mlflow --mlflow-experiment eka-eval`
- Add outsider-friendly descriptions and GO/NO-GO context:
  - `python scripts/compare_ablation.py --source files --metadata-file artifacts/baselines/ablation_metadata.example.json`
- Auto-fill Decision and Run ID from phase-gate MLflow runs:
  - default experiment: `eka-phase-gate`
  - disable if needed: `--disable-phase-gate-autofill`
  - note: phase-gate runs must include `--ablation-version <label>` to map rows correctly
- Metadata template (edit per version):
  - `artifacts/baselines/ablation_metadata.example.json`
- Default output path:
  - `artifacts/baselines/ablation_comparison_table.md`

## One-Command Batch Runner
- Run eval + phase-gate + table generation in one command:
  - `python scripts/experiments/run_bundle.py --ablation-version v3_intent_clf`
- Windows shortcut:
  - `scripts\run_bundle.bat v3_intent_clf`
- Optional flags:
  - `--skip-eval`
  - `--skip-phase-gate`
  - `--skip-table`
