# Enterprise Knowledge Assistant

A Retrieval-Augmented Generation (RAG) API for answering questions about the Python standard library. It combines dense vector retrieval, BM25 sparse retrieval, cross-encoder reranking, and a multi-stage confidence gate to return answers, ask for clarification, or refuse — prioritizing correctness over coverage.

## What It Does

- **Corpus**: Python stdlib RST documentation, chunked and indexed
- **Retrieval**: Hybrid dense (FAISS/all-MiniLM-L6-v2) + BM25, with optional cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
- **Decision logic**: Intent classification → confidence gate → GPT-4o-mini generation → post-generation refusal override
- **Response types**: `answer`, `clarify`, or `refuse` — never a hallucinated answer
- **Observability**: Every query logged to `logs/queries.jsonl`; `GET /stats` aggregates in real time

## Pipeline

```
Query
  └─► Intent Classification  (in_domain / out_of_domain / ambiguous)
        └─► [early refuse if clearly out-of-domain]
        └─► Retrieval  (dense · BM25 · hybrid RRF)
              └─► Cross-encoder Reranking  (conditional, low-margin only)
                    └─► Confidence Gate  (answer · clarify · refuse)
                          └─► [generation if answer]
                                └─► Post-gen Refusal Override  (safety net)
                                      └─► Response
```

## Prerequisites

- Python 3.11
- `OPENAI_API_KEY` — required for generation
- ~500 MB disk for model caches (SentenceTransformer + cross-encoder download on first run)

## Quick Start

### 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure API key

```bash
export OPENAI_API_KEY="sk-..."   # Windows: $env:OPENAI_API_KEY="sk-..."
```

Or create a `.env` file:

```env
OPENAI_API_KEY=sk-...
```

### 3. Validate or build artifacts

```bash
python scripts/validate_docs.py
python scripts/validate_chunks.py
python scripts/validate_index.py
```

If validation fails or artifacts are missing, build them:

```bash
python scripts/build_docs.py     # Stage 1: RST → data/processed/docs.jsonl
python scripts/build_chunks.py   # Stage 2: docs → data/processed/chunks.jsonl
python scripts/build_index.py    # Stage 3: chunks → indexes/faiss.index + meta.jsonl
```

### 4. Run the API

```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## API

### `POST /query`

```json
{ "query": "How do I open a sqlite3 connection?" }
```

Optional header: `X-Request-ID`

**Response fields**

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"answer"` \| `"clarify"` \| `"refuse"` | Decision outcome |
| `answer` | string | Response text (empty string when type is `refuse`) |
| `confidence` | float | Top retrieval score used by the gate |
| `sources` | array | Chunk metadata (module, score, path) |
| `citations` | array | Full citation objects with char offsets and headings |
| `meta` | object | Gate rationale, intent label, latency breakdown, reranker flags |

**Response types**

- `answer` — retrieval confidence is high; context was used to generate a grounded response
- `clarify` — query is ambiguous or confidence is borderline; asks for more specificity
- `refuse` — query is out-of-domain, confidence is too low, or generation returned a refusal

**Error codes**

| Code | Cause |
|------|-------|
| 400 | Empty or whitespace query |
| 500 | Internal pipeline error |
| 502 | OpenAI generation failure |
| 503 | Pipeline not initialized (check `/health`) |

### `GET /health`

Returns `{ status, pipeline_loaded, dependencies, startup_errors }`. Status is `"ok"` or `"degraded"`.

### `GET /stats`

Returns aggregate counts and averages computed live from `logs/queries.jsonl`: total queries, type distribution, avg confidence, avg latency, avg groundedness.

## Configuration

Key fields in `config.yaml`:

| Field | Default | Description |
|-------|---------|-------------|
| `retrieval.mode` | `dense` | `dense`, `bm25`, or `hybrid` |
| `retrieval.top_k` | `5` | Chunks returned to the gate and generator |
| `reranker.enabled` | `true` | Enable cross-encoder reranking |
| `reranker.strategy` | `low_margin_only` | Only rerank when top-2 score margin is narrow |
| `reranker.low_margin_threshold` | `0.15` | Margin threshold that triggers reranking |
| `confidence.threshold_high` | `0.4` | Scores above this → `answer` |
| `confidence.threshold_low` | `0.25` | Scores below this → `refuse` |
| `generation.model` | `gpt-4o-mini` | OpenAI model for response generation |
| `generation.temperature` | `0.0` | Deterministic generation |
| `logging.enabled` | `true` | Log all queries to `logs/queries.jsonl` |

## Docker

```bash
docker build -t enterprise-knowledge-assistant:latest .
docker run --rm -p 8000:8000 -e OPENAI_API_KEY=sk-... enterprise-knowledge-assistant:latest
```

Or with compose (mounts `./logs` into the container):

```bash
docker compose up --build
```

## Key Scripts

**Build & validate**
```bash
python scripts/build_docs.py
python scripts/build_chunks.py
python scripts/build_index.py
python scripts/validate_docs.py
python scripts/validate_chunks.py
python scripts/validate_index.py
```

**Debug**
```bash
python scripts/debug/query_pipeline.py "<query>"   # full pipeline trace
python scripts/debug/query_retrieve.py "<query>"   # retrieval only
python scripts/debug/query_gate.py "<query>"       # gate decision only
```

**Evaluation & experiments**
```bash
python scripts/run_eval.py
python scripts/experiments/run_eval_v2_synthetic.py
python scripts/experiments/run_phase0_baseline.py
python scripts/experiments/run_phase_gate.py
python scripts/experiments/run_bundle.py
python scripts/experiments/run_ragas_eval.py
```

## Experiments & Results

The system was developed through 12 iterative experiments. Each version was evaluated on three held-out sets:

- **Manual** (33 queries) — human-curated across five categories
- **Synthetic** (22 queries) — LLM-generated covering in-domain, near-domain, and out-of-domain
- **Holdout** (20 queries) — paraphrase set held out from all tuning

**False Answer**: expected refuse, predicted answer (safety-critical).
**False Refusal**: expected answer, predicted refuse (quality-impacting).

### Decision Quality Ablation

| Version | What Changed | Manual | Synth | Holdout | False Ans | False Ref | Safety | Decision |
|---------|-------------|--------|-------|---------|-----------|-----------|--------|----------|
| baseline | Confidence-gate cleanup | 60.6% | 90.9% | 65.0% | 6.1% | 6.1% | fail | REF |
| v1 | Intent layer — upstream conservative routing | 69.7% | 90.9% | 70.0% | 0.0% | 6.1% | warn | GO |
| v2 | Post-gen refusal soften to clarify (broad) | 54.6% | 77.3% | 60.0% | 0.0% | 0.0% | pass | **NO-GO** — large accuracy regression |
| v3 | Narrow grounded rescue fallback | 69.7% | 90.9% | 70.0% | 0.0% | 6.1% | warn | GO |
| v4 | Symbol-anchor rescue refinement | 69.7% | 90.9% | 70.0% | 0.0% | 6.1% | warn | HOLD — no change vs v3 |
| v5 | Retrieval rerank for explicit dotted symbols | 72.7% | 90.9% | 70.0% | 0.0% | 3.0% | pass | GO — false-refusal halved |
| v6 | Citation-retry post-generation | 72.7% | 90.9% | 70.0% | 0.0% | 3.0% | pass | HOLD — no improvement |
| v7 | Hybrid retrieval (initial RRF scoring) | 54.6% | 63.6% | 60.0% | 0.0% | 30.3% | fail | **NO-GO** — RRF scores mismatched confidence thresholds, near-universal refusals |
| v8 | Hybrid retrieval (RRF rank + dense-scale score) | 72.7% | 90.9% | 70.0% | 0.0% | 3.0% | pass | HOLD — parity with v5, no uplift |
| v9 | Cross-encoder reranker (ms-marco-MiniLM-L-6-v2), always-on | 75.8% | 90.9% | 75.0% | 0.0% | 0.0% | pass | GO — new champion |
| v10 | Conditional reranking (low_margin_only, threshold 0.05) | 72.7% | 90.9% | 70.0% | 0.0% | 0.0% | pass | HOLD — slight regression vs v9 |
| v11 | Conditional reranking (low_margin_only, threshold 0.15) | **75.8%** | **90.9%** | **75.0%** | 0.0% | 0.0% | pass | **GO — champion; preserved quality, improved latency** |
| v12 | Lock-in confirmation run | 75.8% | 90.9% | 75.0% | 0.0% | 0.0% | pass | Final confirmation |

### Answer & Retrieval Quality (Champion Versions)

Metrics computed via RAGAS on answer-producing predictions only.

| Version | Faithfulness | Answer Relevancy | Context Precision | Keep Rate | Latency M/S/H (ms) |
|---------|-------------|-----------------|-------------------|-----------|---------------------|
| v9 | — | — | — | — | 2006 / 1434 / 1464 |
| v10 | — | — | — | — | 2012 / 1484 / 1370 |
| v11 | **0.914** | **0.911** | 0.757 | 0.452 | 1608 / 1438 / 1273 |
| v12 | 0.914 | 0.911 | 0.757 | 0.452 | **1602 / 1331 / 1213** |

Latency columns: M = Manual set, S = Synthetic set, H = Holdout set.

### Confidence Threshold Calibration

Top retrieval score distribution by expected response type (67 samples):

| Expected Type | n | Mean Score | p25 | p75 |
|--------------|---|-----------|-----|-----|
| answer | 29 | 0.665 | 0.610 | 0.740 |
| clarify | 11 | 0.580 | 0.578 | 0.654 |
| refuse | 27 | 0.250 | 0.177 | 0.239 |

Current thresholds: `threshold_high=0.4`, `threshold_low=0.25`. Calibration proposed `0.62/0.41` but was not adopted — simulated accuracy gain (+11 pp) came at the cost of synthetic set regression.

## Evaluation

```bash
# Basic eval against eval/questions.jsonl
python scripts/run_eval.py

# Synthetic query eval (generates + evaluates)
python scripts/experiments/run_eval_v2_synthetic.py

# RAGAS metrics (faithfulness, answer relevancy, context precision)
python scripts/experiments/run_ragas_eval.py
```

Outputs: `eval/results.jsonl`, `eval/report.json`, `eval/report.md`

Metrics computed: accuracy by expected type, false-answer rate, false-refusal rate, groundedness (lexical overlap), avg latency.

MLflow tracking data is stored in `artifacts/mlflow/`.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

11 test files covering: confidence gate, retriever, intent classifier, prompt formatting, cross-encoder reranker, ingest, index artifact integrity, and the post-generation refusal override guard.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `/health` returns `degraded` | Missing artifacts or API key | Check `startup_errors` field; rebuild index or set `OPENAI_API_KEY` |
| `POST /query` → 400 | Empty query | Send a non-empty `query` string |
| `POST /query` → 502 | OpenAI API error | Check API key, network, and OpenAI status |
| `GET /stats` not updating | Logging disabled | Set `logging.enabled: true` in `config.yaml` |
| All queries returning `refuse` | Score scale mismatch | Check `retrieval.mode`; if using `hybrid`, verify `confidence.threshold_high/low` are calibrated for RRF scores |
| Models not downloading | Network/cache issue | Check `models/cache/`; delete and retry, or set `HF_HOME` to a writable path |
