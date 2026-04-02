# Enterprise Knowledge Assistant

Enterprise Knowledge Assistant is a RAG API for Python standard-library help.

It uses:
- FAISS retrieval over chunked stdlib docs
- A confidence gate that returns `answer`, `clarify`, or `refuse`
- GPT-based response generation (when enabled)
- FastAPI endpoints for query serving, health, and usage stats

## What Matters

- Corpus source: `data/raw/python_stdlib`
- Query logs: `logs/queries.jsonl`
- Main API: `src/api/main.py`
- Runtime config: `config.yaml`

Behavior:
- `POST /query` logs each request when `logging.enabled: true`.
- `GET /stats` summarizes what is already present in `logs/queries.jsonl`.

## Quick Start (Local)

### 1) Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Configure key

Set one of these:

```powershell
$env:OPENAI_API_KEY="sk-..."
```

or create a local `.env` file:

```env
OPENAI_API_KEY=sk-...
```

### 3) Validate artifacts

```powershell
python scripts/validate_docs.py
python scripts/validate_chunks.py
python scripts/validate_index.py
```

If validation fails or files are missing, rebuild:

```powershell
python scripts/build_docs.py
python scripts/build_chunks.py
python scripts/build_index.py
```

### 4) Run API

```powershell
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

## API

### `POST /query`

Request:

```json
{
  "query": "How do I open a sqlite3 connection?"
}
```

Notes:
- Empty/whitespace queries return `400`.
- Optional header: `X-Request-ID`.
- Response includes `type`, `answer`, `confidence`, `sources`, `citations`, and `meta`.

### `GET /health`

Returns runtime status and dependency readiness.

### `GET /stats`

Returns aggregate counts and averages computed from query logs.

## Docker

Build and run:

```powershell
docker build -t enterprise-knowledge-assistant:latest .
docker run --rm -p 8000:8000 -e OPENAI_API_KEY=sk-... enterprise-knowledge-assistant:latest
```

Or with compose:

```powershell
docker compose up --build
```

`docker-compose.yml` mounts local `./logs` into `/app/logs`.

## Key Scripts

Build/validate:
- `python scripts/build_docs.py`
- `python scripts/build_chunks.py`
- `python scripts/build_index.py`
- `python scripts/validate_docs.py`
- `python scripts/validate_chunks.py`
- `python scripts/validate_index.py`

Debug:
- `python scripts/debug/query_retrieve.py "<query>"`
- `python scripts/debug/query_gate.py "<query>"`
- `python scripts/debug/query_pipeline.py "<query>"`

Eval/experiments (common):
- `python scripts/run_eval.py`
- `python scripts/experiments/run_eval_v2_synthetic.py ...`
- `python scripts/experiments/run_phase0_baseline.py ...`
- `python scripts/experiments/run_phase_gate.py ...`
- `python scripts/experiments/run_bundle.py ...`

Windows helpers:
- `scripts/run_api.bat`
- `scripts/run_eval.bat`
- `scripts/run_bundle.bat`

## Testing

`requirements.txt` is runtime-focused. For local tests, install pytest first:

```powershell
pip install pytest
pytest tests/ -v
```

## Troubleshooting

- API is `degraded`:
  Check `GET /health` for startup errors (missing artifacts or missing key).
- `POST /query` returns `400`:
  Query is empty.
- `POST /query` returns `502`:
  Generation backend/API key/network issue.
- `/stats` not changing:
  Ensure `logging.enabled: true` and send traffic through `POST /query`.
