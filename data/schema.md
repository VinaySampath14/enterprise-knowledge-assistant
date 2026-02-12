# Enterprise Knowledge Assistant — Data Schemas (JSONL)

All processed datasets are stored as JSONL:
- One JSON object per line
- UTF-8
- No trailing commas
- Records must be JSON objects (dict-like)

---

## 1) data/processed/docs.jsonl  (Stage 1 Output)

**Purpose:** One record per source document (.rst file). Used as input for chunking.

### Required fields

| field        | type   | description |
|-------------|--------|-------------|
| id          | string | Stable unique document id. Example: `py-stdlib:json.rst` or `py-stdlib:os/path.rst` |
| module      | string | Logical module name from filename. Example: `json`, `pathlib`, `os.path` |
| source      | object | Information about where the doc came from |
| text        | string | Full raw text content of the .rst file |
| sha256      | string | SHA-256 hash of `text` for change detection |
| created_at  | string | ISO-8601 UTC timestamp when record was created |

### `source` object fields

| field   | type   | description |
|--------|--------|-------------|
| path   | string | Original file path (repo-relative or absolute, but be consistent) |
| type   | string | Always `"rst"` for this dataset |

### Example record

```json
{
  "id": "py-stdlib:json.rst",
  "module": "json",
  "source": {"path": "data/raw/python_stdlib/json.rst", "type": "rst"},
  "text": "... full rst content ...",
  "sha256": "ab12...ff",
  "created_at": "2026-02-12T10:15:00+00:00"
}

---

## 2) data/processed/chunks.jsonl  (Stage 2 Output)

Purpose: Chunked text units for embedding + retrieval.
One record per chunk.

### Required fields

| field        | type    | description |
|-------------|---------|-------------|
| chunk_id    | string  | Stable unique id. Example: `py-stdlib:json.rst#c0003` |
| doc_id      | string  | References docs.jsonl.id |
| module      | string  | Copied from parent document |
| text        | string  | Chunk text |
| start_char  | integer | Start character offset in original doc |
| end_char    | integer | End character offset in original doc |
| chunk_index | integer | 0-based index of chunk within document |
| created_at  | string  | ISO-8601 UTC timestamp |
| meta        | object  | Additional retrieval metadata |

### meta object fields

| field        | type    | description |
|-------------|---------|-------------|
| source_path | string  | Copy of source.path from docs.jsonl |
| heading     | string? | Optional nearest section heading (can be null) |
