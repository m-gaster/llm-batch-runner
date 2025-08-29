LLM Batch Runner
=================

Durable, async batched prompting for LLMs with retries, progress tracking, and resumability via a lightweight SQLite database.

Key features
- Async concurrency with backoff retries
- Durable progress stored in SQLite (resume on rerun)
- Pluggable worker: bring your own function or use a Pydantic AI + OpenRouter worker
- Optional structured outputs via Pydantic `result_type`
- Choose return shape: unique prompts only or expanded to original input length
- Return results in‑memory or export to JSONL

Requirements
- Python 3.12+

Installation
`uv add llm-batch-runner git+https://github.com/m-gaster/llm-batch-runner`

Quick start
The simplest way is to rely on environment variables (`.env` works too):

```
# .env
MODEL=openai/gpt-4o-miniI
OPENROUTER_API_KEY=sk-or-...
```

Then run a small script:

```python
import asyncio
from llm_batch_runner.main import prompt_map

prompts = [
    "Summarize: The quick brown fox jumps over the lazy dog.",
    "Give me 3 bullet points on why the sky appears blue.",
    "Rewrite this in pirate speak: Hello, friend!",
]

async def main():
    results = await prompt_map(prompts, concurrency=16, teardown=True)
    for row in results:
        print(row)

asyncio.run(main())
```

This will:
- Create (or reuse) a SQLite DB at `runs.db`
- Run prompts concurrently with retries
- Print progress and return ordered results
- Remove the DB on exit when `teardown=True`

Other ways to provide a worker
- Direct params (OpenRouter):

```python
results = await prompt_map(
    prompts,
    model_name="openai/gpt-4o-mini",
    openrouter_api_key="sk-or-...",
)
```

- Custom async worker:

```python
async def echo_worker(p: str) -> str:
    return p.upper()

results = await prompt_map(prompts, worker=echo_worker)
```

Structured outputs
You can ask the built-in Pydantic AI worker to return structured data by passing a Pydantic model class as `result_type`. The `result` stored in the DB and returned from `prompt_map` will be a JSON string matching your schema.

```python
from pydantic import BaseModel

class Bullets(BaseModel):
    points: list[str]

results = await prompt_map(
    prompts,
    model_name="openai/gpt-4o-mini",
    openrouter_api_key="sk-or-...",
    result_type=Bullets,
)
# each row["result"] is a JSON string for Bullets
```

Output shape and exporting
By default, `prompt_map` deduplicates identical prompts internally. You can control the returned shape via `output_shape`:

```python
results_unique = await prompt_map(prompts, output_shape="unique")   # default
results_orig   = await prompt_map(prompts, output_shape="original")
```

The results DB (`*-results.db`) mirrors the chosen `output_shape` for that call. With `original`, duplicate prompts are written as multiple rows (distinguished by their `idx`).

Exporting to JSONL
If you prefer a file output, you can export after a run:

```python
from llm_batch_runner.utils import export_jsonl, DB_URL_DEFAULT
import asyncio

asyncio.run(export_jsonl(DB_URL_DEFAULT, out="results.jsonl"))
```

Tuning
- `concurrency`: maximum simultaneous jobs (default 32)
- `max_attempts`: total attempts per job with exponential backoff (default 8)
- `db_url`: override SQLite location, e.g. `sqlite+aiosqlite:///my_runs.db`
- `progress_every`: print frequency for progress updates (default 200)
- `output_shape`: `"unique"` (default) returns one row per unique prompt; `"original"` expands results back to the input length, duplicating rows for duplicate prompts. Missing/failed prompts appear with `status="missing"` and `result=None` in dict/Polars forms.

Notes
- The library uses SQLAlchemy (async) with a simple `jobs` table and stores `pending|inflight|done|failed` states.
- With `output_shape="unique"`, results are ordered by the first occurrence index of each unique prompt. With `output_shape="original"`, results are one-per-input in original order.
