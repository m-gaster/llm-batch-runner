LLM Batch Runner
=================

Durable, async batched prompting for LLMs with retries, progress tracking, and resumability via a lightweight SQLite database.

Key features
- Async concurrency with backoff retries
- Durable progress stored in SQLite (resume on rerun)
- Pluggable worker: bring your own function or use a Pydantic AI + OpenRouter worker
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

Notes
- The library uses SQLAlchemy (async) with a simple `jobs` table and stores `pending|inflight|done|failed` states.
- Results from `prompt_map` are ordered by original prompt index.
