import json
import hashlib
from typing import Iterable, List, Optional
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

DB_URL_DEFAULT = "sqlite+aiosqlite:///runs.db"

# ---------- minimal persistence helpers (SQLAlchemy Core, async) ----------
CREATE_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS jobs(
  key TEXT PRIMARY KEY,
  idx INTEGER,
  prompt TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',  -- pending|inflight|done|failed
  result TEXT,
  error TEXT,
  attempts INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


def key_for(p: str) -> str:
    """Compute a deterministic key for a prompt.

    Uses SHA-1 over the UTF-8 bytes of the prompt to generate a hex string
    suitable for use as a primary key.

    Args:
        p: Prompt text.

    Returns:
        Hex-encoded SHA-1 digest for the prompt.
    """
    return hashlib.sha1(p.encode("utf-8")).hexdigest()


async def init_db(engine):
    """Initialize the jobs table and indexes if they do not exist.

    Executes the SQL statements in ``CREATE_SQL`` in a single transactional
    begin block. Safe to call multiple times.

    Args:
        engine: Async SQLAlchemy engine.
    """
    async with engine.begin() as conn:
        for stmt in CREATE_SQL.strip().split(";"):
            s = stmt.strip()
            if s:
                await conn.execute(text(s))


async def seed(engine, prompts: Iterable[str]):
    """Insert pending jobs for each prompt, ignoring duplicates by key.

    Args:
        engine: Async SQLAlchemy engine.
        prompts: Iterable of prompt strings to enqueue.
    """
    rows = [{"k": key_for(p), "i": i, "p": p} for i, p in enumerate(prompts)]
    if not rows:
        return
    async with engine.begin() as c:
        await c.execute(
            text(
                "INSERT OR IGNORE INTO jobs(key,idx,prompt,status,attempts) "
                "VALUES (:k,:i,:p,'pending',0)"
            ),
            rows,
        )


async def select_to_run(engine, keys: List[str]):
    """Select non-done jobs for the given keys, ordered by original index.

    Args:
        engine: Async SQLAlchemy engine.
        keys: List of job keys to consider.

    Returns:
        List of tuples ``(key, prompt)`` for jobs that are not yet done.
    """
    if not keys:
        return []
    marks = ",".join([f":k{i}" for i in range(len(keys))])
    params = {f"k{i}": k for i, k in enumerate(keys)}
    sql = f"SELECT key,prompt FROM jobs WHERE key IN ({marks}) AND status!='done' ORDER BY idx"
    async with engine.connect() as c:
        return list((await c.execute(text(sql), params)).all())


async def set_inflight(engine, key: str):
    """Mark a job as inflight and increment its attempt counter.

    Args:
        engine: Async SQLAlchemy engine.
        key: Job key to update.
    """
    async with engine.begin() as c:
        await c.execute(
            text("UPDATE jobs SET status='inflight', attempts=attempts+1 WHERE key=:k"),
            {"k": key},
        )


async def set_done(engine, key: str, result: str):
    """Mark a job as done and store its result, clearing any previous error.

    Args:
        engine: Async SQLAlchemy engine.
        key: Job key to update.
        result: Serialized result payload to store.
    """
    async with engine.begin() as c:
        await c.execute(
            text("UPDATE jobs SET status='done', result=:r, error=NULL WHERE key=:k"),
            {"k": key, "r": result},
        )


async def set_failed(engine, key: str, err: str):
    """Mark a job as failed and persist a truncated error message.

    Args:
        engine: Async SQLAlchemy engine.
        key: Job key to update.
        err: Error string; truncated to 1000 chars for storage.
    """
    async with engine.begin() as c:
        await c.execute(
            text("UPDATE jobs SET status='failed', error=:e WHERE key=:k"),
            {"k": key, "e": err[:1000]},
        )


async def count_status(engine):
    """Return a mapping of job status to row count.

    Args:
        engine: Async SQLAlchemy engine.

    Returns:
        Dict of ``{status: count}`` summarizing the jobs table.
    """
    async with engine.connect() as c:
        rows = (
            await c.execute(text("SELECT status,COUNT(1) FROM jobs GROUP BY status"))
        ).all()
        return {s: n for s, n in rows}


async def export_jsonl(db_url: str, out: str = "results.jsonl"):
    """Export all done jobs to a JSONL file.

    Each line contains an object with ``idx``, ``prompt`` and ``result``.

    Args:
        db_url: Database URL to read from.
        out: Output path for the JSONL file.
    """
    eng = create_async_engine(db_url, future=True)
    try:
        async with eng.connect() as c:
            rows = (
                await c.execute(
                    text(
                        "SELECT idx,prompt,result FROM jobs WHERE status='done' ORDER BY idx"
                    )
                )
            ).all()
            with open(out, "w", encoding="utf-8") as f:
                for i, p, r in rows:
                    f.write(
                        json.dumps(
                            {"idx": i, "prompt": p, "result": r}, ensure_ascii=False
                        )
                        + "\n"
                    )
        print(f"Exported {len(rows)} rows to {out}")
    finally:
        await eng.dispose()


# New: fetch results to return from prompt_map without writing to disk
async def fetch_results(engine, keys: Optional[List[str]] = None):
    """Fetch done job rows (optionally filtered) as a list of dicts.

    Args:
        engine: Async SQLAlchemy engine.
        keys: Optional list of job keys to include; if omitted, returns all done rows.

    Returns:
        List of dicts with keys ``idx``, ``prompt``, and ``result``, ordered by ``idx``.
    """
    sql = "SELECT idx,prompt,result FROM jobs WHERE status='done'"
    params = {}
    if keys:
        marks = ",".join([f":k{i}" for i in range(len(keys))])
        sql += f" AND key IN ({marks})"
        params = {f"k{i}": k for i, k in enumerate(keys)}
    sql += " ORDER BY idx"
    async with engine.connect() as c:
        rows = (await c.execute(text(sql), params)).all()
        return [{"idx": i, "prompt": p, "result": r} for (i, p, r) in rows]
