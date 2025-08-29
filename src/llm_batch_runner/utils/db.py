import hashlib
from typing import Iterable, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

# Default progress DB URL
DB_URL_DEFAULT = "sqlite+aiosqlite:///.llm_batch_cache/runs.db"

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
    """Compute a deterministic key for a prompt."""
    return hashlib.sha1(p.encode("utf-8")).hexdigest()


async def init_db(engine: AsyncEngine) -> None:
    """Initialize the jobs table and indexes if they do not exist."""
    async with engine.begin() as conn:
        for stmt in CREATE_SQL.strip().split(";"):
            s = stmt.strip()
            if s:
                await conn.execute(text(s))


async def seed(engine: AsyncEngine, prompts: Iterable[str]) -> None:
    """Insert pending jobs for each prompt, ignoring duplicates by key."""
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


async def select_to_run(engine: AsyncEngine, keys: List[str]):
    """Select non-done jobs for the given keys, ordered by original index."""
    if not keys:
        return []
    marks = ",".join([f":k{i}" for i in range(len(keys))])
    params = {f"k{i}": k for i, k in enumerate(keys)}
    sql = f"SELECT key,prompt FROM jobs WHERE key IN ({marks}) AND status!='done' ORDER BY idx"
    async with engine.connect() as c:
        return list((await c.execute(text(sql), params)).all())


async def set_inflight(engine: AsyncEngine, key: str) -> None:
    """Mark a job as inflight and increment its attempt counter."""
    async with engine.begin() as c:
        await c.execute(
            text("UPDATE jobs SET status='inflight', attempts=attempts+1 WHERE key=:k"),
            {"k": key},
        )


async def set_done(engine: AsyncEngine, key: str, result: str) -> None:
    """Mark a job as done and store its result, clearing any previous error."""
    async with engine.begin() as c:
        await c.execute(
            text("UPDATE jobs SET status='done', result=:r, error=NULL WHERE key=:k"),
            {"k": key, "r": result},
        )


async def set_failed(engine: AsyncEngine, key: str, err: str) -> None:
    """Mark a job as failed and persist a truncated error message."""
    async with engine.begin() as c:
        await c.execute(
            text("UPDATE jobs SET status='failed', error=:e WHERE key=:k"),
            {"k": key, "e": err[:1000]},
        )


async def count_status(engine: AsyncEngine):
    """Return a mapping of job status to row count."""
    async with engine.connect() as c:
        rows = (
            await c.execute(text("SELECT status,COUNT(1) FROM jobs GROUP BY status"))
        ).all()
        return {s: n for s, n in rows}


async def fetch_results(engine: AsyncEngine, keys: Optional[List[str]] = None):
    """Fetch done job rows (optionally filtered) as a list of dicts."""
    sql = "SELECT idx,key,prompt,status,result FROM jobs WHERE status='done'"
    params = {}
    if keys:
        marks = ",".join([f":k{i}" for i in range(len(keys))])
        sql += f" AND key IN ({marks})"
        params = {f"k{i}": k for i, k in enumerate(keys)}
    sql += " ORDER BY idx"
    async with engine.connect() as c:
        rows = (await c.execute(text(sql), params)).all()
        return [
            {"idx": i, "key": k, "prompt": p, "status": s, "result": r}
            for (i, k, p, s, r) in rows
        ]


__all__ = [
    "DB_URL_DEFAULT",
    "CREATE_SQL",
    "key_for",
    "init_db",
    "seed",
    "select_to_run",
    "set_inflight",
    "set_done",
    "set_failed",
    "count_status",
    "fetch_results",
]

