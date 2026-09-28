import hashlib
from collections.abc import Iterable, Sequence
from typing import cast

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from .types import JobRow, JobStatus

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
# TODO: use sqlalchemy to define the above


def key_for(p: str) -> str:
    """Compute a deterministic key for a prompt."""
    return hashlib.sha1(p.encode("utf-8")).hexdigest()


def _validate_job_status(status: str) -> JobStatus:
    valid_statuses: tuple[JobStatus, ...] = ("pending", "inflight", "done", "failed")
    if status not in valid_statuses:
        raise ValueError(f"Unexpected job status {status!r}.")
    return cast(JobStatus, status)


def _job_row_from_db_row(
    idx: int,
    key: str,
    prompt: str,
    status: str,
    result: str | None,
    error: str | None,
    attempts: int,
) -> JobRow:
    return JobRow(
        idx=idx,
        key=key,
        prompt=prompt,
        status=_validate_job_status(status),
        result=result,
        error=error,
        attempts=attempts,
    )


def _build_key_params(keys: Sequence[str]) -> tuple[str, dict[str, str]]:
    marks = ",".join(f":k{i}" for i in range(len(keys)))
    params = {f"k{i}": key for i, key in enumerate(keys)}
    return marks, params


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


async def select_to_run(engine: AsyncEngine, keys: Sequence[str]) -> list[JobRow]:
    """Select non-done jobs for the given keys, ordered by original index."""
    if not keys:
        return []
    marks, params = _build_key_params(keys)
    sql = (
        "SELECT idx,key,prompt,status,result,error,attempts "
        f"FROM jobs WHERE key IN ({marks}) AND status!='done' ORDER BY idx"
    )
    async with engine.connect() as c:
        rows = list((await c.execute(text(sql), params)).all())
        return [
            _job_row_from_db_row(idx, key, prompt, status, result, error, attempts)
            for idx, key, prompt, status, result, error, attempts in rows
        ]


async def set_inflight(engine: AsyncEngine, key: str) -> None:
    """Mark a job as inflight and increment its attempt counter."""
    async with engine.begin() as c:
        result = await c.execute(
            text("UPDATE jobs SET status='inflight', attempts=attempts+1 WHERE key=:k"),
            {"k": key},
        )
        if result.rowcount != 1:
            raise RuntimeError(
                f"Expected to mark exactly one job inflight for key {key!r}."
            )


async def set_done(engine: AsyncEngine, key: str, result: str) -> None:
    """Mark a job as done and store its result, clearing any previous error."""
    async with engine.begin() as c:
        db_result = await c.execute(
            text("UPDATE jobs SET status='done', result=:r, error=NULL WHERE key=:k"),
            {"k": key, "r": result},
        )
        if db_result.rowcount != 1:
            raise RuntimeError(
                f"Expected to mark exactly one job done for key {key!r}."
            )


async def set_failed(engine: AsyncEngine, key: str, err: str) -> None:
    """Mark a job as failed and persist a truncated error message."""
    async with engine.begin() as c:
        result = await c.execute(
            text("UPDATE jobs SET status='failed', result=NULL, error=:e WHERE key=:k"),
            {"k": key, "e": err[:1000]},
        )
        if result.rowcount != 1:
            raise RuntimeError(
                f"Expected to mark exactly one job failed for key {key!r}."
            )


async def count_status(engine: AsyncEngine) -> dict[JobStatus, int]:
    """Return a mapping of job status to row count."""
    async with engine.connect() as c:
        rows = (
            await c.execute(text("SELECT status,COUNT(1) FROM jobs GROUP BY status"))
        ).all()
        counts: dict[JobStatus, int] = {}
        for status, count in rows:
            counts[_validate_job_status(status)] = int(count)
        return counts


async def fetch_job_rows(
    engine: AsyncEngine,
    keys: Sequence[str] | None = None,
) -> list[JobRow]:
    """Fetch job rows (optionally filtered) ordered by original index."""
    sql = "SELECT idx,key,prompt,status,result,error,attempts FROM jobs"
    params: dict[str, str] = {}
    if keys is not None:
        if not keys:
            return []
        marks, params = _build_key_params(keys)
        sql += f" WHERE key IN ({marks})"
    sql += " ORDER BY idx"
    async with engine.connect() as c:
        rows = (await c.execute(text(sql), params)).all()
        return [
            _job_row_from_db_row(idx, key, prompt, status, result, error, attempts)
            for idx, key, prompt, status, result, error, attempts in rows
        ]


async def fetch_results(
    engine: AsyncEngine,
    keys: Sequence[str] | None = None,
) -> list[dict[str, int | str | None]]:
    """Fetch done jobs as legacy plain dictionaries, optionally filtered by key."""
    sql = "SELECT idx,key,prompt,status,result FROM jobs WHERE status='done'"
    params: dict[str, str] = {}
    if keys:
        marks, params = _build_key_params(keys)
        sql += f" AND key IN ({marks})"
    sql += " ORDER BY idx"
    async with engine.connect() as c:
        rows = (await c.execute(text(sql), params)).all()
        return [
            {
                "idx": idx,
                "key": key,
                "prompt": prompt,
                "status": status,
                "result": result,
            }
            for idx, key, prompt, status, result in rows
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
    "fetch_job_rows",
    "fetch_results",
]
