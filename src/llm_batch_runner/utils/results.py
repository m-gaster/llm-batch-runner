import json
import os
from typing import Any, List, Optional
from warnings import warn

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from .db import key_for


async def export_jsonl(db_url: str, out: str = "results.jsonl"):
    """Export all done jobs to a JSONL file."""
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


def derive_results_db_url(progress_url: str, explicit_results_url: Optional[str]) -> str:
    """Derive a safe default results DB URL from the progress DB URL."""
    if explicit_results_url:
        if explicit_results_url == progress_url:
            warn(
                "results_db_url is identical to db_url; deriving a separate results DB instead."
            )
        else:
            return explicit_results_url

    url = make_url(progress_url)
    db_path: Optional[str] = url.database

    # If no file path (e.g., :memory: or non-file DB), fall back to a local file
    if not db_path or db_path == ":memory:":
        return "sqlite+aiosqlite:///results.db"

    # For file-based DBs, create a sibling "<name>-results.ext"
    base = os.path.basename(db_path)
    root, ext = os.path.splitext(base)
    res_name = f"{root}-results{ext or '.db'}"
    res_path = os.path.join(os.path.dirname(db_path), res_name)
    return str(url.set(database=res_path))


async def write_results_to_db(
    results_db_url: str, prompts: List[str], results: List[dict]
) -> None:
    """Persist final results to a separate results DB."""
    try:
        # Build key->idx mapping to preserve original order in the results DB
        idx_map = {key_for(p): i for i, p in enumerate(prompts)}

        results_engine = create_async_engine(results_db_url, future=True)
        try:
            async with results_engine.begin() as conn:
                # Create a very simple, stable schema for outputs
                await conn.execute(
                    text(
                        """
                        CREATE TABLE IF NOT EXISTS results (
                            idx INTEGER NOT NULL,
                            key TEXT PRIMARY KEY,
                            prompt TEXT,
                            status TEXT,
                            result TEXT
                        )
                        """
                    )
                )
                # Clear existing rows (fresh snapshot each run)
                await conn.execute(text("DELETE FROM results"))

                # Bulk insert rows
                payload = [
                    {
                        "idx": idx_map.get(r["key"], -1),
                        "key": r["key"],
                        "prompt": r["prompt"],
                        "status": r["status"],
                        "result": r["result"],
                    }
                    for r in results
                ]
                if payload:
                    await conn.execute(
                        text(
                            "INSERT INTO results (idx, key, prompt, status, result) "
                            "VALUES (:idx, :key, :prompt, :status, :result)"
                        ),
                        payload,
                    )
            print(f"Wrote {len(results)} rows to results DB: {results_db_url}")
        finally:
            await results_engine.dispose()
    except Exception as e:
        warn(f"Failed to write results to separate DB ({results_db_url}): {e!r}")


def shape_return(results: List[dict], return_dtype: str) -> Any:
    """Shape results into the requested return dtype."""
    if return_dtype == "list[dict]":
        return results
    elif return_dtype == "list[str]":
        return [r["result"] for r in results]
    elif return_dtype == "list[tuple[str,str]]":
        return [(r["prompt"], r["result"]) for r in results]
    elif return_dtype == "polars":
        try:
            import polars as pl  # type: ignore
        except Exception:
            raise RuntimeError(
                "Polars is required for return_dtype='polars'. Please install the 'polars' package."
            )
        return pl.DataFrame(results)
    else:
        raise ValueError(
            "Invalid return_dtype. Expected one of: 'list[dict]', 'list[str]', 'list[tuple[str,str]]', 'polars'"
        )


__all__ = [
    "export_jsonl",
    "derive_results_db_url",
    "write_results_to_db",
    "shape_return",
]

