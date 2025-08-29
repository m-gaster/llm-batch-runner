import json
import os
from typing import Any, List, Optional
from warnings import warn

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine


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


def derive_results_db_url(
    progress_url: str, explicit_results_url: Optional[str]
) -> str:
    """Derive a safe default results DB URL from the progress DB URL."""
    if explicit_results_url:
        if explicit_results_url == progress_url:
            warn(
                "results_db_url matches the progress DB URL; deriving a separate results DB instead."
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
        results_engine = create_async_engine(results_db_url, future=True)
        try:
            async with results_engine.begin() as conn:
                # Recreate the table each run to ensure schema matches current version
                await conn.execute(text("DROP TABLE IF EXISTS results"))
                await conn.execute(
                    text(
                        """
                        CREATE TABLE results (
                            idx INTEGER NOT NULL,
                            key TEXT NOT NULL,
                            prompt TEXT,
                            status TEXT,
                            result TEXT,
                            PRIMARY KEY (idx, key)
                        )
                        """
                    )
                )

                # Bulk insert rows
                payload = []
                for r in results:
                    # Trust the idx provided by the result rows; if missing, try to
                    # derive from the prompts list (fallback -1 if not found).
                    idx = r.get("idx")
                    if idx is None:
                        # Fallback: take first occurrence index if present
                        try:
                            idx = prompts.index(r.get("prompt", ""))
                        except Exception:
                            idx = -1
                    payload.append(
                        {
                            "idx": idx,
                            "key": r.get("key"),
                            "prompt": r.get("prompt"),
                            "status": r.get("status"),
                            "result": r.get("result"),
                        }
                    )
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
