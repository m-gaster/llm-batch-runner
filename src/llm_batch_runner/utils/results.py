import json
import os
from collections.abc import Sequence
from typing import Literal, overload
from warnings import warn

import polars as pl
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from .types import (
    DoneResultRow,
    PolarsDataFrame,
    ResultRow,
    ResultRowDict,
    ReturnDType,
    result_row_to_dict,
)


async def export_jsonl(db_url: str, out: str = "results.jsonl") -> None:
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
    progress_url: str,
    explicit_results_url: str | None,
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
    db_path: str | None = url.database

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
    results_db_url: str,
    results: Sequence[ResultRow],
) -> None:
    """Persist final results to a separate results DB."""
    results_engine = create_async_engine(results_db_url, future=True)
    try:
        async with results_engine.begin() as conn:
            await conn.execute(text("DROP TABLE IF EXISTS results"))
            await conn.execute(
                text(
                    """
                    CREATE TABLE results (
                        idx INTEGER NOT NULL,
                        key TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        status TEXT NOT NULL,
                        result TEXT,
                        error TEXT,
                        PRIMARY KEY (idx, key)
                    )
                    """
                )
            )

            payload = [result_row_to_dict(row) for row in results]
            if payload:
                await conn.execute(
                    text(
                        "INSERT INTO results (idx, key, prompt, status, result, error) "
                        "VALUES (:idx, :key, :prompt, :status, :result, :error)"
                    ),
                    payload,
                )
        print(f"Wrote {len(results)} rows to results DB: {results_db_url}")
    finally:
        await results_engine.dispose()


def _require_done_results(
    results: Sequence[ResultRow],
    return_dtype: Literal["list[str]", "list[tuple[str,str]]"],
) -> list[DoneResultRow]:
    done_results: list[DoneResultRow] = []
    failed_keys: list[str] = []
    for row in results:
        if isinstance(row, DoneResultRow):
            done_results.append(row)
            continue
        failed_keys.append(row.key)

    if failed_keys:
        sample = ", ".join(failed_keys[:3])
        raise ValueError(
            f"return_dtype={return_dtype!r} requires every row to be done; "
            f"found {len(failed_keys)} failed rows (e.g. keys: {sample})."
        )

    return done_results


@overload
def shape_return(
    results: Sequence[ResultRow],
    return_dtype: Literal["list[dict]"],
) -> list[ResultRowDict]: ...


@overload
def shape_return(
    results: Sequence[ResultRow],
    return_dtype: Literal["list[str]"],
) -> list[str]: ...


@overload
def shape_return(
    results: Sequence[ResultRow],
    return_dtype: Literal["list[tuple[str,str]]"],
) -> list[tuple[str, str]]: ...


@overload
def shape_return(
    results: Sequence[ResultRow],
    return_dtype: Literal["polars"],
) -> PolarsDataFrame: ...


def shape_return(
    results: Sequence[ResultRow],
    return_dtype: ReturnDType,
) -> list[ResultRowDict] | list[str] | list[tuple[str, str]] | PolarsDataFrame:
    """Shape results into the requested return dtype."""
    result_dicts = [result_row_to_dict(row) for row in results]
    if return_dtype == "list[dict]":
        return result_dicts
    if return_dtype == "list[str]":
        return [row.result for row in _require_done_results(results, return_dtype)]
    if return_dtype == "list[tuple[str,str]]":
        return [
            (row.prompt, row.result)
            for row in _require_done_results(results, return_dtype)
        ]
    if return_dtype == "polars":
        return pl.DataFrame(result_dicts)

    raise ValueError(
        "Invalid return_dtype. Expected one of: 'list[dict]', 'list[str]', 'list[tuple[str,str]]', 'polars'"
    )


__all__ = [
    "export_jsonl",
    "derive_results_db_url",
    "write_results_to_db",
    "shape_return",
]
