from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from llm_batch_runner.utils.db import (
    fetch_results,
    init_db,
    key_for,
    seed,
    set_done,
    set_failed,
    set_inflight,
)


def sqlite_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path}"


@pytest.mark.asyncio
async def test_fetch_results_preserves_done_only_legacy_dict_contract(
    tmp_path: Path,
) -> None:
    prompts = ["done-a", "failed", "inflight", "pending", "done-b"]
    engine = create_async_engine(sqlite_url(tmp_path / "runs.db"), future=True)
    try:
        await init_db(engine)
        await seed(engine, prompts)
        await set_done(engine, key_for("done-a"), "A")
        await set_failed(engine, key_for("failed"), "failure")
        await set_inflight(engine, key_for("inflight"))
        await set_done(engine, key_for("done-b"), "B")

        expected = [
            {
                "idx": 0,
                "key": key_for("done-a"),
                "prompt": "done-a",
                "status": "done",
                "result": "A",
            },
            {
                "idx": 4,
                "key": key_for("done-b"),
                "prompt": "done-b",
                "status": "done",
                "result": "B",
            },
        ]

        results = await fetch_results(engine)
        assert results == expected
        assert isinstance(results[0], dict)
        assert results[0]["result"] == "A"
        assert await fetch_results(engine, []) == expected
        assert await fetch_results(
            engine,
            [key_for("failed"), key_for("done-b")],
        ) == [expected[1]]
    finally:
        await engine.dispose()
