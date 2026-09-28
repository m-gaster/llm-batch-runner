import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest
from pydantic import BaseModel

from llm_batch_runner.main import prompt_map
from llm_batch_runner.utils.results import write_results_to_db
from llm_batch_runner.utils.types import ResultRow


def sqlite_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path}"


class Bullets(BaseModel):
    points: list[str]


@pytest.mark.asyncio
async def test_prompt_map_records_failed_rows_in_dict_output(
    tmp_path: Path,
) -> None:
    async def worker(prompt: str) -> str:
        if prompt == "boom":
            raise RuntimeError("boom")
        return prompt.upper()

    results = await prompt_map(
        ["alpha", "boom"],
        worker=worker,
        rate_limit={
            "requests_per_minute": 120,
            "requests_per_second": 10,
        },
        cache_db_url=sqlite_url(tmp_path / "runs.db"),
        results_db_url=sqlite_url(tmp_path / "results.db"),
        max_attempts=1,
        verbose=False,
        teardown=False,
        teardown_results=False,
    )

    assert results[0] == {
        "idx": 0,
        "key": results[0]["key"],
        "prompt": "alpha",
        "status": "done",
        "result": "ALPHA",
        "error": None,
    }
    assert results[1]["prompt"] == "boom"
    assert results[1]["status"] == "failed"
    assert results[1]["result"] is None
    assert "RuntimeError('boom')" in cast(str, results[1]["error"])


@pytest.mark.asyncio
async def test_prompt_map_processes_result_before_caching(
    tmp_path: Path,
) -> None:
    worker_calls = 0
    processor_calls = 0

    async def worker(_prompt: str) -> str:
        nonlocal worker_calls
        worker_calls += 1
        if worker_calls == 1:
            return '{"points":"invalid"}'
        return '{ "points": ["valid"] }'

    def process_result(raw_result: str) -> str:
        nonlocal processor_calls
        processor_calls += 1
        return Bullets.model_validate_json(raw_result).model_dump_json()

    cache_db_url = sqlite_url(tmp_path / "runs.db")
    results_db_url = sqlite_url(tmp_path / "results.db")
    results = await prompt_map(
        ["alpha"],
        worker=worker,
        result_processor=process_result,
        rate_limit={
            "requests_per_minute": 120,
            "requests_per_second": 10,
        },
        cache_db_url=cache_db_url,
        results_db_url=results_db_url,
        max_attempts=2,
        verbose=False,
        teardown=False,
        teardown_results=False,
    )

    assert worker_calls == 2
    assert processor_calls == 2
    assert results[0]["status"] == "done"
    assert results[0]["result"] == '{"points":["valid"]}'

    async def must_not_run(_prompt: str) -> str:
        raise AssertionError("cached result should not rerun the worker")

    cached_results = await prompt_map(
        ["alpha"],
        worker=must_not_run,
        result_processor=process_result,
        rate_limit={
            "requests_per_minute": 120,
            "requests_per_second": 10,
        },
        cache_db_url=cache_db_url,
        results_db_url=results_db_url,
        verbose=False,
    )

    assert processor_calls == 2
    assert cached_results[0]["result"] == '{"points":["valid"]}'


@pytest.mark.asyncio
async def test_prompt_map_records_exhausted_result_processing_as_failed(
    tmp_path: Path,
) -> None:
    async def worker(_prompt: str) -> str:
        return '{"points":"invalid"}'

    def process_result(raw_result: str) -> str:
        return Bullets.model_validate_json(raw_result).model_dump_json()

    results = await prompt_map(
        ["alpha"],
        worker=worker,
        result_processor=process_result,
        rate_limit={
            "requests_per_minute": 120,
            "requests_per_second": 10,
        },
        cache_db_url=sqlite_url(tmp_path / "runs.db"),
        results_db_url=sqlite_url(tmp_path / "results.db"),
        max_attempts=1,
        verbose=False,
        teardown=False,
        teardown_results=False,
    )

    assert results[0]["status"] == "failed"
    assert results[0]["result"] is None
    assert "validation error for Bullets" in cast(str, results[0]["error"])


@pytest.mark.asyncio
async def test_prompt_map_unique_output_keeps_failed_rows(
    tmp_path: Path,
) -> None:
    async def worker(prompt: str) -> str:
        if prompt == "boom":
            raise RuntimeError("boom")
        return prompt.upper()

    results = await prompt_map(
        ["boom", "alpha", "boom"],
        worker=worker,
        rate_limit={
            "requests_per_minute": 120,
            "requests_per_second": 10,
        },
        cache_db_url=sqlite_url(tmp_path / "runs.db"),
        results_db_url=sqlite_url(tmp_path / "results.db"),
        max_attempts=1,
        verbose=False,
        teardown=False,
        teardown_results=False,
        output_shape="unique",
    )

    assert len(results) == 2
    assert [row["prompt"] for row in results] == ["boom", "alpha"]
    assert results[0]["status"] == "failed"
    assert results[1]["status"] == "done"


@pytest.mark.asyncio
async def test_prompt_map_list_str_raises_when_any_prompt_fails(
    tmp_path: Path,
) -> None:
    async def worker(prompt: str) -> str:
        if prompt == "boom":
            raise RuntimeError("boom")
        return prompt.upper()

    with pytest.raises(ValueError, match="requires every row to be done"):
        await prompt_map(
            ["alpha", "boom"],
            worker=worker,
            rate_limit={
                "requests_per_minute": 120,
                "requests_per_second": 10,
            },
            cache_db_url=sqlite_url(tmp_path / "runs.db"),
            results_db_url=sqlite_url(tmp_path / "results.db"),
            max_attempts=1,
            verbose=False,
            teardown=False,
            teardown_results=False,
            return_dtype="list[str]",
        )


@pytest.mark.asyncio
async def test_prompt_map_rejects_worker_and_response_model() -> None:
    async def worker(prompt: str) -> str:
        return prompt

    with pytest.raises(
        ValueError,
        match="`response_model` can only be used with the built-in Pydantic AI worker",
    ):
        await prompt_map(
            ["alpha"],
            worker=worker,
            response_model=Bullets,
            verbose=False,
        )


@pytest.mark.asyncio
async def test_prompt_map_raises_for_non_string_worker_result(
    tmp_path: Path,
) -> None:
    async def worker(prompt: str) -> str:
        return cast(str, 123)

    with pytest.raises(TypeError, match="Worker must return str"):
        await prompt_map(
            ["alpha"],
            worker=worker,
            rate_limit={
                "requests_per_minute": 120,
                "requests_per_second": 10,
            },
            cache_db_url=sqlite_url(tmp_path / "runs.db"),
            results_db_url=sqlite_url(tmp_path / "results.db"),
            max_attempts=1,
            verbose=False,
            teardown=False,
            teardown_results=False,
        )


@pytest.mark.asyncio
async def test_prompt_map_cancels_sibling_tasks_before_contract_error_returns(
    tmp_path: Path,
) -> None:
    slow_worker_started = asyncio.Event()
    slow_worker_cancelled = asyncio.Event()
    slow_worker_completed = False

    async def worker(prompt: str) -> str:
        nonlocal slow_worker_completed
        if prompt == "bad":
            await slow_worker_started.wait()
            return cast(str, 123)

        slow_worker_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            slow_worker_cancelled.set()
            raise
        slow_worker_completed = True
        return prompt.upper()

    with pytest.raises(TypeError, match="Worker must return str"):
        await prompt_map(
            ["slow", "bad"],
            worker=worker,
            rate_limit={
                "requests_per_minute": 120,
                "requests_per_second": 10,
            },
            cache_db_url=sqlite_url(tmp_path / "runs.db"),
            results_db_url=sqlite_url(tmp_path / "results.db"),
            max_attempts=1,
            verbose=False,
            teardown=False,
            teardown_results=False,
        )

    assert slow_worker_cancelled.is_set()
    assert not slow_worker_completed


@pytest.mark.asyncio
async def test_prompt_map_cancels_worker_tasks_when_caller_cancels(
    tmp_path: Path,
) -> None:
    all_workers_started = asyncio.Event()
    started_count = 0
    cancelled_prompts: set[str] = set()

    async def worker(prompt: str) -> str:
        nonlocal started_count
        started_count += 1
        if started_count == 2:
            all_workers_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled_prompts.add(prompt)
            raise
        return prompt.upper()

    batch_task = asyncio.create_task(
        prompt_map(
            ["alpha", "beta"],
            worker=worker,
            rate_limit={
                "requests_per_minute": 120,
                "requests_per_second": 10,
            },
            cache_db_url=sqlite_url(tmp_path / "runs.db"),
            results_db_url=sqlite_url(tmp_path / "results.db"),
            max_attempts=1,
            verbose=False,
            teardown=False,
            teardown_results=False,
        )
    )
    await all_workers_started.wait()
    batch_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await batch_task

    assert cancelled_prompts == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_write_results_to_db_rejects_malformed_rows(
    tmp_path: Path,
) -> None:
    malformed_rows = cast(Sequence[ResultRow], [object()])

    with pytest.raises(TypeError, match="Unsupported result row type"):
        await write_results_to_db(
            sqlite_url(tmp_path / "results.db"),
            malformed_rows,
        )
