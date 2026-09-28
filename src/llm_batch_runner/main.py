import asyncio
import time
from collections import deque
from collections.abc import Iterable, Sequence
from typing import Literal, overload

from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine
from tenacity import AsyncRetrying, stop_after_attempt, wait_random_exponential

from .utils.db import (
    DB_URL_DEFAULT,
    count_status,
    fetch_job_rows,
    init_db,
    key_for,
    seed,
    select_to_run,
    set_done,
    set_failed,
    set_inflight,
)
from .utils.paths import ensure_sqlite_dir, teardown_sqlite_file
from .utils.rate_limits import (
    RateLimitConfig,
    RollingWindowRateLimiter,
    resolve_rate_limit,
)
from .utils.results import derive_results_db_url, shape_return, write_results_to_db
from .utils.types import (
    JobRow,
    OutputShape,
    PolarsDataFrame,
    ResultProcessor,
    ResultRow,
    ResultRowDict,
    ReturnDType,
    Worker,
    result_row_from_job,
)
from .utils.workers import (
    RecoverableWorkerError,
    resolve_model_credentials,
    resolve_worker,
)


def _build_terminal_results(
    prompt_list: Sequence[str],
    job_rows: Sequence[JobRow],
    output_shape: OutputShape,
) -> list[ResultRow]:
    terminal_by_key: dict[str, ResultRow] = {}
    for job_row in job_rows:
        row = result_row_from_job(job_row)
        if row.key in terminal_by_key:
            raise RuntimeError(f"Duplicate terminal row found for key {row.key!r}.")
        terminal_by_key[row.key] = row

    results: list[ResultRow] = []
    seen_keys: set[str] = set()
    for idx, prompt in enumerate(prompt_list):
        key = key_for(prompt)
        terminal_row = terminal_by_key.get(key)
        if terminal_row is None:
            raise RuntimeError(f"Missing terminal row for prompt key {key!r}.")
        if output_shape == "unique":
            if key in seen_keys:
                continue
            seen_keys.add(key)
            results.append(terminal_row)
            continue
        results.append(terminal_row.with_idx(idx))

    return results


async def _run_worker_with_retry(
    worker: Worker,
    result_processor: ResultProcessor | None,
    prompt: str,
    limiter: RollingWindowRateLimiter,
    sem: asyncio.Semaphore,
    max_attempts: int,
) -> str:
    async for attempt in AsyncRetrying(
        wait=wait_random_exponential(max=60),
        stop=stop_after_attempt(max_attempts),
        reraise=True,
    ):
        with attempt:
            await limiter.acquire()
            async with sem:
                result = await worker(prompt)
                if result_processor is None:
                    return result
                try:
                    processed_result = result_processor(result)
                    if not isinstance(processed_result, str):
                        raise TypeError(
                            "Result processor must return str, got "
                            f"{type(processed_result).__name__}."
                        )
                    return processed_result
                except Exception as exc:
                    raise RecoverableWorkerError(str(exc)) from exc

    raise RuntimeError("Retry loop exited without returning a worker result.")


@overload
async def prompt_map(
    prompts: Iterable[str],
    worker: Worker | None = None,
    model_name: str | None = None,
    openrouter_api_key: str | None = None,
    providers: Sequence[str] | None = None,
    allow_provider_fallbacks: bool | None = None,
    response_model: type[BaseModel] | None = None,
    result_processor: ResultProcessor | None = None,
    temperature: float = 0.0,
    cache_db_url: str = DB_URL_DEFAULT,
    results_db_url: str | None = None,
    concurrency: int = 128,
    rate_limit: RateLimitConfig | None = None,
    max_attempts: int = 3,
    progress_update_every: int = 200,
    teardown: bool = True,
    teardown_results: bool = True,
    return_dtype: Literal["list[dict]"] = "list[dict]",
    output_shape: OutputShape = "original",
    verbose: bool = True,
) -> list[ResultRowDict]: ...


@overload
async def prompt_map(
    prompts: Iterable[str],
    worker: Worker | None = None,
    model_name: str | None = None,
    openrouter_api_key: str | None = None,
    providers: Sequence[str] | None = None,
    allow_provider_fallbacks: bool | None = None,
    response_model: type[BaseModel] | None = None,
    result_processor: ResultProcessor | None = None,
    temperature: float = 0.0,
    cache_db_url: str = DB_URL_DEFAULT,
    results_db_url: str | None = None,
    concurrency: int = 128,
    rate_limit: RateLimitConfig | None = None,
    max_attempts: int = 3,
    progress_update_every: int = 200,
    teardown: bool = True,
    teardown_results: bool = True,
    return_dtype: Literal["list[str]"] = "list[str]",
    output_shape: OutputShape = "original",
    verbose: bool = True,
) -> list[str]: ...


@overload
async def prompt_map(
    prompts: Iterable[str],
    worker: Worker | None = None,
    model_name: str | None = None,
    openrouter_api_key: str | None = None,
    providers: Sequence[str] | None = None,
    allow_provider_fallbacks: bool | None = None,
    response_model: type[BaseModel] | None = None,
    result_processor: ResultProcessor | None = None,
    temperature: float = 0.0,
    cache_db_url: str = DB_URL_DEFAULT,
    results_db_url: str | None = None,
    concurrency: int = 128,
    rate_limit: RateLimitConfig | None = None,
    max_attempts: int = 3,
    progress_update_every: int = 200,
    teardown: bool = True,
    teardown_results: bool = True,
    return_dtype: Literal["list[tuple[str,str]]"] = "list[tuple[str,str]]",
    output_shape: OutputShape = "original",
    verbose: bool = True,
) -> list[tuple[str, str]]: ...


@overload
async def prompt_map(
    prompts: Iterable[str],
    worker: Worker | None = None,
    model_name: str | None = None,
    openrouter_api_key: str | None = None,
    providers: Sequence[str] | None = None,
    allow_provider_fallbacks: bool | None = None,
    response_model: type[BaseModel] | None = None,
    result_processor: ResultProcessor | None = None,
    temperature: float = 0.0,
    cache_db_url: str = DB_URL_DEFAULT,
    results_db_url: str | None = None,
    concurrency: int = 128,
    rate_limit: RateLimitConfig | None = None,
    max_attempts: int = 3,
    progress_update_every: int = 200,
    teardown: bool = True,
    teardown_results: bool = True,
    return_dtype: Literal["polars"] = "polars",
    output_shape: OutputShape = "original",
    verbose: bool = True,
) -> PolarsDataFrame: ...


async def prompt_map(
    prompts: Iterable[str],
    worker: Worker | None = None,
    model_name: str | None = None,
    openrouter_api_key: str | None = None,
    providers: Sequence[str] | None = None,
    allow_provider_fallbacks: bool | None = None,
    response_model: type[BaseModel] | None = None,
    result_processor: ResultProcessor | None = None,
    temperature: float = 0.0,
    cache_db_url: str = DB_URL_DEFAULT,
    results_db_url: str | None = None,
    concurrency: int = 128,
    rate_limit: RateLimitConfig | None = None,
    max_attempts: int = 3,
    progress_update_every: int = 200,
    teardown: bool = True,
    teardown_results: bool = True,
    return_dtype: ReturnDType = "list[dict]",
    output_shape: OutputShape = "original",
    verbose: bool = True,
) -> list[ResultRowDict] | list[str] | list[tuple[str, str]] | PolarsDataFrame:
    """Map prompts through a worker with durable progress tracking.

    The worker boundary is always `str -> await str`. If `response_model` is
    provided, the built-in Pydantic AI worker validates the model response and
    serializes it to JSON before the result crosses that boundary. If
    `result_processor` is provided, each new worker result is processed inside
    the retry loop before it is cached as done.

    Failed prompts remain explicit failed rows in `"list[dict]"` and `"polars"`
    outputs. Return shapes that cannot represent failures (`"list[str]"` and
    `"list[tuple[str,str]]"`) raise when any row is not `done`.
    """
    if isinstance(prompts, str):
        raise ValueError(
            f"`prompts` must be an iterable of prompt strings, not {prompts!r}."
        )
    prompt_list = list(prompts)

    resolved_model_name, resolved_openrouter_api_key = resolve_model_credentials(
        model_name,
        openrouter_api_key,
    )

    # Resolve worker via (a) explicit worker, (b) params, or (c) .env
    worker = resolve_worker(
        worker=worker,
        model_name=resolved_model_name,
        openrouter_api_key=resolved_openrouter_api_key,
        providers=providers,
        allow_provider_fallbacks=allow_provider_fallbacks,
        response_model=response_model,
        temperature=temperature,
    )

    effective_rate_limit = resolve_rate_limit(
        model_name=resolved_model_name,
        rate_limit=rate_limit,
    )

    # Derive a safe default results DB path if not provided, ensuring it differs from the progress DB URL
    results_db_url = derive_results_db_url(cache_db_url, results_db_url)

    # Ensure parent directories exist for SQLite files
    ensure_sqlite_dir(cache_db_url)
    ensure_sqlite_dir(results_db_url)

    engine = create_async_engine(cache_db_url, future=True)
    limiter = RollingWindowRateLimiter(**effective_rate_limit)
    results: list[ResultRow] = []
    try:
        await init_db(engine)
        await seed(engine, prompt_list)
        keys = [key_for(prompt) for prompt in prompt_list]
        unique_keys = list(dict.fromkeys(keys))
        to_run = await select_to_run(engine, unique_keys)
        total = len(prompt_list)
        rem = len(to_run)
        if verbose:
            print(f"Total: {total} | already done: {total-rem} | remaining: {rem}")

        sem = asyncio.Semaphore(concurrency)

        async def runner(job: JobRow) -> None:
            """Execute a single job: mark inflight, call worker with retry, persist result."""
            await set_inflight(engine, job.key)
            try:
                result = await _run_worker_with_retry(
                    worker,
                    result_processor,
                    job.prompt,
                    limiter,
                    sem,
                    max_attempts,
                )
            except RecoverableWorkerError as exc:
                root_cause = exc.__cause__ if exc.__cause__ is not None else exc
                await set_failed(engine, job.key, repr(root_cause))
                return

            await set_done(engine, job.key, result)

        tasks = [asyncio.create_task(runner(job)) for job in to_run]
        done = 0
        # lightweight rolling RPM meter
        finish_times: deque[float] = deque()
        try:
            for fut in asyncio.as_completed(tasks):
                await fut
                done += 1
                # update rolling window
                now = time.monotonic()
                finish_times.append(now)
                # keep only last 60s
                cutoff = now - 60.0
                while finish_times and finish_times[0] < cutoff:
                    finish_times.popleft()

                if done % progress_update_every == 0 or (rem and done == rem):
                    stats = await count_status(engine)
                    if verbose:
                        rpm = len(finish_times)
                        print(
                            f"[{time.strftime('%H:%M:%S')}] progress: {stats} | rolling_rpm≈{rpm}"
                        )
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        if verbose:
            print("Batch complete.")

        job_rows = await fetch_job_rows(engine, unique_keys)
        results = _build_terminal_results(prompt_list, job_rows, output_shape)

    finally:
        await engine.dispose()

    await write_results_to_db(results_db_url, results)

    try:
        return shape_return(results, return_dtype)
    finally:
        if teardown:
            teardown_sqlite_file(cache_db_url)
        if teardown_results:
            teardown_sqlite_file(results_db_url)
