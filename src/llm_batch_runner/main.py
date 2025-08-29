import asyncio
import time
from typing import Awaitable, Callable, List, Optional, Literal

from sqlalchemy.ext.asyncio import create_async_engine
from tenacity import AsyncRetrying, stop_after_attempt, wait_random_exponential

from .utils.db import (
    DB_URL_DEFAULT,
    count_status,
    fetch_results,
    init_db,
    key_for,
    seed,
    select_to_run,
    set_done,
    set_failed,
    set_inflight,
)
from .utils.paths import ensure_sqlite_dir, teardown_sqlite_file
from .utils.results import (
    derive_results_db_url,
    write_results_to_db,
    shape_return,
)
from .utils.workers import resolve_worker


async def prompt_map(
    prompts: List[str],
    worker: Optional[Callable[[str], Awaitable[str]]] = None,
    # (b) Direct params
    model_name: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    # Optional: structured return type for Pydantic AI agent
    response_model: Optional[type] = None,
    # existing config
    cache_db_url: str = DB_URL_DEFAULT,
    # NEW: separate destination for final outputs (defaults to "<db>-results.db")
    results_db_url: Optional[str] = None,
    concurrency: int = 32,
    max_attempts: int = 8,
    progress_update_every: int = 200,
    teardown: bool = True,
    # NEW: return dtype options
    return_dtype: Literal[
        "list[dict]",
        "list[str]",
        "list[tuple[str,str]]",
        "polars",
    ] = "list[dict]",
    # NEW: control whether returned rows are unique prompts only,
    # or expanded back to the original input shape (duplicates included)
    output_shape: Literal["unique", "original"] = "original",
):
    """Processes a list of prompts concurrently with durable, persistent state.

    This function maps an asynchronous worker over a list of prompts, using a
    database (SQLite by default) to track progress. If the process is interrupted,
    it can be resumed, and already completed prompts will not be re-processed.

    The worker can be provided in one of three ways:
    1.  As a pre-built async function via the `worker` argument.
    2.  By providing `model_name` and `openrouter_api_key` arguments, which
        will create a default worker.
    3.  By setting `MODEL` and `OPENROUTER_API_KEY` in your environment or a
        `.env` file.

    For structured outputs, you can provide a Pydantic model to the
    `response_model` argument, and the internal worker will serialize its output
    as a JSON string matching that model.

    Args:
        prompts (List[str]): A list of prompt strings to be processed.
        worker (Optional[Callable[[str], Awaitable[str]]]): An optional pre-built
            asynchronous function that takes a prompt string and returns a result string.
        model_name (Optional[str]): The name of the language model to use (e.g.,
            "openai/gpt-4o"). Required if `worker` is not provided.
        openrouter_api_key (Optional[str]): The API key for the model provider.
            Required if `worker` is not provided.
        response_model (Optional[type]): A Pydantic model class to be used for
            structuring the model's output. The `result` field will be a JSON
            string representation of an instance of this model.
        cache_db_url (str): The SQLAlchemy database URL for storing job progress
            (cache). Defaults to a local SQLite file `.llm_batch_cache/runs.db`.
        results_db_url (Optional[str]): A separate SQLAlchemy URL for writing the
            final outputs (key/prompt/status/result). Defaults to a sibling file
            derived from `cache_db_url` (e.g., `runs-results.db` for `runs.db`).
        concurrency (int): The maximum number of concurrent tasks to run.
        max_attempts (int): The maximum number of retry attempts for each prompt
            in case of failure.
        progress_update_every (int): How often (in terms of completed tasks) to print
            progress statistics to the console.
        teardown (bool): If True, the *progress* DB file at `cache_db_url` will be removed
            after the run is complete. The results DB is never torn down automatically.
        output_shape (Literal["unique","original"]): Controls whether the returned
            results are deduplicated by prompt ("unique") or expanded back to the
            original input length ("original"). Defaults to "original".

    Returns:
        Depends on `return_dtype` and `output_shape`:
        - "list[dict]" (default): list of dicts with keys `idx`, `key`, `prompt`, `status`, `result`.
        - "list[str]": list of result strings only (order dictated by `output_shape`).
        - "list[tuple[str,str]]": list of `(prompt, result)` tuples.
        - "polars": a Polars DataFrame converted from the default "list[dict]" output.

        When `output_shape="unique"`, the return contains one row per
        unique prompt (deduplicated by prompt hash), ordered by the first
        occurrence of each unique prompt.

        When `output_shape="original"`, the return is expanded back to the same
        length and order as the input `prompts`, duplicating rows for duplicate
        prompts. If any prompt did not complete successfully and thus has no
        `done` row, its `result` will be `None` and `status` will be "missing" in
        the dict form.

    Raises:
        ValueError: If a worker cannot be resolved because neither a `worker` function
            nor the necessary model credentials (`model_name` and `openrouter_api_key`
            via arguments or environment variables) are provided.

    Example:
        >>> import asyncio
        >>>
        >>> async def main():
        ...     prompts = ["What is the capital of France?", "Who wrote 'Hamlet'?"]
        ...     # Make sure to set OPENROUTER_API_KEY in your environment
        ...     results = await prompt_map(prompts, model_name="openai/gpt-3.5-turbo")
        ...     for res in results:
        ...         print(f"Prompt: {res['prompt']}\\nResult: {res['result']}\\n---")
        >>>
        >>> if __name__ == "__main__":
        ...     asyncio.run(main())
    """

    # Resolve worker via (a) explicit worker, (b) params, or (c) .env
    worker = resolve_worker(worker, model_name, openrouter_api_key, response_model)

    # Derive a safe default results DB path if not provided, ensuring it differs from the progress DB URL
    results_db_url = derive_results_db_url(cache_db_url, results_db_url)

    # Ensure parent directories exist for SQLite files
    ensure_sqlite_dir(cache_db_url)
    ensure_sqlite_dir(results_db_url)

    engine = create_async_engine(cache_db_url, future=True)
    results: List[dict] = []
    try:
        await init_db(engine)
        await seed(engine, prompts)
        keys = [key_for(p) for p in prompts]
        to_run = await select_to_run(engine, keys)
        total = len(prompts)
        rem = len(to_run)
        print(f"Total: {total} | already done: {total-rem} | remaining: {rem}")

        sem = asyncio.Semaphore(concurrency)

        async def runner(k: str, p: str):
            """Execute a single job: mark inflight, call worker with retry, persist result.

            Args:
                k: Deterministic job key for the prompt.
                p: Prompt text to process.
            """
            # throttle the LLM call with the semaphore
            async with sem:
                await set_inflight(engine, k)
                try:
                    async for attempt in AsyncRetrying(
                        wait=wait_random_exponential(max=60),
                        stop=stop_after_attempt(max_attempts),
                        reraise=True,
                    ):
                        with attempt:
                            out = await worker(p)
                    await set_done(engine, k, out)
                except Exception as e:
                    await set_failed(engine, k, repr(e))

        tasks = [asyncio.create_task(runner(k, p)) for (k, p) in to_run]
        done = 0
        for fut in asyncio.as_completed(tasks):
            await fut
            done += 1
            if done % progress_update_every == 0 or (rem and done == rem):
                stats = await count_status(engine)
                print(f"[{time.strftime('%H:%M:%S')}] progress:", stats)
        print("Batch complete.")

        # Collect results from the *progress* DB (only rows with status='done')
        # Ordered by the first occurrence index of each unique prompt
        unique_results = await fetch_results(engine, keys)

    finally:
        await engine.dispose()
        if teardown:
            teardown_sqlite_file(cache_db_url)

    # Expand back to original shape if requested
    if output_shape == "original":
        # Map key -> row for quick lookup
        by_key = {r["key"]: r for r in unique_results}
        expanded: List[dict] = []
        for i, p in enumerate(prompts):
            k = key_for(p)
            base = by_key.get(k)
            if base is not None:
                expanded.append(
                    {
                        "idx": i,
                        "key": k,
                        "prompt": p,
                        "status": base["status"],  # 'done'
                        "result": base["result"],
                    }
                )
            else:
                # Not present among 'done' rows → mark as missing
                expanded.append(
                    {
                        "idx": i,
                        "key": k,
                        "prompt": p,
                        "status": "missing",
                        "result": None,
                    }
                )
        results = expanded
    else:
        results = unique_results

    # === Write final outputs to a separate results DB ===
    await write_results_to_db(results_db_url, prompts, results)

    # Shape the return value according to return_dtype
    return shape_return(results, return_dtype)
