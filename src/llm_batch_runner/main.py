import asyncio
import os
import time
from typing import Awaitable, Callable, List, Optional
from warnings import warn

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine
from tenacity import AsyncRetrying, stop_after_attempt, wait_random_exponential

from llm_batch_runner.utils import (
    DB_URL_DEFAULT,
    count_status,
    fetch_results,
    init_db,
    key_for,
    make_pydantic_ai_worker,
    ensure_sqlite_dir,
    seed,
    select_to_run,
    set_done,
    set_failed,
    set_inflight,
)


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
        db_url (str): The SQLAlchemy database URL for storing job progress.
            Defaults to a local SQLite file `runsjobs.db`.
        results_db_url (Optional[str]): A separate SQLAlchemy URL for writing the
            final outputs (key/prompt/status/result). Defaults to a sibling file
            derived from `db_url` (e.g., `jobs-results.db` for `jobs.db`).
        concurrency (int): The maximum number of concurrent tasks to run.
        max_attempts (int): The maximum number of retry attempts for each prompt
            in case of failure.
        progress_update_every (int): How often (in terms of completed tasks) to print
            progress statistics to the console.
        teardown (bool): If True, the *progress* DB file at `db_url` will be removed
            after the run is complete. The results DB is never torn down automatically.

    Returns:
        List[dict]: A list of dictionary objects representing the results for
            each prompt, sorted in the original order. Each dictionary includes
            the job `key`, `prompt`, final `status`, and `result`.

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
    if worker is None:
        # Try params; if missing, fall back to .env
        if model_name is None or openrouter_api_key is None:
            try:
                from dotenv import find_dotenv, load_dotenv

                load_dotenv(find_dotenv(usecwd=True))
            except Exception:
                # dotenv is optional; if not present we just rely on os.environ
                pass
            model_name = model_name or os.getenv("MODEL")
            openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")

        if not model_name or not openrouter_api_key:
            raise ValueError(
                "Missing model credentials. Provide either: "
                "(a) a `worker`, or (b) `model_name` + `openrouter_api_key`, "
                "or (c) set MODEL and OPENROUTER_API_KEY in your environment/.env."
            )
        worker = make_pydantic_ai_worker(
            model_name=model_name,
            api_key=openrouter_api_key,
            result_type=response_model,
        )

    # Derive a safe default results DB path if not provided, ensuring it differs from db_url
    def _derive_results_db_url(
        progress_url: str, explicit_results_url: Optional[str]
    ) -> str:
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

    results_db_url = _derive_results_db_url(cache_db_url, results_db_url)

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

        # Collect and return results from the *progress* DB (ordered by idx)
        results = await fetch_results(engine, keys)

    finally:
        await engine.dispose()
        if teardown:
            # Best-effort removal of SQLite file if present (progress DB only)
            try:
                url = make_url(cache_db_url)
                db_path: Optional[str] = url.database
                if (
                    isinstance(db_path, str)
                    and db_path
                    and db_path != ":memory:"
                    and os.path.exists(db_path)
                ):
                    os.remove(db_path)
                    print(f"Tore down progress DB file at: {db_path}")
            except Exception:
                # Swallow teardown issues silently to avoid masking run success
                warn("Progress DB teardown failed.")

    # === Write final outputs to a separate results DB ===
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

    return results
