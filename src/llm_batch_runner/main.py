import asyncio
import os
import time
from typing import Callable, Awaitable, List, Optional
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.engine import make_url
from tenacity import AsyncRetrying, wait_random_exponential, stop_after_attempt
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from llm_batch_runner.utils import (
    DB_URL_DEFAULT,
    count_status,
    init_db,
    key_for,
    seed,
    select_to_run,
    set_done,
    set_failed,
    set_inflight,
    fetch_results,
)


def make_pydantic_ai_worker(
    *, model_name: str, api_key: str
) -> Callable[[str], Awaitable[str]]:
    """
    Build a reusable async worker backed by pydantic-ai + OpenRouter.
    """
    model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenRouterProvider(api_key=api_key),
    )
    agent = Agent(model)

    async def _worker(prompt: str) -> str:
        run = await agent.run(
            prompt,
            model_settings={"temperature": 0.0, "timeout": 90.0},
        )
        return run.output

    return _worker


# ---------- main access point ----------
async def prompt_map(
    prompts: List[str],
    worker: Optional[Callable[[str], Awaitable[str]]] = None,
    # (b) Direct params
    model_name: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    # existing config
    db_url: str = DB_URL_DEFAULT,
    concurrency: int = 32,
    max_attempts: int = 8,
    progress_every: int = 200,
    teardown: bool = False,
):
    """
    Map a single-argument async worker over a list of prompt strings, with durable progress.
    Returns the completed results (ordered by original idx) instead of exporting to disk.

    You can provide the worker in three ways:
      (a) Pass a prebuilt async `worker(prompt) -> str`
      (b) Pass `model_name` and `openrouter_api_key` params (OpenAI-compatible via OpenRouter)
      (c) Omit both and set MODEL and OPENROUTER_API_KEY in your .env (loaded automatically)
    """
    # Resolve worker via (a) explicit worker, (b) params, or (c) .env
    if worker is None:
        # Try params; if missing, fall back to .env
        if model_name is None or openrouter_api_key is None:
            try:
                from dotenv import load_dotenv, find_dotenv

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
            model_name=model_name, api_key=openrouter_api_key
        )

    engine = create_async_engine(db_url, future=True)
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
            if done % progress_every == 0 or (rem and done == rem):
                stats = await count_status(engine)
                print(f"[{time.strftime('%H:%M:%S')}] progress:", stats)
        print("Batch complete.")

        # Collect and return results from the DB (ordered by idx)
        results = await fetch_results(engine, keys)
    finally:
        await engine.dispose()
        if teardown:
            # Best-effort removal of SQLite file if present
            try:
                url = make_url(db_url)
                db_path: Optional[str] = url.database
                if (
                    isinstance(db_path, str)
                    and db_path
                    and db_path != ":memory:"
                    and os.path.exists(db_path)
                ):
                    os.remove(db_path)
                    print(f"Tore down DB file at: {db_path}")
            except Exception:
                # Swallow teardown issues silently to avoid masking run success
                pass
    return results


# ---------- worker (Pydantic AI, OpenAI-compatible) ----------
async def pydantic_ai_worker(prompt: str) -> str:
    """
    Uses Pydantic AI Agent with an OpenAI-compatible provider (e.g. OpenRouter).
    Requires:
      - MODEL: model name (e.g. 'openai/gpt-4o-mini' on OpenRouter)
      - OPENROUTER_API_KEY (if using OpenRouter)
    """
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(usecwd=True))  # make sure this runs before os.getenv

    model_name = os.getenv("MODEL")
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(model_name)
    if model_name is None or api_key is None:
        raise ValueError("Could not retrive MODEL or OPENROUTER_API_KEY from .env")

    model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenRouterProvider(api_key=api_key),
    )
    agent = Agent(model)

    # Mirror original behavior: low randomness and a 90s timeout
    run = await agent.run(
        prompt,
        model_settings={"temperature": 0.0, "timeout": 90.0},
    )
    return run.output


# ---------- demo ----------
if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv(usecwd=True))  # make sure this runs before os.getenv

    model_name = os.getenv("MODEL")
    print(f"{model_name=}")
    if model_name is None:
        raise ValueError()
    prompts = [
        "Summarize: The quick brown fox jumps over the lazy dog.",
        "Give me 3 bullet points on why the sky appears blue.",
        "Rewrite this in pirate speak: Hello, friend!",
    ]

    async def main():
        # Option (a): provide your own worker (unchanged behavior)
        results = await prompt_map(
            prompts,
            # pydantic_ai_worker,
            # model_name=os.getenv("MODEL"),
            # openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            concurrency=24,
            max_attempts=8,
            teardown=True,
        )
        print(f"Got {len(results)} results.")
        for row in results:
            print(row)

        # Option (b): pass params directly
        # results = await prompt_map(
        #     prompts,
        #     None,
        #     model_name=os.getenv("MODEL"),
        #     openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        # )

        # Option (c): omit worker and params; rely on .env (MODEL, OPENROUTER_API_KEY)
        # results = await prompt_map(prompts)

    asyncio.run(main())
