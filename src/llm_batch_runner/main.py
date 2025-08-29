import asyncio
import os
import time
from typing import Callable, Awaitable, List
from sqlalchemy.ext.asyncio import create_async_engine
from tenacity import AsyncRetrying, wait_random_exponential, stop_after_attempt
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from llm_batch_runner.utils import (
    DB_URL_DEFAULT,
    count_status,
    export_jsonl,
    init_db,
    key_for,
    seed,
    select_to_run,
    set_done,
    set_failed,
    set_inflight,
)


# ---------- main access point ----------
async def prompt_map(
    prompts: List[str],
    worker: Callable[[str], Awaitable[str]],
    *,
    db_url: str = DB_URL_DEFAULT,
    concurrency: int = 32,
    max_attempts: int = 8,
    progress_every: int = 200,
):
    """
    Map a single-argument async worker over a list of prompt strings, with durable progress.
    Re-running resumes any pending/failed/inflight rows until all are 'done'.
    """
    engine = create_async_engine(db_url, future=True)
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
    finally:
        await engine.dispose()


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
    return run.output or ""


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
        await prompt_map(prompts, pydantic_ai_worker, concurrency=24, max_attempts=8)
        await export_jsonl(DB_URL_DEFAULT, "results.jsonl")

    asyncio.run(main())
