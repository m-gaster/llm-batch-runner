import asyncio
import os

from llm_batch_runner.main import prompt_map

if __name__ == "__main__":
    from dotenv import find_dotenv, load_dotenv

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
        """Small demo that runs three prompts and prints results."""
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
