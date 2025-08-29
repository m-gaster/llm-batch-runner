import asyncio
import os

from pydantic import BaseModel

from llm_batch_runner.main import prompt_map


class Bullets(BaseModel):
    points: list[str]


if __name__ == "__main__":
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))  # make sure this runs before os.getenv

    model_name = os.getenv("MODEL")
    print(f"{model_name=}")
    if model_name is None:
        raise ValueError()
    prompts = [
        f"Add 1000 to this number. Respond with ONLY the number. Number: {i // 2}"
        for i in range(30)
    ]

    print(prompts)

    async def main():
        """Demo: run ~30 prompts concurrently and save a CSV of results."""
        # Option (a): provide your own worker (unchanged behavior)
        results = await prompt_map(
            prompts,
            # pydantic_ai_worker,
            # model_name=os.getenv("MODEL"),
            # openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            concurrency=24,
            max_attempts=8,
            teardown=True,
            teardown_results=False,
            return_dtype="polars",
            # response_model=Bullets,
        )
        print(f"Got {len(results)} results.")
        print(results)
        results.write_csv("trash/demo.csv")

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
