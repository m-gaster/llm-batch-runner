import os
from collections.abc import Sequence

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings

from .types import Worker


class RecoverableWorkerError(RuntimeError):
    """A worker execution failure that should mark a job as failed."""


def load_env_defaults() -> None:
    """Load .env values into the process environment."""
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))


def resolve_model_credentials(
    model_name: str | None,
    openrouter_api_key: str | None,
) -> tuple[str | None, str | None]:
    """Resolve model name and API key from args or environment."""
    if model_name is None or openrouter_api_key is None:
        load_env_defaults()

    return (
        model_name or os.getenv("MODEL"),
        openrouter_api_key or os.getenv("OPENROUTER_API_KEY"),
    )


def make_pydantic_ai_worker(
    *,
    model_name: str,
    api_key: str,
    providers: Sequence[str] | None = None,
    allow_provider_fallbacks: bool | None = None,
    result_type: type[BaseModel] | None = None,
    temperature: float = 0.0,
) -> Worker:
    """Build a reusable async worker backed by pydantic-ai + OpenRouter."""
    model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenRouterProvider(api_key=api_key),
    )
    model_settings: ModelSettings = {
        "temperature": temperature,
        "timeout": 20.0,
    }
    provider_settings: dict[str, object] = {}
    if providers is not None:
        provider_settings["order"] = list(providers)
    if allow_provider_fallbacks is not None:
        provider_settings["allow_fallbacks"] = allow_provider_fallbacks
    if provider_settings:
        model_settings["extra_body"] = {"provider": provider_settings}

    if result_type is not None:
        agent = Agent(model, output_type=result_type)

        async def _structured_worker(prompt: str) -> str:
            try:
                run = await agent.run(
                    prompt,
                    model_settings=model_settings,
                )
            except Exception as exc:
                raise RecoverableWorkerError(str(exc)) from exc

            output = run.output
            if not isinstance(output, result_type):
                raise TypeError(
                    f"Structured worker expected {result_type.__name__}, got {type(output).__name__}."
                )
            return output.model_dump_json()

        return _structured_worker

    agent = Agent(model)

    async def _worker(prompt: str) -> str:
        try:
            run = await agent.run(
                prompt,
                model_settings=model_settings,
            )
        except Exception as exc:
            raise RecoverableWorkerError(str(exc)) from exc

        output = run.output
        if not isinstance(output, str):
            raise TypeError(f"Worker expected str output, got {type(output).__name__}.")
        return output

    return _worker


def _wrap_worker(worker: Worker) -> Worker:
    async def _wrapped_worker(prompt: str) -> str:
        try:
            result = await worker(prompt)
        except Exception as exc:
            raise RecoverableWorkerError(str(exc)) from exc

        if not isinstance(result, str):
            raise TypeError(f"Worker must return str, got {type(result).__name__}.")
        return result

    return _wrapped_worker


def resolve_worker(
    worker: Worker | None,
    model_name: str | None,
    openrouter_api_key: str | None,
    providers: Sequence[str] | None,
    allow_provider_fallbacks: bool | None,
    response_model: type[BaseModel] | None,
    temperature: float = 0.0,
) -> Worker:
    """Resolve a worker function from explicit worker or model credentials/env."""
    if worker is not None:
        if response_model is not None:
            raise ValueError(
                "`response_model` can only be used with the built-in Pydantic AI worker."
            )
        if providers is not None or allow_provider_fallbacks is not None:
            raise ValueError(
                "`providers` and `allow_provider_fallbacks` can only be used with "
                "the built-in Pydantic AI worker."
            )
        return _wrap_worker(worker)

    model_name, openrouter_api_key = resolve_model_credentials(
        model_name,
        openrouter_api_key,
    )

    if not model_name or not openrouter_api_key:
        raise ValueError(
            "Missing model credentials. Provide either: "
            "(a) a `worker`, or (b) `model_name` + `openrouter_api_key`, "
            "or (c) set MODEL and OPENROUTER_API_KEY in your environment/.env."
        )

    return make_pydantic_ai_worker(
        model_name=model_name,
        api_key=openrouter_api_key,
        providers=providers,
        allow_provider_fallbacks=allow_provider_fallbacks,
        result_type=response_model,
        temperature=temperature,
    )


__all__ = [
    "RecoverableWorkerError",
    "load_env_defaults",
    "make_pydantic_ai_worker",
    "resolve_model_credentials",
    "resolve_worker",
]
