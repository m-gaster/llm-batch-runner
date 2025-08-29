import os
from typing import Any, Awaitable, Callable, Optional

from pydantic import TypeAdapter
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider


def make_pydantic_ai_worker(
    *, model_name: str, api_key: str, result_type: Optional[type] = None
) -> Callable[[str], Awaitable[str]]:
    """Build a reusable async worker backed by pydantic-ai + OpenRouter."""
    model = OpenAIChatModel(
        model_name=model_name,
        provider=OpenRouterProvider(api_key=api_key),
    )
    # If a Pydantic return type is provided, configure the agent for structured output
    if result_type is not None:
        agent = Agent(model, output_type=result_type)
    else:
        agent = Agent(model)

    async def _worker(prompt: str) -> str:
        run = await agent.run(
            prompt,
            model_settings={"temperature": 0.0, "timeout": 20.0},
        )
        output = run.output

        # No structured type configured → just return text
        if result_type is None:
            return output if isinstance(output, str) else str(output)

        # Structured type configured → serialize robustly to JSON
        if isinstance(output, str):
            # In case the model still returned plain text, pass it through
            return output

        # Prefer TypeAdapter so we handle lists, TypedDicts, dataclasses, etc.
        try:
            return TypeAdapter(result_type).dump_json(output).decode("utf-8")
        except Exception:
            pass

        # Fallbacks: Pydantic BaseModel, dataclasses, or generic JSON encoding
        try:
            return output.model_dump_json()  # Pydantic v2 BaseModel
        except Exception:
            try:
                import dataclasses as _dc
                import json as _json

                if _dc.is_dataclass(output):
                    return _json.dumps(
                        _dc.asdict(output),  # type: ignore
                        ensure_ascii=False,
                    )
            except Exception:
                pass

        try:
            import json as _json

            return _json.dumps(
                output,
                ensure_ascii=False,
                default=lambda o: getattr(o, "model_dump", lambda: str(o))(),
            )
        except Exception:
            return str(output)

    return _worker


def resolve_worker(
    worker: Optional[Callable[[str], Awaitable[str]]],
    model_name: Optional[str],
    openrouter_api_key: Optional[str],
    response_model: Optional[type],
) -> Callable[[str], Awaitable[str]]:
    """Resolve a worker function from explicit worker or model credentials/env."""
    if worker is not None:
        return worker

    # Try params; if missing, fall back to .env
    if model_name is None or openrouter_api_key is None:
        try:
            from dotenv import find_dotenv, load_dotenv  # type: ignore

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

    return make_pydantic_ai_worker(
        model_name=model_name,
        api_key=openrouter_api_key,
        result_type=response_model,
    )


__all__ = [
    "make_pydantic_ai_worker",
    "resolve_worker",
]

