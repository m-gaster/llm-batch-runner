from types import SimpleNamespace

import pytest

from llm_batch_runner.main import prompt_map
import llm_batch_runner.utils.workers as workers_module


@pytest.mark.asyncio
async def test_prompt_map_rejects_provider_preferences_with_custom_worker() -> None:
    async def worker(prompt: str) -> str:
        return prompt

    with pytest.raises(
        ValueError,
        match="`providers` and `allow_provider_fallbacks` can only be used",
    ):
        await prompt_map(
            ["alpha"],
            worker=worker,
            providers=["anthropic"],
            verbose=False,
        )


@pytest.mark.asyncio
async def test_builtin_worker_sends_openrouter_provider_preferences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent_settings: list[dict[str, object]] = []

    class FakeAgent:
        def __init__(self, model: object) -> None:
            self.model = model

        async def run(
            self, prompt: str, *, model_settings: dict[str, object]
        ) -> SimpleNamespace:
            sent_settings.append(model_settings)
            return SimpleNamespace(output=prompt.upper())

    monkeypatch.setattr(workers_module, "OpenAIChatModel", lambda **_: object())
    monkeypatch.setattr(workers_module, "OpenRouterProvider", lambda **_: object())
    monkeypatch.setattr(workers_module, "Agent", FakeAgent)

    worker = workers_module.make_pydantic_ai_worker(
        model_name="anthropic/claude-sonnet-4",
        api_key="test-key",
        providers=["anthropic", "amazon-bedrock"],
        allow_provider_fallbacks=False,
    )

    assert await worker("hello") == "HELLO"
    assert sent_settings == [
        {
            "temperature": 0.0,
            "timeout": 20.0,
            "extra_body": {
                "provider": {
                    "order": ["anthropic", "amazon-bedrock"],
                    "allow_fallbacks": False,
                }
            },
        }
    ]
