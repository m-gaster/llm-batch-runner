import time
from pathlib import Path

import pytest
from tenacity import wait_none

import llm_batch_runner.main as main_module
import llm_batch_runner.utils.rate_limits as rate_limits_module
from llm_batch_runner.main import prompt_map
from llm_batch_runner.utils.rate_limits import RollingWindowRateLimiter


def sqlite_url(path: Path) -> str:
    return f"sqlite+aiosqlite:///{path}"


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, duration: float) -> None:
        self.now += duration


@pytest.mark.asyncio
async def test_rolling_window_limiter_enforces_second_and_minute_windows() -> None:
    clock = FakeClock()
    limiter = RollingWindowRateLimiter(
        requests_per_minute=3,
        requests_per_second=2,
        clock=clock.monotonic,
        sleep=clock.sleep,
        minute_window_seconds=6.0,
        second_window_seconds=1.0,
    )

    await limiter.acquire()
    await limiter.acquire()
    assert clock.now == pytest.approx(0.0)

    await limiter.acquire()
    assert clock.now == pytest.approx(1.0)

    await limiter.acquire()
    assert clock.now == pytest.approx(6.0)


@pytest.mark.asyncio
async def test_prompt_map_uses_model_table_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        rate_limits_module,
        "MODEL_RATE_LIMITS",
        {
            "test-model": {
                "requests_per_minute": 120,
                "requests_per_second": 1,
            }
        },
    )
    start_times: list[float] = []

    async def worker(prompt: str) -> str:
        start_times.append(time.monotonic())
        return prompt.upper()

    results = await prompt_map(
        ["alpha", "beta"],
        worker=worker,
        model_name="test-model",
        cache_db_url=sqlite_url(tmp_path / "runs.db"),
        results_db_url=sqlite_url(tmp_path / "results.db"),
        max_attempts=1,
        verbose=False,
    )

    assert [row["result"] for row in results] == ["ALPHA", "BETA"]
    assert len(start_times) == 2
    assert start_times[1] - start_times[0] >= 0.95


@pytest.mark.asyncio
async def test_prompt_map_accepts_explicit_rate_limit_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(rate_limits_module, "MODEL_RATE_LIMITS", {})

    async def worker(prompt: str) -> str:
        return prompt[::-1]

    results = await prompt_map(
        ["alpha"],
        worker=worker,
        rate_limit={
            "requests_per_minute": 120,
            "requests_per_second": 10,
        },
        cache_db_url=sqlite_url(tmp_path / "runs.db"),
        results_db_url=sqlite_url(tmp_path / "results.db"),
        max_attempts=1,
        verbose=False,
    )

    assert results[0]["result"] == "ahpla"


@pytest.mark.asyncio
async def test_prompt_map_raises_for_missing_model_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rate_limits_module, "MODEL_RATE_LIMITS", {})

    async def worker(prompt: str) -> str:
        return prompt

    with pytest.raises(
        ValueError,
        match="No rate limit configured for model 'missing-model'",
    ):
        await prompt_map(
            ["alpha"],
            worker=worker,
            model_name="missing-model",
            max_attempts=1,
            verbose=False,
        )


@pytest.mark.asyncio
async def test_prompt_map_uses_model_from_env_for_rate_limit_lookup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("MODEL", "env-model")
    monkeypatch.setattr(
        rate_limits_module,
        "MODEL_RATE_LIMITS",
        {
            "env-model": {
                "requests_per_minute": 120,
                "requests_per_second": 10,
            }
        },
    )

    async def worker(prompt: str) -> str:
        return prompt.upper()

    results = await prompt_map(
        ["alpha"],
        worker=worker,
        cache_db_url=sqlite_url(tmp_path / "runs.db"),
        results_db_url=sqlite_url(tmp_path / "results.db"),
        max_attempts=1,
        verbose=False,
    )

    assert results[0]["result"] == "ALPHA"


@pytest.mark.asyncio
async def test_prompt_map_rate_limits_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(rate_limits_module, "MODEL_RATE_LIMITS", {})
    monkeypatch.setattr(main_module, "wait_random_exponential", lambda max: wait_none())
    start_times: list[float] = []
    attempts = 0

    async def flaky_worker(prompt: str) -> str:
        nonlocal attempts
        start_times.append(time.monotonic())
        attempts += 1
        if attempts < 3:
            raise RuntimeError("try again")
        return prompt.upper()

    results = await prompt_map(
        ["alpha"],
        worker=flaky_worker,
        rate_limit={
            "requests_per_minute": 120,
            "requests_per_second": 1,
        },
        cache_db_url=sqlite_url(tmp_path / "runs.db"),
        results_db_url=sqlite_url(tmp_path / "results.db"),
        max_attempts=3,
        verbose=False,
    )

    assert results[0]["result"] == "ALPHA"
    assert len(start_times) == 3
    assert start_times[1] - start_times[0] >= 0.95
    assert start_times[2] - start_times[1] >= 0.95
