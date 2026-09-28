import asyncio
import time
from collections import deque
from typing import Awaitable, Callable, Mapping, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, PositiveInt, ValidationError


class RateLimitConfig(TypedDict):
    requests_per_minute: int
    requests_per_second: int


MODEL_RATE_LIMITS: dict[str, RateLimitConfig] = {}


class _RateLimitModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests_per_minute: PositiveInt
    requests_per_second: PositiveInt


def validate_rate_limit(
    rate_limit: Mapping[str, object],
    *,
    source: str = "rate_limit",
) -> RateLimitConfig:
    """Validate and normalize a rate-limit mapping."""
    try:
        parsed = _RateLimitModel.model_validate(rate_limit)
    except ValidationError as exc:
        raise ValueError(f"Invalid {source}: {exc}") from exc

    return {
        "requests_per_minute": int(parsed.requests_per_minute),
        "requests_per_second": int(parsed.requests_per_second),
    }


def resolve_rate_limit(
    *,
    model_name: Optional[str],
    rate_limit: Optional[Mapping[str, object]],
    model_rate_limits: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> RateLimitConfig:
    """Resolve the effective rate limit for a run."""
    if rate_limit is not None:
        return validate_rate_limit(rate_limit, source="rate_limit")

    limits_table = MODEL_RATE_LIMITS if model_rate_limits is None else model_rate_limits
    if not model_name:
        raise ValueError(
            "No model name available for rate limit lookup. "
            "Pass `model_name`, set `MODEL`, or provide `rate_limit`."
        )

    model_limit = limits_table.get(model_name)
    if model_limit is None:
        raise ValueError(
            f"No rate limit configured for model {model_name!r}. "
            "Add an entry to `MODEL_RATE_LIMITS` or pass `rate_limit`."
        )

    return validate_rate_limit(
        model_limit,
        source=f"MODEL_RATE_LIMITS[{model_name!r}]",
    )


class RollingWindowRateLimiter:
    """Async rolling-window limiter enforcing both second and minute caps."""

    def __init__(
        self,
        *,
        requests_per_minute: int,
        requests_per_second: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        minute_window_seconds: float = 60.0,
        second_window_seconds: float = 1.0,
    ):
        self._config = validate_rate_limit(
            {
                "requests_per_minute": requests_per_minute,
                "requests_per_second": requests_per_second,
            }
        )
        self._clock = clock
        self._sleep = sleep
        self._minute_window_seconds = minute_window_seconds
        self._second_window_seconds = second_window_seconds
        self._minute_timestamps: deque[float] = deque()
        self._second_timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block until both the second and minute windows can accept a request."""
        while True:
            async with self._lock:
                now = self._clock()
                self._trim(self._minute_timestamps, now, self._minute_window_seconds)
                self._trim(self._second_timestamps, now, self._second_window_seconds)

                wait_for = max(
                    self._required_wait(
                        self._minute_timestamps,
                        self._config["requests_per_minute"],
                        self._minute_window_seconds,
                        now,
                    ),
                    self._required_wait(
                        self._second_timestamps,
                        self._config["requests_per_second"],
                        self._second_window_seconds,
                        now,
                    ),
                )
                if wait_for <= 0:
                    self._minute_timestamps.append(now)
                    self._second_timestamps.append(now)
                    return

            await self._sleep(wait_for)

    @staticmethod
    def _trim(history: deque[float], now: float, window_seconds: float) -> None:
        cutoff = now - window_seconds
        while history and history[0] <= cutoff:
            history.popleft()

    @staticmethod
    def _required_wait(
        history: deque[float],
        limit: int,
        window_seconds: float,
        now: float,
    ) -> float:
        if len(history) < limit:
            return 0.0
        return max(0.0, history[0] + window_seconds - now)


__all__ = [
    "MODEL_RATE_LIMITS",
    "RateLimitConfig",
    "RollingWindowRateLimiter",
    "resolve_rate_limit",
    "validate_rate_limit",
]
