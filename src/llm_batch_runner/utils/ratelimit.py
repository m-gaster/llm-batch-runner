import asyncio
import time
from contextlib import asynccontextmanager


class AsyncTokenBucket:
    """Simple async token-bucket rate limiter.

    - rate: tokens added per second (float).
    - capacity: maximum burst size (int >= 1).

    Usage:
        bucket = AsyncTokenBucket(rate=50.0, capacity=100)
        async with bucket.limit():
            await do_io()
    """

    def __init__(self, *, rate: float, capacity: int) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._rate = float(rate)
        self._capacity = int(capacity)
        self._tokens = float(capacity)
        self._updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._updated_at
                self._updated_at = now
                # Refill based on elapsed time
                self._tokens = min(
                    self._capacity, self._tokens + elapsed * self._rate
                )
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Need to wait for enough tokens to accumulate
                needed = 1.0 - self._tokens
                wait_time = max(needed / self._rate, 0.0)
            # Sleep outside the lock
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(0)

    @asynccontextmanager
    async def limit(self):
        await self.acquire()
        try:
            yield
        finally:
            # Token consumption is one-way; no release on exit
            pass


__all__ = ["AsyncTokenBucket"]

