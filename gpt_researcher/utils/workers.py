from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Any

from .rate_limiter import get_global_rate_limiter


class WorkerPool:
    def __init__(self, max_workers: int | None = None, rate_limit_delay: float = 0.0):
        """
        Initialize WorkerPool with concurrency and global rate limiting support.

        Args:
            max_workers: Maximum number of concurrent workers. Defaults to 5x CPU count.
            rate_limit_delay: Minimum seconds between requests applied globally
                              across all worker pools (0 disables rate limiting).
        """
        if max_workers is None:
            max_workers = (os.cpu_count() or 1) * 5

        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)

        global_limiter = get_global_rate_limiter()
        global_limiter.configure(rate_limit_delay)

    @asynccontextmanager
    async def throttle(self) -> AsyncGenerator[Any, None]:
        """
        Limit concurrent work and enforce the shared global rate limiter.
        """
        async with self.semaphore:
            global_limiter = get_global_rate_limiter()
            await global_limiter.wait_if_needed()
            yield
