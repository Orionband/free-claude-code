"""Shared mocks for :class:`providers.rate_limit.GlobalRateLimiter`."""

from collections.abc import Callable
from typing import Any

_EXECUTE_WITH_RETRY_CONTROL_KEYS = frozenset(
    {"api_key", "max_retries", "base_delay", "max_delay", "jitter"}
)


async def passthrough_execute_with_retry(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    """Invoke ``fn`` like :meth:`GlobalRateLimiter.execute_with_retry` without retries."""
    forwarded = {
        key: value
        for key, value in kwargs.items()
        if key not in _EXECUTE_WITH_RETRY_CONTROL_KEYS
    }
    return await fn(*args, **forwarded)
