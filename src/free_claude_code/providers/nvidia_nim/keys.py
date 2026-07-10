"""NVIDIA NIM API key parsing and fair round-robin selection."""

import random
import threading


def parse_nvidia_nim_api_keys(raw: str) -> tuple[str, ...]:
    """Return non-empty API keys from a comma-separated env value."""
    return tuple(part.strip() for part in raw.split(",") if part.strip())


class FairRoundRobin[T]:
    """Pick items so each is used once before any repeats.

    Each round starts by choosing randomly among the full set. Subsequent
    picks choose randomly among the remaining unused items. When the pool
    is empty, a new round begins.
    """

    def __init__(self, items: tuple[T, ...]) -> None:
        if not items:
            raise ValueError("FairRoundRobin requires at least one item")
        self._items = items
        self._remaining: list[T] = []
        self._lock = threading.Lock()

    def next(self) -> T:
        with self._lock:
            if not self._remaining:
                self._remaining = list(self._items)
            index = random.randrange(len(self._remaining))
            return self._remaining.pop(index)


_pick_cycles: dict[tuple[str, ...], FairRoundRobin[str]] = {}
_pick_cycles_lock = threading.Lock()


def reset_nvidia_nim_api_key_cycles() -> None:
    """Clear cached key cycles (for tests)."""
    with _pick_cycles_lock:
        _pick_cycles.clear()


def pick_nvidia_nim_api_key(raw: str) -> str:
    """Pick the next API key with fair round-robin from a comma-separated value.

    Starts each round with a random unused key; a key is not reused until
    every other configured key has been used once.
    """
    keys = parse_nvidia_nim_api_keys(raw)
    if not keys:
        return ""
    if len(keys) == 1:
        return keys[0]
    with _pick_cycles_lock:
        cycle = _pick_cycles.get(keys)
        if cycle is None:
            cycle = FairRoundRobin(keys)
            _pick_cycles[keys] = cycle
    return cycle.next()
