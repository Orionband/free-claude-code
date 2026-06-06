"""NVIDIA NIM API key parsing and selection."""

from __future__ import annotations

import random


def parse_nvidia_nim_api_keys(raw: str) -> tuple[str, ...]:
    """Return non-empty API keys from a comma-separated env value."""
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def pick_nvidia_nim_api_key(raw: str) -> str:
    """Pick one API key at random from a comma-separated env value."""
    keys = parse_nvidia_nim_api_keys(raw)
    if not keys:
        return ""
    return random.choice(keys)
