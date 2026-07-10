"""NVIDIA NIM provider package."""

from free_claude_code.providers.defaults import NVIDIA_NIM_DEFAULT_BASE

from .client import NvidiaNimProvider
from .keys import (
    FairRoundRobin,
    parse_nvidia_nim_api_keys,
    pick_nvidia_nim_api_key,
    reset_nvidia_nim_api_key_cycles,
)

__all__ = [
    "NVIDIA_NIM_DEFAULT_BASE",
    "FairRoundRobin",
    "NvidiaNimProvider",
    "parse_nvidia_nim_api_keys",
    "pick_nvidia_nim_api_key",
    "reset_nvidia_nim_api_key_cycles",
]
