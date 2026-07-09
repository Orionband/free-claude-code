"""Tests for NVIDIA NIM API key parsing and fair selection."""

from __future__ import annotations

from collections import Counter
from unittest.mock import patch

import pytest

from providers.nvidia_nim.keys import (
    FairRoundRobin,
    parse_nvidia_nim_api_keys,
    pick_nvidia_nim_api_key,
    reset_nvidia_nim_api_key_cycles,
)


@pytest.fixture(autouse=True)
def _reset_key_cycles() -> None:
    reset_nvidia_nim_api_key_cycles()
    yield
    reset_nvidia_nim_api_key_cycles()


def test_parse_nvidia_nim_api_keys_splits_trims_and_drops_empty() -> None:
    assert parse_nvidia_nim_api_keys("a,b,c") == ("a", "b", "c")
    assert parse_nvidia_nim_api_keys(" a , b ,c ") == ("a", "b", "c")
    assert parse_nvidia_nim_api_keys("only") == ("only",)
    assert parse_nvidia_nim_api_keys("") == ()
    assert parse_nvidia_nim_api_keys("  ,  , ") == ()


def test_pick_nvidia_nim_api_key_returns_empty_when_missing() -> None:
    assert pick_nvidia_nim_api_key("") == ""
    assert pick_nvidia_nim_api_key(" , ") == ""


def test_pick_nvidia_nim_api_key_returns_sole_key() -> None:
    assert pick_nvidia_nim_api_key("only-key") == "only-key"


def test_fair_round_robin_uses_each_item_once_per_round() -> None:
    cycle = FairRoundRobin(("a", "b", "c"))
    with patch(
        "providers.nvidia_nim.keys.random.randrange",
        side_effect=[1, 1, 0, 0, 0, 0],
    ):
        # remaining [a,b,c] -> pop index 1 -> b; [a,c] -> pop 1 -> c; [a] -> a
        first_round = [cycle.next(), cycle.next(), cycle.next()]
        # new round [a,b,c] -> pop 0 -> a; [b,c] -> pop 0 -> b; [c] -> c
        second_round = [cycle.next(), cycle.next(), cycle.next()]

    assert first_round == ["b", "c", "a"]
    assert second_round == ["a", "b", "c"]
    assert set(first_round) == {"a", "b", "c"}
    assert set(second_round) == {"a", "b", "c"}


def test_fair_round_robin_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one item"):
        FairRoundRobin(())


def test_pick_nvidia_nim_api_key_cycles_without_reuse_until_exhausted() -> None:
    keys = "key-a,key-b,key-c"
    with patch(
        "providers.nvidia_nim.keys.random.randrange",
        side_effect=[1, 0, 0, 2, 0, 0],
    ):
        # round 1: [a,b,c]->b; [a,c]->a; [c]->c
        first = [
            pick_nvidia_nim_api_key(keys),
            pick_nvidia_nim_api_key(keys),
            pick_nvidia_nim_api_key(keys),
        ]
        # round 2: [a,b,c]->c; [a,b]->a; [b]->b
        second = [
            pick_nvidia_nim_api_key(keys),
            pick_nvidia_nim_api_key(keys),
            pick_nvidia_nim_api_key(keys),
        ]

    assert first == ["key-b", "key-a", "key-c"]
    assert second == ["key-c", "key-a", "key-b"]
    assert Counter(first) == Counter({"key-a": 1, "key-b": 1, "key-c": 1})
    assert Counter(second) == Counter({"key-a": 1, "key-b": 1, "key-c": 1})


def test_pick_nvidia_nim_api_key_reuses_cycle_for_same_key_set() -> None:
    with patch(
        "providers.nvidia_nim.keys.random.randrange",
        side_effect=[0, 0],
    ):
        assert pick_nvidia_nim_api_key("a,b") == "a"
        # Same keys with different spacing still share the cycle.
        assert pick_nvidia_nim_api_key(" a , b ") == "b"
