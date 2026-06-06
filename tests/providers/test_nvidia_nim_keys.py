"""Tests for NVIDIA NIM API key parsing and selection."""

from __future__ import annotations

from unittest.mock import patch

from providers.nvidia_nim.keys import parse_nvidia_nim_api_keys, pick_nvidia_nim_api_key


def test_parse_nvidia_nim_api_keys_splits_trims_and_drops_empty() -> None:
    assert parse_nvidia_nim_api_keys("a,b,c") == ("a", "b", "c")
    assert parse_nvidia_nim_api_keys(" a , b ,c ") == ("a", "b", "c")
    assert parse_nvidia_nim_api_keys("only") == ("only",)
    assert parse_nvidia_nim_api_keys("") == ()
    assert parse_nvidia_nim_api_keys("  ,  , ") == ()


def test_pick_nvidia_nim_api_key_returns_empty_when_missing() -> None:
    assert pick_nvidia_nim_api_key("") == ""
    assert pick_nvidia_nim_api_key(" , ") == ""


def test_pick_nvidia_nim_api_key_chooses_from_parsed_list() -> None:
    with patch(
        "providers.nvidia_nim.keys.random.choice",
        return_value="key-b",
    ) as choice:
        picked = pick_nvidia_nim_api_key("key-a,key-b,key-c")

    choice.assert_called_once_with(("key-a", "key-b", "key-c"))
    assert picked == "key-b"
