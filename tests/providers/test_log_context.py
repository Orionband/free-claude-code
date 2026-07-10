"""Tests for provider log context helpers."""

from free_claude_code.providers.log_context import (
    resolve_upstream_api_key_for_log,
    should_log_upstream_transport_http_status,
    stamp_upstream_api_key_on_exception,
    upstream_api_key_from_exception,
    upstream_auth_failure_label,
)


def test_should_log_upstream_transport_http_status() -> None:
    assert should_log_upstream_transport_http_status(401)
    assert should_log_upstream_transport_http_status(403)
    assert should_log_upstream_transport_http_status(429)
    assert should_log_upstream_transport_http_status(500)
    assert not should_log_upstream_transport_http_status(400)
    assert not should_log_upstream_transport_http_status(None)


def test_upstream_auth_failure_label() -> None:
    assert upstream_auth_failure_label(401, exc_type="HTTPStatusError") == (
        "Authentication failed (401)"
    )
    assert upstream_auth_failure_label(403, exc_type="HTTPStatusError") == (
        "Forbidden (403)"
    )
    assert (
        upstream_auth_failure_label(None, exc_type="AuthenticationError")
        == "Authentication failed (401)"
    )
    assert upstream_auth_failure_label(429, exc_type="HTTPStatusError") is None


def test_stamp_and_resolve_upstream_api_key_on_exception() -> None:
    error = RuntimeError("boom")
    stamp_upstream_api_key_on_exception(error, "nvapi-key-1")
    assert upstream_api_key_from_exception(error) == "nvapi-key-1"
    assert (
        resolve_upstream_api_key_for_log(
            error, upstream_api_key="", fallback_api_key="fallback"
        )
        == "nvapi-key-1"
    )


def test_resolve_upstream_api_key_prefers_explicit_value() -> None:
    error = RuntimeError("boom")
    stamp_upstream_api_key_on_exception(error, "stamped")
    assert (
        resolve_upstream_api_key_for_log(
            error, upstream_api_key="explicit", fallback_api_key="fallback"
        )
        == "explicit"
    )
