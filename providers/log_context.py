"""Shared log suffix helpers for provider transports."""


def format_api_key_for_log(api_key: str | None) -> str:
    """Return a log suffix identifying the upstream API key, if known."""
    if not api_key:
        return ""
    return f" api_key={api_key}"
