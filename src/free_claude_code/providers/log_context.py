"""Shared log suffix helpers for provider transports."""

UPSTREAM_API_KEY_EXCEPTION_ATTR = "_fcc_upstream_api_key"


def format_api_key_for_log(api_key: str | None) -> str:
    """Return a log suffix identifying the upstream API key, if known."""
    if not api_key:
        return ""
    return f" api_key={api_key}"


_LOGGED_UPSTREAM_HTTP_STATUSES = frozenset({401, 403, 429})


def should_log_upstream_transport_http_status(status: int | None) -> bool:
    """Return whether an upstream HTTP status should emit transport error logs."""
    if not isinstance(status, int):
        return False
    return status in _LOGGED_UPSTREAM_HTTP_STATUSES or status >= 500


def upstream_auth_failure_label(status: int | None, *, exc_type: str) -> str | None:
    """Return a human-readable auth failure label for warning logs."""
    if status == 401:
        return "Authentication failed (401)"
    if status == 403:
        return "Forbidden (403)"
    if status is None and exc_type == "AuthenticationError":
        return "Authentication failed (401)"
    return None


def stamp_upstream_api_key_on_exception(
    error: BaseException, api_key: str | None
) -> None:
    """Attach the upstream credential to an exception for later logging."""
    if not api_key:
        return
    if getattr(error, UPSTREAM_API_KEY_EXCEPTION_ATTR, None):
        return
    setattr(error, UPSTREAM_API_KEY_EXCEPTION_ATTR, api_key)


def upstream_api_key_from_exception(error: BaseException) -> str | None:
    """Return an upstream API key previously stamped on ``error``."""
    key = getattr(error, UPSTREAM_API_KEY_EXCEPTION_ATTR, None)
    if isinstance(key, str) and key:
        return key
    return None


def resolve_upstream_api_key_for_log(
    error: BaseException,
    *,
    upstream_api_key: str | None = None,
    fallback_api_key: str | None = None,
) -> str | None:
    """Pick the best upstream API key available for transport error logs."""
    if upstream_api_key:
        return upstream_api_key
    stamped = upstream_api_key_from_exception(error)
    if stamped:
        return stamped
    if fallback_api_key:
        return fallback_api_key
    return None


def normalize_upstream_http_status(
    error: Exception, response: object | None
) -> int | None:
    """Best-effort HTTP status from an upstream transport exception."""
    http_status = (
        getattr(response, "status_code", None) if response is not None else None
    )
    if isinstance(http_status, int):
        return http_status
    sdk_status = getattr(error, "status_code", None)
    if isinstance(sdk_status, int):
        return sdk_status
    return None
