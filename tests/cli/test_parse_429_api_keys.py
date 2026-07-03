"""Tests for upstream API key log parsing (401/429)."""

import json
from pathlib import Path

from cli.parse_429_api_keys import (
    filter_hits,
    iter_429_api_key_hits,
    iter_upstream_api_key_hits,
    main,
    parse_log_line,
)


def _row(*, time: str, message: str) -> str:
    return json.dumps({"time": time, "level": "WARNING", "message": message})


def test_parse_retry_line() -> None:
    hit = parse_log_line(
        _row(
            time="2026-07-02T22:00:00",
            message=(
                "Rate limited (429) api_key=nvapi-key-a, attempt 1/5. "
                "Retrying in 2.0s..."
            ),
        )
    )
    assert hit is not None
    assert hit.api_key == "nvapi-key-a"
    assert hit.status_code == 429
    assert hit.kind == "retry"
    assert hit.time == "2026-07-02T22:00:00"


def test_parse_auth_failed_line() -> None:
    hit = parse_log_line(
        _row(
            time="2026-07-02T22:00:01",
            message=(
                "Authentication failed (401) api_key=nvapi-key-auth "
                "exc_type=HTTPStatusError"
            ),
        )
    )
    assert hit is not None
    assert hit.api_key == "nvapi-key-auth"
    assert hit.status_code == 401
    assert hit.kind == "auth_failed"


def test_parse_exhausted_line() -> None:
    hit = parse_log_line(
        _row(
            time="2026-07-02T22:00:05",
            message=(
                "Rate limited (429) api_key=nvapi-key-b retry exhausted "
                "after 4 retries (attempts=5)"
            ),
        )
    )
    assert hit is not None
    assert hit.api_key == "nvapi-key-b"
    assert hit.kind == "exhausted"


def test_parse_transport_error_line() -> None:
    hit = parse_log_line(
        _row(
            time="2026-07-02T22:00:10",
            message=(
                "NIM_ERROR: api_key=nvapi-key-c exc_type=HTTPStatusError "
                "http_status=429"
            ),
        )
    )
    assert hit is not None
    assert hit.api_key == "nvapi-key-c"
    assert hit.status_code == 429
    assert hit.kind == "transport_error"


def test_parse_transport_error_401_line() -> None:
    hit = parse_log_line(
        _row(
            time="2026-07-02T22:00:11",
            message=(
                "NIM_ERROR: api_key=nvapi-key-bad exc_type=HTTPStatusError "
                "http_status=401"
            ),
        )
    )
    assert hit is not None
    assert hit.api_key == "nvapi-key-bad"
    assert hit.status_code == 401
    assert hit.kind == "transport_error"


def test_skips_redacted_and_unrelated_status() -> None:
    assert (
        parse_log_line(
            _row(
                time="t1",
                message="Rate limited (429) api_key=<redacted>, attempt 1/5.",
            )
        )
        is None
    )
    assert (
        parse_log_line(
            _row(
                time="t2",
                message="NIM_ERROR: api_key=nvapi-key-d exc_type=TimeoutError",
            )
        )
        is None
    )


def test_iter_hits_in_file_order(tmp_path: Path) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text(
        "\n".join(
            [
                _row(
                    time="t1",
                    message="Rate limited (429) api_key=key-1, attempt 1/5.",
                ),
                _row(
                    time="t2",
                    message=(
                        "Authentication failed (401) api_key=key-2 "
                        "exc_type=HTTPStatusError"
                    ),
                ),
                _row(
                    time="t3",
                    message="Rate limited (429) api_key=key-1, attempt 2/5.",
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    hits = list(iter_upstream_api_key_hits(log_path))
    assert [hit.api_key for hit in hits] == ["key-1", "key-2", "key-1"]


def test_iter_429_alias_filters_to_429_only(tmp_path: Path) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text(
        _row(
            time="t1",
            message="Authentication failed (401) api_key=key-auth exc_type=X",
        )
        + "\n"
        + _row(
            time="t2",
            message="Rate limited (429) api_key=key-rate, attempt 1/5.",
        )
        + "\n",
        encoding="utf-8",
    )

    assert [hit.api_key for hit in iter_429_api_key_hits(log_path)] == ["key-rate"]


def test_filter_unique_preserves_first_seen_order() -> None:
    rows = [
        parse_log_line(
            _row(
                time="t1",
                message="Rate limited (429) api_key=key-1, attempt 1/5.",
            )
        ),
        parse_log_line(
            _row(
                time="t2",
                message="Rate limited (429) api_key=key-2, attempt 1/5.",
            )
        ),
        parse_log_line(
            _row(
                time="t3",
                message="Rate limited (429) api_key=key-1, attempt 2/5.",
            )
        ),
    ]
    hits = [row for row in rows if row is not None]
    filtered = filter_hits(iter(hits), unique=True)
    assert [hit.api_key for hit in filtered] == ["key-1", "key-2"]


def test_main_prints_keys_in_order(tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text(
        _row(
            time="t1",
            message="Rate limited (429) api_key=alpha, attempt 1/5.",
        )
        + "\n"
        + _row(
            time="t2",
            message=(
                "Authentication failed (401) api_key=beta exc_type=HTTPStatusError"
            ),
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["--log-file", str(log_path)]) == 0
    assert capsys.readouterr().out.splitlines() == ["alpha", "beta"]


def test_main_status_filter(tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_text(
        _row(
            time="t1",
            message="Rate limited (429) api_key=alpha, attempt 1/5.",
        )
        + "\n"
        + _row(
            time="t2",
            message=(
                "Authentication failed (401) api_key=beta exc_type=HTTPStatusError"
            ),
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["--log-file", str(log_path), "--status", "401"]) == 0
    assert capsys.readouterr().out.splitlines() == ["beta"]
