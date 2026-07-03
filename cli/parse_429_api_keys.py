"""Parse server logs for upstream API keys that received HTTP 401/429."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from config.paths import server_log_path

_API_KEY_RE = re.compile(r"api_key=([^\s,]+)")
_RATE_LIMIT_429_RE = re.compile(r"Rate limited \(429\)")
_AUTH_FAILED_401_RE = re.compile(r"Authentication failed \(401\)")
_FORBIDDEN_403_RE = re.compile(r"Forbidden \(403\)")
_HTTP_STATUS_RE = re.compile(r"http_status=(\d+)\b")

DEFAULT_STATUSES = frozenset({401, 429})


@dataclass(frozen=True, slots=True)
class UpstreamApiKeyHit:
    """One log row where an upstream API key hit HTTP 401/403/429."""

    time: str
    status_code: int
    api_key: str
    kind: str
    message: str


def _classify_message(message: str) -> tuple[int, str] | None:
    if _RATE_LIMIT_429_RE.search(message):
        if "retry exhausted" in message:
            return 429, "exhausted"
        if "Retrying in" in message:
            return 429, "retry"
        return 429, "rate_limited"
    if _AUTH_FAILED_401_RE.search(message):
        return 401, "auth_failed"
    if _FORBIDDEN_403_RE.search(message):
        return 403, "auth_failed"
    status_match = _HTTP_STATUS_RE.search(message)
    if status_match is not None:
        status_code = int(status_match.group(1))
        if status_code in {401, 403, 429}:
            return status_code, "transport_error"
    return None


def _extract_api_key(message: str) -> str | None:
    match = _API_KEY_RE.search(message)
    if match is None:
        return None
    key = match.group(1)
    if key in {"", "<redacted>"}:
        return None
    return key


def parse_log_line(line: str) -> UpstreamApiKeyHit | None:
    """Return an upstream auth/rate-limit hit from one JSON log line, or None."""
    line = line.strip()
    if not line:
        return None
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(row, dict):
        return None

    message = row.get("message")
    if not isinstance(message, str):
        return None

    classified = _classify_message(message)
    if classified is None:
        return None

    status_code, kind = classified
    api_key = _extract_api_key(message)
    if api_key is None:
        return None

    time = row.get("time")
    return UpstreamApiKeyHit(
        time=str(time) if time is not None else "",
        status_code=status_code,
        api_key=api_key,
        kind=kind,
        message=message,
    )


def iter_upstream_api_key_hits(
    log_path: Path,
    *,
    statuses: frozenset[int] = DEFAULT_STATUSES,
) -> Iterator[UpstreamApiKeyHit]:
    """Yield matching hits from ``log_path`` in file order."""
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            hit = parse_log_line(line)
            if hit is not None and hit.status_code in statuses:
                yield hit


def iter_429_api_key_hits(log_path: Path) -> Iterator[UpstreamApiKeyHit]:
    """Backward-compatible alias for 429-only parsing."""
    return iter_upstream_api_key_hits(log_path, statuses=frozenset({429}))


def filter_hits(
    hits: Iterator[UpstreamApiKeyHit],
    *,
    unique: bool = False,
    dedupe_consecutive: bool = False,
) -> list[UpstreamApiKeyHit]:
    """Apply optional deduplication while preserving order."""
    out: list[UpstreamApiKeyHit] = []
    seen: set[str] = set()
    previous_key: str | None = None

    for hit in hits:
        if dedupe_consecutive and hit.api_key == previous_key:
            continue
        if unique:
            if hit.api_key in seen:
                previous_key = hit.api_key
                continue
            seen.add(hit.api_key)
        out.append(hit)
        previous_key = hit.api_key
    return out


def format_hit(hit: UpstreamApiKeyHit, *, verbose: bool) -> str:
    if verbose:
        return (
            f"{hit.time}\t{hit.status_code}\t{hit.kind}\t{hit.api_key}\t{hit.message}"
        )
    return hit.api_key


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "List upstream API keys that hit HTTP 401 or 429 in server.log, "
            "in chronological order."
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=f"Log file to read (default: {server_log_path()})",
    )
    parser.add_argument(
        "--status",
        type=int,
        action="append",
        dest="statuses",
        help="HTTP status to include (repeatable). Default: 401 and 429.",
    )
    parser.add_argument(
        "--unique",
        action="store_true",
        help="Only print the first hit per API key (still chronological).",
    )
    parser.add_argument(
        "--dedupe-consecutive",
        action="store_true",
        help="Drop back-to-back rows for the same API key.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print timestamp, status, kind, api_key, and message (tab-separated).",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Print the number of matching rows instead of the keys.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    log_path = args.log_file if args.log_file is not None else server_log_path()
    statuses = frozenset(args.statuses) if args.statuses else DEFAULT_STATUSES

    if not log_path.is_file():
        print(f"Log file not found: {log_path}", file=sys.stderr)
        return 1

    hits = filter_hits(
        iter_upstream_api_key_hits(log_path, statuses=statuses),
        unique=args.unique,
        dedupe_consecutive=args.dedupe_consecutive,
    )

    if args.count:
        print(len(hits))
        return 0

    if not hits:
        status_label = ", ".join(str(code) for code in sorted(statuses))
        print(
            f"No API key entries found for HTTP status {status_label}.",
            file=sys.stderr,
        )
        return 0

    for hit in hits:
        print(format_hit(hit, verbose=args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
