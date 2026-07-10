"""SSE streaming for local web_search / web_fetch server tool results."""

import json
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from free_claude_code.api.models.anthropic import MessagesRequest
from free_claude_code.core.anthropic.server_tool_sse import (
    SERVER_TOOL_USE,
    WEB_FETCH_TOOL_ERROR,
    WEB_FETCH_TOOL_RESULT,
    WEB_SEARCH_TOOL_RESULT,
    WEB_SEARCH_TOOL_RESULT_ERROR,
)
from free_claude_code.core.anthropic.streaming import format_sse_event

from . import outbound
from .constants import _MAX_FETCH_CHARS
from .egress import WebFetchEgressPolicy
from .parsers import extract_query, extract_url
from .request import (
    forced_server_tool_name,
    forced_tool_turn_text,
    has_tool_named,
    is_local_web_tool_name,
)


def _search_summary(query: str, results: list[dict[str, str]]) -> str:
    if not results:
        return f"No web search results found for: {query}"
    lines = [f"Search results for: {query}"]
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. {result['title']}\n{result['url']}")
    return "\n\n".join(lines)


def normalize_web_tool_input(
    tool_name: str, tool_input: dict[str, Any]
) -> dict[str, str]:
    """Normalize model-provided tool arguments into the local executor shape."""
    if tool_name == "web_search":
        query = tool_input.get("query")
        if not isinstance(query, str) or not query.strip():
            query = str(tool_input.get("q") or tool_input.get("search") or "").strip()
        return {"query": query}
    url = tool_input.get("url")
    if not isinstance(url, str) or not url.strip():
        url = str(tool_input.get("uri") or tool_input.get("href") or "").strip()
    return {"url": url}


async def execute_local_web_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    *,
    web_fetch_egress: WebFetchEgressPolicy,
    verbose_client_errors: bool = False,
) -> tuple[str, Any, str]:
    """Run one local web tool.

    Returns ``(result_block_type, result_content, summary_text)``.
    """
    normalized = normalize_web_tool_input(tool_name, tool_input)
    result_block_for_tool = {
        "web_search": WEB_SEARCH_TOOL_RESULT,
        "web_fetch": WEB_FETCH_TOOL_RESULT,
    }
    error_payload_type_for_tool = {
        "web_search": WEB_SEARCH_TOOL_RESULT_ERROR,
        "web_fetch": WEB_FETCH_TOOL_ERROR,
    }
    try:
        if tool_name == "web_search":
            query = normalized["query"]
            results = await outbound._run_web_search(query)
            result_content: Any = [
                {
                    "type": "web_search_result",
                    "title": result["title"],
                    "url": result["url"],
                }
                for result in results
            ]
            summary = _search_summary(query, results)
            return WEB_SEARCH_TOOL_RESULT, result_content, summary

        fetched = await outbound._run_web_fetch(normalized["url"], web_fetch_egress)
        result_content = {
            "type": "web_fetch_result",
            "url": fetched["url"],
            "content": {
                "type": "document",
                "source": {
                    "type": "text",
                    "media_type": fetched["media_type"],
                    "data": fetched["data"],
                },
                "title": fetched["title"],
                "citations": {"enabled": True},
            },
            "retrieved_at": datetime.now(UTC).isoformat(),
        }
        summary = fetched["data"][:_MAX_FETCH_CHARS]
        return WEB_FETCH_TOOL_RESULT, result_content, summary
    except Exception as error:
        fetch_url = normalized.get("url") if tool_name == "web_fetch" else None
        outbound._log_web_tool_failure(tool_name, error, fetch_url=fetch_url)
        return (
            result_block_for_tool[tool_name],
            {
                "type": error_payload_type_for_tool[tool_name],
                "error_code": "unavailable",
            },
            outbound._web_tool_client_error_summary(
                tool_name, error, verbose=verbose_client_errors
            ),
        )


async def iter_local_web_tool_sse(
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    model: str,
    input_tokens: int,
    web_fetch_egress: WebFetchEgressPolicy,
    verbose_client_errors: bool = False,
    message_id: str | None = None,
    starting_block_index: int = 0,
    include_message_envelope: bool = True,
    tool_id: str | None = None,
) -> AsyncIterator[str]:
    """Emit Anthropic SSE for one locally executed web_search / web_fetch call."""
    if not is_local_web_tool_name(tool_name):
        return

    normalized = normalize_web_tool_input(tool_name, tool_input)
    resolved_message_id = message_id or f"msg_{uuid.uuid4()}"
    resolved_tool_id = tool_id or f"srvtoolu_{uuid.uuid4().hex}"
    usage_key = (
        "web_search_requests" if tool_name == "web_search" else "web_fetch_requests"
    )
    block_index = starting_block_index

    if include_message_envelope:
        yield format_sse_event(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": resolved_message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": input_tokens, "output_tokens": 1},
                },
            },
        )

    yield format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {
                "type": SERVER_TOOL_USE,
                "id": resolved_tool_id,
                "name": tool_name,
                "input": normalized,
            },
        },
    )
    yield format_sse_event(
        "content_block_stop", {"type": "content_block_stop", "index": block_index}
    )
    block_index += 1

    result_block_type, result_content, summary = await execute_local_web_tool(
        tool_name,
        normalized,
        web_fetch_egress=web_fetch_egress,
        verbose_client_errors=verbose_client_errors,
    )
    output_tokens = max(1, len(summary) // 4)

    yield format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {
                "type": result_block_type,
                "tool_use_id": resolved_tool_id,
                "content": result_content,
            },
        },
    )
    yield format_sse_event(
        "content_block_stop", {"type": "content_block_stop", "index": block_index}
    )
    block_index += 1

    yield format_sse_event(
        "content_block_start",
        {
            "type": "content_block_start",
            "index": block_index,
            "content_block": {"type": "text", "text": ""},
        },
    )
    yield format_sse_event(
        "content_block_delta",
        {
            "type": "content_block_delta",
            "index": block_index,
            "delta": {"type": "text_delta", "text": summary},
        },
    )
    yield format_sse_event(
        "content_block_stop", {"type": "content_block_stop", "index": block_index}
    )

    if include_message_envelope:
        yield format_sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "server_tool_use": {usage_key: 1},
                },
            },
        )
        yield format_sse_event("message_stop", {"type": "message_stop"})


async def stream_web_server_tool_response(
    request: MessagesRequest,
    input_tokens: int,
    *,
    web_fetch_egress: WebFetchEgressPolicy,
    verbose_client_errors: bool = False,
) -> AsyncIterator[str]:
    """Stream a minimal Anthropic-shaped turn for forced `web_search` / `web_fetch`."""
    tool_name = forced_server_tool_name(request)
    if tool_name is None or not has_tool_named(request, tool_name):
        return

    text = forced_tool_turn_text(request)
    tool_input = (
        {"query": extract_query(text)}
        if tool_name == "web_search"
        else {"url": extract_url(text)}
    )
    async for event in iter_local_web_tool_sse(
        tool_name=tool_name,
        tool_input=tool_input,
        model=request.model,
        input_tokens=input_tokens,
        web_fetch_egress=web_fetch_egress,
        verbose_client_errors=verbose_client_errors,
        include_message_envelope=True,
    ):
        yield event


def parse_tool_arguments_json(raw: str) -> dict[str, Any]:
    """Parse streamed tool argument JSON into a dict (empty on failure)."""
    text = (raw or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
