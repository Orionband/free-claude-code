"""Detect and prepare Anthropic web server tool requests."""

from free_claude_code.api.models.anthropic import MessagesRequest, Tool

_WEB_SEARCH_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query to look up on the web.",
        }
    },
    "required": ["query"],
}

_WEB_FETCH_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "The URL to fetch and read.",
        }
    },
    "required": ["url"],
}

_WEB_SEARCH_DESCRIPTION = "Search the web for current information. Returns titles and URLs for matching pages."
_WEB_FETCH_DESCRIPTION = (
    "Fetch a URL and return readable page text for the model to use."
)


def request_text(request: MessagesRequest) -> str:
    """Join all user/assistant message content into one string for tool input parsing."""
    from .parsers import content_text

    return "\n".join(content_text(message.content) for message in request.messages)


def forced_tool_turn_text(request: MessagesRequest) -> str:
    """Text for parsing forced server-tool inputs: latest user turn only (avoids stale history)."""
    if not request.messages:
        return ""

    from .parsers import content_text

    for message in reversed(request.messages):
        if message.role == "user":
            return content_text(message.content)
    return ""


def forced_server_tool_name(request: MessagesRequest) -> str | None:
    """Return web_search or web_fetch only when tool_choice forces that server tool."""
    tc = request.tool_choice
    if not isinstance(tc, dict):
        return None
    if tc.get("type") != "tool":
        return None
    name = tc.get("name")
    if name in {"web_search", "web_fetch"}:
        return str(name)
    return None


def has_tool_named(request: MessagesRequest, name: str) -> bool:
    return any(tool.name == name for tool in request.tools or [])


def is_web_server_tool_request(request: MessagesRequest) -> bool:
    """True when the client forces a web server tool via tool_choice (not merely listed)."""
    forced = forced_server_tool_name(request)
    if forced is None:
        return False
    return has_tool_named(request, forced)


def is_anthropic_server_tool_definition(tool: Tool) -> bool:
    """Whether ``tool`` refers to an Anthropic server tool (web_search / web_fetch family)."""
    name = (tool.name or "").strip()
    if name in ("web_search", "web_fetch"):
        return True
    typ = tool.type
    if isinstance(typ, str):
        return typ.startswith("web_search") or typ.startswith("web_fetch")
    return False


def is_local_web_tool_name(name: str | None) -> bool:
    """True when ``name`` is a locally executable Anthropic web server tool."""
    return (name or "").strip() in {"web_search", "web_fetch"}


def openai_chat_upstream_server_tool_error(
    request: MessagesRequest, *, web_tools_enabled: bool
) -> str | None:
    """Return a user-facing error when a forced server tool cannot be handled locally."""
    forced = forced_server_tool_name(request)
    if forced and not web_tools_enabled:
        return (
            f"tool_choice forces Anthropic server tool {forced!r}, but local web server tools are "
            "disabled (ENABLE_WEB_SERVER_TOOLS=false). Enable them or use a native Anthropic "
            "Messages transport such as ollama or llama.cpp."
        )
    return None


def _synthetic_web_tool(tool: Tool) -> Tool | None:
    name = (tool.name or "").strip()
    typ = tool.type if isinstance(tool.type, str) else ""
    if name == "web_search" or typ.startswith("web_search"):
        return Tool(
            name="web_search",
            type="custom",
            description=tool.description or _WEB_SEARCH_DESCRIPTION,
            input_schema=_WEB_SEARCH_INPUT_SCHEMA,
        )
    if name == "web_fetch" or typ.startswith("web_fetch"):
        return Tool(
            name="web_fetch",
            type="custom",
            description=tool.description or _WEB_FETCH_DESCRIPTION,
            input_schema=_WEB_FETCH_INPUT_SCHEMA,
        )
    return None


def prepare_openai_chat_server_tools(
    request: MessagesRequest, *, web_tools_enabled: bool
) -> MessagesRequest:
    """Prepare listed Anthropic server tools for OpenAI-chat upstreams.

    When local web tools are enabled, rewrite ``web_search`` / ``web_fetch`` into callable
    function tools with real JSON schemas so NIM (and peers) can invoke them. FCC then
    executes those calls locally and returns Anthropic ``server_tool_use`` SSE.

    When disabled, strip listed server tools so the request does not 400. Forced
    ``tool_choice`` turns are left intact for the dedicated intercept / error path.
    """
    if forced_server_tool_name(request) is not None:
        return request
    tools = request.tools
    if not tools:
        return request

    if not web_tools_enabled:
        kept = [tool for tool in tools if not is_anthropic_server_tool_definition(tool)]
        if len(kept) == len(tools):
            return request
        return request.model_copy(update={"tools": kept or None})

    rewritten: list[Tool] = []
    changed = False
    for tool in tools:
        if not is_anthropic_server_tool_definition(tool):
            rewritten.append(tool)
            continue
        synthetic = _synthetic_web_tool(tool)
        if synthetic is None:
            changed = True
            continue
        if (
            tool.type != synthetic.type
            or tool.input_schema != synthetic.input_schema
            or (tool.description or "") != (synthetic.description or "")
        ):
            changed = True
        rewritten.append(synthetic)

    if not changed:
        return request
    return request.model_copy(update={"tools": rewritten or None})
