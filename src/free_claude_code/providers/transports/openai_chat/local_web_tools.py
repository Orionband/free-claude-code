"""Accumulate OpenAI-chat tool_call deltas for local web_search / web_fetch execution."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from free_claude_code.api.web_tools.request import is_local_web_tool_name
from free_claude_code.api.web_tools.streaming import parse_tool_arguments_json


@dataclass
class PendingLocalWebToolCall:
    """One in-progress or completed local web tool call from an OpenAI-chat upstream."""

    name: str
    tool_id: str
    arguments: str = ""


@dataclass
class LocalWebToolCallBuffer:
    """Track web_search / web_fetch tool_call deltas without emitting client tool_use SSE."""

    _by_index: dict[int, PendingLocalWebToolCall] = field(default_factory=dict)
    _name_by_index: dict[int, str] = field(default_factory=dict)

    def observe(self, tc: dict[str, object]) -> bool:
        """Ingest one OpenAI tool_call delta.

        Returns True when this delta belongs to a local web tool and should not be
        emitted as a normal Anthropic ``tool_use`` block.
        """
        raw_index = tc.get("index", 0)
        tc_index = raw_index if isinstance(raw_index, int) else 0
        if tc_index < 0:
            tc_index = len(self._by_index)

        fn_delta = tc.get("function", {})
        incoming_name = ""
        arguments = ""
        if isinstance(fn_delta, dict):
            raw_name = fn_delta.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                incoming_name = raw_name.strip()
            raw_args = fn_delta.get("arguments", "") or ""
            if isinstance(raw_args, str):
                arguments = raw_args

        if incoming_name:
            self._name_by_index[tc_index] = incoming_name

        resolved_name = self._name_by_index.get(tc_index, "")
        if not is_local_web_tool_name(resolved_name) and tc_index not in self._by_index:
            return False

        if tc_index not in self._by_index:
            if not is_local_web_tool_name(resolved_name):
                return False
            raw_id = tc.get("id")
            tool_id = (
                str(raw_id)
                if isinstance(raw_id, str) and raw_id
                else f"srvtoolu_{uuid.uuid4().hex}"
            )
            self._by_index[tc_index] = PendingLocalWebToolCall(
                name=resolved_name,
                tool_id=tool_id,
            )

        pending = self._by_index[tc_index]
        if incoming_name:
            pending.name = incoming_name
        raw_id = tc.get("id")
        if isinstance(raw_id, str) and raw_id:
            pending.tool_id = raw_id
        if arguments:
            pending.arguments += arguments
        return True

    def completed_calls(self) -> list[tuple[str, str, dict[str, object]]]:
        """Return completed ``(name, tool_id, input_dict)`` tuples in stream order."""
        completed: list[tuple[str, str, dict[str, object]]] = []
        for index in sorted(self._by_index):
            pending = self._by_index[index]
            if not is_local_web_tool_name(pending.name):
                continue
            completed.append(
                (
                    pending.name,
                    pending.tool_id,
                    parse_tool_arguments_json(pending.arguments),
                )
            )
        return completed

    @property
    def has_calls(self) -> bool:
        return bool(self._by_index)
