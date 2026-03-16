"""JSON trace parser for AgentDoctor."""

from __future__ import annotations

from typing import Any

from agentdoctor.models import Message, Role, ToolCall, Trace
from agentdoctor.parsers.base import BaseParser

_ROLE_MAP: dict[str, Role] = {r.value: r for r in Role}


class JSONParser(BaseParser):
    """Parse JSON traces into :class:`Trace` objects.

    Accepts three input forms:

    - **str** — treated as a file path; loaded via :meth:`_load_file`.
    - **dict** — expected to have a ``messages`` key (list of message dicts),
      plus optional ``trace_id`` and ``metadata``.
    - **list** — treated as a bare list of message dicts.
    """

    def parse(self, source: str | dict | list) -> Trace:
        if isinstance(source, str):
            data = self._load_file(source)
            if isinstance(data, list):
                return self._parse_messages(data)
            if isinstance(data, dict):
                return self._parse_dict(data)
            raise TypeError(f"Unsupported JSON root type: {type(data).__name__}")
        if isinstance(source, dict):
            return self._parse_dict(source)
        if isinstance(source, list):
            return self._parse_messages(source)
        raise TypeError(f"Unsupported source type: {type(source).__name__}")

    def _parse_dict(self, data: dict) -> Trace:
        if "messages" not in data:
            raise ValueError("Dict source must contain a 'messages' key")
        messages = self._build_messages(data["messages"])
        return Trace(
            messages=messages,
            trace_id=data.get("trace_id"),
            metadata=data.get("metadata", {}),
        )

    def _parse_messages(self, raw_messages: list[dict[str, Any]]) -> Trace:
        return Trace(messages=self._build_messages(raw_messages))

    def _build_messages(self, raw_messages: list[dict[str, Any]]) -> list[Message]:
        messages: list[Message] = []
        for idx, raw in enumerate(raw_messages):
            role_str = raw.get("role", "")
            if role_str not in _ROLE_MAP:
                raise ValueError(
                    f"Invalid role '{role_str}' at message index {idx}. "
                    f"Valid roles: {list(_ROLE_MAP.keys())}"
                )
            tool_calls = self._build_tool_calls(raw.get("tool_calls", []), idx)
            messages.append(
                Message(
                    role=_ROLE_MAP[role_str],
                    content=raw.get("content", ""),
                    tool_calls=tool_calls,
                    step_index=idx,
                    timestamp=raw.get("timestamp"),
                    metadata=raw.get("metadata", {}),
                )
            )
        return messages

    @staticmethod
    def _build_tool_calls(raw_calls: list[dict[str, Any]], msg_index: int) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for i, raw in enumerate(raw_calls):
            if "tool_name" not in raw:
                raise ValueError(
                    f"tool_calls[{i}] in message index {msg_index} is missing 'tool_name'"
                )
            calls.append(
                ToolCall(
                    tool_name=raw["tool_name"],
                    arguments=raw.get("arguments", {}),
                    result=raw.get("result"),
                    success=raw.get("success", True),
                    error_message=raw.get("error_message"),
                    timestamp=raw.get("timestamp"),
                )
            )
        return calls
