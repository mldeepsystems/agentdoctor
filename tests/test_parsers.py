"""Tests for BaseParser and JSONParser."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from agentdoctor.models import Role, ToolCall, Trace
from agentdoctor.parsers import BaseParser, JSONParser

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "traces")
HEALTHY_TRACE = os.path.join(FIXTURES_DIR, "healthy_trace.json")
THRASHING_TRACE = os.path.join(FIXTURES_DIR, "thrashing_trace.json")


# ---------- BaseParser ----------


class TestBaseParser:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseParser()

    def test_partial_subclass_cannot_instantiate(self):
        class Incomplete(BaseParser):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_load_file_valid(self):
        data = BaseParser._load_file(HEALTHY_TRACE)
        assert isinstance(data, dict)
        assert "messages" in data

    def test_load_file_missing(self):
        with pytest.raises(FileNotFoundError):
            BaseParser._load_file("/nonexistent/path.json")

    def test_load_file_invalid_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            f.flush()
            try:
                with pytest.raises(json.JSONDecodeError):
                    BaseParser._load_file(f.name)
            finally:
                os.unlink(f.name)

    def test_load_file_utf8_content(self):
        """Verify _load_file reads UTF-8 encoded content correctly."""
        data = {"messages": [], "note": "caf\u00e9 \u2014 \u00fc\u00f1\u00efc\u00f6d\u00e9"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            try:
                loaded = BaseParser._load_file(f.name)
                assert loaded["note"] == "caf\u00e9 \u2014 \u00fc\u00f1\u00efc\u00f6d\u00e9"
            finally:
                os.unlink(f.name)


# ---------- JSONParser: file input ----------


class TestJSONParserFromFile:
    def test_parse_healthy_trace(self):
        trace = JSONParser().parse(HEALTHY_TRACE)
        assert isinstance(trace, Trace)
        assert trace.trace_id == "healthy-001"
        assert trace.metadata == {"source": "test"}
        assert len(trace.messages) == 5

    def test_parse_thrashing_trace(self):
        trace = JSONParser().parse(THRASHING_TRACE)
        assert trace.trace_id == "thrashing-001"
        assert len(trace.messages) == 6

    def test_parse_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            JSONParser().parse("/nonexistent/trace.json")

    def test_parse_file_as_list(self):
        """File containing a bare JSON array of messages."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(messages, f)
            f.flush()
            try:
                trace = JSONParser().parse(f.name)
                assert len(trace.messages) == 2
                assert trace.trace_id is None
            finally:
                os.unlink(f.name)


# ---------- JSONParser: dict input ----------


class TestJSONParserFromDict:
    def test_parse_dict_with_messages(self):
        data = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
            "trace_id": "dict-001",
            "metadata": {"key": "value"},
        }
        trace = JSONParser().parse(data)
        assert trace.trace_id == "dict-001"
        assert trace.metadata == {"key": "value"}
        assert len(trace.messages) == 2

    def test_parse_dict_missing_messages(self):
        with pytest.raises(ValueError, match="messages"):
            JSONParser().parse({"trace_id": "no-messages"})

    def test_parse_dict_minimal(self):
        trace = JSONParser().parse({"messages": [{"role": "user", "content": "x"}]})
        assert trace.trace_id is None
        assert trace.metadata == {}


# ---------- JSONParser: list input ----------


class TestJSONParserFromList:
    def test_parse_bare_list(self):
        trace = JSONParser().parse(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        )
        assert len(trace.messages) == 2
        assert trace.trace_id is None

    def test_parse_empty_list(self):
        trace = JSONParser().parse([])
        assert len(trace.messages) == 0


# ---------- Role mapping ----------


class TestRoleMapping:
    @pytest.mark.parametrize(
        "role_str,expected",
        [
            ("system", Role.SYSTEM),
            ("user", Role.USER),
            ("assistant", Role.ASSISTANT),
            ("tool", Role.TOOL),
            ("function", Role.FUNCTION),
        ],
    )
    def test_valid_roles(self, role_str, expected):
        trace = JSONParser().parse([{"role": role_str, "content": "x"}])
        assert trace.messages[0].role is expected

    def test_invalid_role(self):
        with pytest.raises(ValueError, match="Invalid role"):
            JSONParser().parse([{"role": "wizard", "content": "x"}])

    def test_missing_role(self):
        with pytest.raises(ValueError, match="Invalid role"):
            JSONParser().parse([{"content": "no role"}])


# ---------- step_index ----------


class TestStepIndex:
    def test_sequential_assignment(self):
        trace = JSONParser().parse(
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ]
        )
        assert [m.step_index for m in trace.messages] == [0, 1, 2]


# ---------- Tool call parsing ----------


class TestToolCallParsing:
    def test_tool_calls_parsed(self):
        trace = JSONParser().parse(
            [
                {
                    "role": "assistant",
                    "content": "calling",
                    "tool_calls": [
                        {
                            "tool_name": "search",
                            "arguments": {"q": "test"},
                            "result": "found",
                            "success": True,
                        }
                    ],
                }
            ]
        )
        msg = trace.messages[0]
        assert len(msg.tool_calls) == 1
        tc = msg.tool_calls[0]
        assert isinstance(tc, ToolCall)
        assert tc.tool_name == "search"
        assert tc.arguments == {"q": "test"}
        assert tc.result == "found"
        assert tc.success is True

    def test_missing_tool_name(self):
        with pytest.raises(ValueError, match="missing 'tool_name'"):
            JSONParser().parse(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [{"arguments": {"q": "test"}}],
                    }
                ]
            )

    def test_tool_call_defaults(self):
        trace = JSONParser().parse(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"tool_name": "noop"}],
                }
            ]
        )
        tc = trace.messages[0].tool_calls[0]
        assert tc.arguments == {}
        assert tc.result is None
        assert tc.success is True
        assert tc.error_message is None
        assert tc.timestamp is None


# ---------- Optional field defaults ----------


class TestOptionalFieldDefaults:
    def test_content_defaults_to_empty(self):
        trace = JSONParser().parse([{"role": "user"}])
        assert trace.messages[0].content == ""

    def test_tool_calls_defaults_to_empty(self):
        trace = JSONParser().parse([{"role": "user", "content": "hi"}])
        assert trace.messages[0].tool_calls == []

    def test_timestamp_defaults_to_none(self):
        trace = JSONParser().parse([{"role": "user", "content": "hi"}])
        assert trace.messages[0].timestamp is None

    def test_metadata_defaults_to_empty_dict(self):
        trace = JSONParser().parse([{"role": "user", "content": "hi"}])
        assert trace.messages[0].metadata == {}


# ---------- Trace metadata preserved ----------


class TestTraceMetadata:
    def test_trace_id_preserved(self):
        trace = JSONParser().parse(HEALTHY_TRACE)
        assert trace.trace_id == "healthy-001"

    def test_metadata_preserved(self):
        trace = JSONParser().parse(HEALTHY_TRACE)
        assert trace.metadata == {"source": "test"}
