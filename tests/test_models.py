import pytest

from agentdoctor.models import (
    DetectorResult,
    Message,
    Role,
    Severity,
    ToolCall,
    Trace,
)
from agentdoctor.taxonomy import Pathology


class TestEnums:
    def test_role_values(self):
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"

    def test_role_is_string(self):
        assert isinstance(Role.SYSTEM, str)

    def test_severity_values(self):
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"
        assert Severity.CRITICAL.value == "critical"

    def test_severity_is_string(self):
        assert isinstance(Severity.HIGH, str)

    def test_role_string_serialization(self):
        assert Role("system") is Role.SYSTEM

    def test_severity_string_serialization(self):
        assert Severity("critical") is Severity.CRITICAL


class TestToolCall:
    def test_creation_with_defaults(self):
        tc = ToolCall(tool_name="search")
        assert tc.tool_name == "search"
        assert tc.arguments == {}
        assert tc.result is None
        assert tc.success is True
        assert tc.error_message is None
        assert tc.timestamp is None

    def test_creation_with_all_fields(self):
        tc = ToolCall(
            tool_name="search",
            arguments={"query": "test"},
            result="found 3 results",
            success=True,
            error_message=None,
            timestamp="2026-01-01T00:00:00Z",
        )
        assert tc.arguments == {"query": "test"}
        assert tc.result == "found 3 results"


class TestMessage:
    def test_creation_with_defaults(self):
        msg = Message(role=Role.USER)
        assert msg.role is Role.USER
        assert msg.content == ""
        assert msg.tool_calls == []
        assert msg.step_index is None
        assert msg.timestamp is None
        assert msg.metadata == {}

    def test_creation_with_tool_calls(self):
        tc = ToolCall(tool_name="search")
        msg = Message(role=Role.ASSISTANT, content="Let me search", tool_calls=[tc])
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].tool_name == "search"


class TestTrace:
    def test_empty_trace(self):
        trace = Trace()
        assert trace.messages == []
        assert trace.metadata == {}
        assert trace.trace_id is None
        assert trace.system_prompt is None
        assert trace.tool_calls == []
        assert trace.messages_by_role() == {}

    def test_system_prompt(self):
        trace = Trace(
            messages=[
                Message(role=Role.SYSTEM, content="You are a helpful assistant."),
                Message(role=Role.USER, content="Hello"),
            ]
        )
        assert trace.system_prompt == "You are a helpful assistant."

    def test_system_prompt_absent(self):
        trace = Trace(messages=[Message(role=Role.USER, content="Hello")])
        assert trace.system_prompt is None

    def test_tool_calls_aggregation(self):
        tc1 = ToolCall(tool_name="search", arguments={"q": "a"})
        tc2 = ToolCall(tool_name="read", arguments={"path": "/tmp"})
        trace = Trace(
            messages=[
                Message(role=Role.ASSISTANT, content="step 1", tool_calls=[tc1]),
                Message(role=Role.ASSISTANT, content="step 2", tool_calls=[tc2]),
            ]
        )
        all_calls = trace.tool_calls
        assert len(all_calls) == 2
        assert all_calls[0].tool_name == "search"
        assert all_calls[1].tool_name == "read"

    def test_messages_by_role(self):
        trace = Trace(
            messages=[
                Message(role=Role.SYSTEM, content="sys"),
                Message(role=Role.USER, content="u1"),
                Message(role=Role.ASSISTANT, content="a1"),
                Message(role=Role.USER, content="u2"),
            ]
        )
        by_role = trace.messages_by_role()
        assert len(by_role[Role.SYSTEM]) == 1
        assert len(by_role[Role.USER]) == 2
        assert len(by_role[Role.ASSISTANT]) == 1
        assert Role.TOOL not in by_role


class TestDetectorResult:
    def test_creation_with_defaults(self):
        result = DetectorResult(
            pathology=Pathology.CONTEXT_EROSION,
            detected=True,
        )
        assert result.pathology is Pathology.CONTEXT_EROSION
        assert result.detected is True
        assert result.confidence == 0.0
        assert result.severity is Severity.LOW
        assert result.evidence == []
        assert result.description == ""
        assert result.recommendation == ""

    def test_creation_with_evidence(self):
        result = DetectorResult(
            pathology=Pathology.TOOL_THRASHING,
            detected=True,
            confidence=0.85,
            severity=Severity.HIGH,
            evidence=[
                "Tool 'search' called 5 times with identical arguments",
                "No progress made between calls",
            ],
            description="Repeated ineffective tool calls detected",
            recommendation="Add deduplication logic to tool invocations",
        )
        assert len(result.evidence) == 2
        assert result.confidence == 0.85
        assert result.severity is Severity.HIGH
