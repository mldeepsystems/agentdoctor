"""Core data models for AgentDoctor traces and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agentdoctor.taxonomy import Pathology


class Role(str, Enum):
    """Role of a message participant in an agent trace.

    FUNCTION is included for compatibility with OpenAI's API, which uses
    "function" alongside "tool" for legacy function-calling traces.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"


class Severity(str, Enum):
    """Severity level for a detected pathology."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ToolCall:
    """A single tool invocation within an agent trace.

    Fields:
        tool_name: Identifier of the tool that was called.
        arguments: Keyword arguments passed to the tool.
        result: String representation of the tool's output. ``None`` means
            no result was captured; ``""`` means the tool returned empty output.
        success: Whether the tool call succeeded. Defaults to ``True``.
        error_message: Error description when the call failed. May be set
            alongside ``success=True`` for partial failures.
        timestamp: ISO-8601 timestamp string, or ``None`` if unavailable.
    """

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    success: bool = True
    error_message: str | None = None
    timestamp: str | None = None


@dataclass
class Message:
    """A single message in an agent execution trace.

    Fields:
        role: The participant role (system, user, assistant, tool, function).
        content: Text body of the message. May be empty for tool-only turns.
        tool_calls: Tool invocations associated with this message.
        step_index: Monotonic index for ordering within the trace.
        timestamp: ISO-8601 timestamp string, or ``None`` if unavailable.
        metadata: Arbitrary key-value pairs from the source framework.
    """

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    step_index: int | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """A complete agent execution trace.

    Traces are mutable so that parsers can build them incrementally via
    ``trace.messages.append(msg)``.
    """

    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None

    @property
    def system_prompt(self) -> str | None:
        """Return the content of the first system message, if any.

        For traces with multiple system messages, use :pyattr:`system_prompts`.
        """
        for msg in self.messages:
            if msg.role is Role.SYSTEM:
                return msg.content
        return None

    @property
    def system_prompts(self) -> list[str]:
        """Return the content of all system messages, in order."""
        return [msg.content for msg in self.messages if msg.role is Role.SYSTEM]

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Collect all tool calls across every message."""
        calls: list[ToolCall] = []
        for msg in self.messages:
            calls.extend(msg.tool_calls)
        return calls

    def messages_by_role(self) -> dict[Role, list[Message]]:
        """Group messages by their role."""
        by_role: dict[Role, list[Message]] = {}
        for msg in self.messages:
            by_role.setdefault(msg.role, []).append(msg)
        return by_role


@dataclass
class DetectorResult:
    """The output of a single pathology detector.

    Fields:
        pathology: Which pathology this result is about.
        detected: Whether the pathology was found.
        confidence: Confidence score in ``[0.0, 1.0]``. Represents certainty
            in the *detected* verdict — ``detected=False, confidence=0.9``
            means "90 % sure this pathology is absent."
        severity: Severity when detected. Defaults to LOW.
        evidence: Human-readable evidence strings supporting the verdict.
            May be non-empty even when ``detected=False`` (investigated but
            not confirmed).
        description: Short summary of the finding.
        recommendation: Suggested remediation.
    """

    pathology: Pathology
    detected: bool
    confidence: float = 0.0
    severity: Severity = Severity.LOW
    evidence: list[str] = field(default_factory=list)
    description: str = ""
    recommendation: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
