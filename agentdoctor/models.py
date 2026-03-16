"""Core data models for AgentDoctor traces and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agentdoctor.taxonomy import Pathology


class Role(str, Enum):
    """Role of a message participant in an agent trace."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Severity(str, Enum):
    """Severity level for a detected pathology."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ToolCall:
    """A single tool invocation within an agent trace."""

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    success: bool = True
    error_message: str | None = None
    timestamp: str | None = None


@dataclass
class Message:
    """A single message in an agent execution trace."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    step_index: int | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """A complete agent execution trace."""

    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None

    @property
    def system_prompt(self) -> str | None:
        """Return the content of the first system message, if any."""
        for msg in self.messages:
            if msg.role is Role.SYSTEM:
                return msg.content
        return None

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
    """The output of a single pathology detector."""

    pathology: Pathology
    detected: bool
    confidence: float = 0.0
    severity: Severity = Severity.LOW
    evidence: list[str] = field(default_factory=list)
    description: str = ""
    recommendation: str = ""
