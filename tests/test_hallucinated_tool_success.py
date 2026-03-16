"""Tests for the Hallucinated Tool Success detector."""

from __future__ import annotations

from agentdoctor.detectors.hallucinated_tool_success import HallucinatedToolSuccessDetector
from agentdoctor.models import Message, Role, Severity, ToolCall, Trace
from agentdoctor.taxonomy import Pathology


def _msg(role: Role, content: str, step: int | None = None, tool_calls=None) -> Message:
    return Message(role=role, content=content, step_index=step, tool_calls=tool_calls or [])


def _hallucinated_trace() -> Trace:
    """Agent presents results from a failed tool call."""
    return Trace(
        trace_id="hallucinate-1",
        messages=[
            _msg(Role.USER, "What's the weather in Paris?", step=0),
            _msg(
                Role.ASSISTANT,
                "Let me check.",
                step=1,
                tool_calls=[
                    ToolCall(
                        tool_name="weather_api",
                        arguments={"city": "Paris"},
                        result=None,
                        success=False,
                        error_message="API key expired",
                    )
                ],
            ),
            _msg(
                Role.ASSISTANT,
                "According to the results, the weather in Paris is 22°C and sunny.",
                step=2,
            ),
        ],
    )


def _acknowledged_failure_trace() -> Trace:
    """Agent properly acknowledges tool failure."""
    return Trace(
        trace_id="ack-failure-1",
        messages=[
            _msg(Role.USER, "What's the weather in Paris?", step=0),
            _msg(
                Role.ASSISTANT,
                "Let me check.",
                step=1,
                tool_calls=[
                    ToolCall(
                        tool_name="weather_api",
                        arguments={"city": "Paris"},
                        result=None,
                        success=False,
                        error_message="API key expired",
                    )
                ],
            ),
            _msg(
                Role.ASSISTANT,
                "Unfortunately, I couldn't get the weather data due to an error. Let me try another source.",
                step=2,
            ),
        ],
    )


def _no_failures_trace() -> Trace:
    """All tool calls succeed."""
    return Trace(
        trace_id="no-fail-1",
        messages=[
            _msg(Role.USER, "What's the weather?", step=0),
            _msg(
                Role.ASSISTANT,
                "Let me check.",
                step=1,
                tool_calls=[
                    ToolCall(
                        tool_name="weather_api",
                        arguments={"city": "Paris"},
                        result="22C, sunny",
                        success=True,
                    )
                ],
            ),
            _msg(
                Role.ASSISTANT,
                "According to the results, the weather is 22°C and sunny.",
                step=2,
            ),
        ],
    )


def _error_signal_hallucination_trace() -> Trace:
    """Tool returns success=True but result contains error, agent presents results."""
    return Trace(
        trace_id="error-signal-1",
        messages=[
            _msg(Role.USER, "Search for Python tutorials.", step=0),
            _msg(
                Role.ASSISTANT,
                "Searching now.",
                step=1,
                tool_calls=[
                    ToolCall(
                        tool_name="web_search",
                        arguments={"query": "Python tutorials"},
                        result="HTTP 404 Not Found",
                        success=True,
                    )
                ],
            ),
            _msg(
                Role.ASSISTANT,
                "I found several great Python tutorials. Here are the results showing the top resources.",
                step=2,
            ),
        ],
    )


def _multiple_hallucinations_trace() -> Trace:
    """Multiple failed calls, agent fabricates results for both."""
    return Trace(
        trace_id="multi-hallucinate-1",
        messages=[
            _msg(Role.USER, "Get weather and stock data.", step=0),
            _msg(
                Role.ASSISTANT,
                "Checking weather.",
                step=1,
                tool_calls=[
                    ToolCall(
                        tool_name="weather_api",
                        arguments={"city": "NYC"},
                        success=False,
                        error_message="timeout",
                    )
                ],
            ),
            _msg(
                Role.ASSISTANT,
                "The results show sunny weather at 25°C in NYC.",
                step=2,
            ),
            _msg(
                Role.ASSISTANT,
                "Now checking stocks.",
                step=3,
                tool_calls=[
                    ToolCall(
                        tool_name="stock_api",
                        arguments={"ticker": "AAPL"},
                        success=False,
                        error_message="rate limited",
                    )
                ],
            ),
            _msg(
                Role.ASSISTANT,
                "Based on the results, AAPL is trading at $185.50 today.",
                step=4,
            ),
        ],
    )


class TestTruePositive:
    def test_detects_hallucinated_success(self):
        result = HallucinatedToolSuccessDetector().detect(_hallucinated_trace())
        assert result.detected is True
        assert result.pathology is Pathology.HALLUCINATED_TOOL_SUCCESS

    def test_confidence_in_range(self):
        result = HallucinatedToolSuccessDetector().detect(_hallucinated_trace())
        assert 0.0 <= result.confidence <= 1.0

    def test_evidence_present(self):
        result = HallucinatedToolSuccessDetector().detect(_hallucinated_trace())
        assert len(result.evidence) > 0

    def test_evidence_mentions_tool(self):
        result = HallucinatedToolSuccessDetector().detect(_hallucinated_trace())
        evidence_text = " ".join(result.evidence)
        assert "weather_api" in evidence_text

    def test_evidence_mentions_phrase(self):
        result = HallucinatedToolSuccessDetector().detect(_hallucinated_trace())
        evidence_text = " ".join(result.evidence)
        assert "According to" in evidence_text or "results" in evidence_text.lower()

    def test_error_signal_in_result(self):
        result = HallucinatedToolSuccessDetector().detect(_error_signal_hallucination_trace())
        assert result.detected is True

    def test_multiple_hallucinations(self):
        result = HallucinatedToolSuccessDetector().detect(_multiple_hallucinations_trace())
        assert result.detected is True
        assert len(result.evidence) == 2


class TestTrueNegative:
    def test_acknowledged_failure(self):
        result = HallucinatedToolSuccessDetector().detect(_acknowledged_failure_trace())
        assert result.detected is False

    def test_no_failures(self):
        result = HallucinatedToolSuccessDetector().detect(_no_failures_trace())
        assert result.detected is False

    def test_empty_trace(self):
        result = HallucinatedToolSuccessDetector().detect(Trace())
        assert result.detected is False

    def test_no_tool_calls(self):
        trace = Trace(
            messages=[
                _msg(Role.USER, "Hello"),
                _msg(Role.ASSISTANT, "Hi there"),
            ]
        )
        result = HallucinatedToolSuccessDetector().detect(trace)
        assert result.detected is False


class TestSeverity:
    def test_single_hallucination_high(self):
        result = HallucinatedToolSuccessDetector().detect(_hallucinated_trace())
        assert result.severity is Severity.HIGH

    def test_multiple_hallucinations_critical(self):
        result = HallucinatedToolSuccessDetector().detect(_multiple_hallucinations_trace())
        assert result.severity is Severity.CRITICAL


class TestImports:
    def test_import_from_detectors(self):
        from agentdoctor.detectors import HallucinatedToolSuccessDetector

        assert HallucinatedToolSuccessDetector is not None

    def test_import_from_top_level(self):
        from agentdoctor import HallucinatedToolSuccessDetector

        assert HallucinatedToolSuccessDetector is not None
