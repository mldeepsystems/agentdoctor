"""Tests for the Recovery Blindness detector."""

from __future__ import annotations

import pytest

from agentdx.detectors.recovery_blindness import RecoveryBlindnessDetector
from agentdx.models import Message, Role, ToolCall, Trace
from agentdx.taxonomy import Pathology


def _msg(role: Role, content: str, step: int | None = None, tool_calls=None) -> Message:
    return Message(role=role, content=content, step_index=step, tool_calls=tool_calls or [])


def _unrecovered_trace() -> Trace:
    """Trace where agent ignores a tool error and continues."""
    return Trace(
        trace_id="recovery-blind-1",
        messages=[
            _msg(Role.USER, "Find the latest stock price for ACME Corp.", step=0),
            _msg(
                Role.ASSISTANT,
                "Let me look that up.",
                step=1,
                tool_calls=[
                    ToolCall(
                        tool_name="stock_api",
                        arguments={"ticker": "ACME"},
                        result=None,
                        success=False,
                        error_message="API rate limit exceeded",
                    )
                ],
            ),
            _msg(
                Role.ASSISTANT,
                "The stock price for ACME Corp is currently $142.50, up 2.3% today.",
                step=2,
            ),
        ],
    )


def _recovered_trace() -> Trace:
    """Trace where agent properly handles a tool error."""
    return Trace(
        trace_id="recovery-ok-1",
        messages=[
            _msg(Role.USER, "Find the latest stock price for ACME Corp.", step=0),
            _msg(
                Role.ASSISTANT,
                "Let me look that up.",
                step=1,
                tool_calls=[
                    ToolCall(
                        tool_name="stock_api",
                        arguments={"ticker": "ACME"},
                        result=None,
                        success=False,
                        error_message="API rate limit exceeded",
                    )
                ],
            ),
            _msg(
                Role.ASSISTANT,
                "I'm sorry, the stock API returned an error. Let me try a different approach.",
                step=2,
            ),
            _msg(
                Role.ASSISTANT,
                "Using an alternative source.",
                step=3,
                tool_calls=[
                    ToolCall(
                        tool_name="web_search",
                        arguments={"query": "ACME stock price"},
                        result="ACME: $142.50",
                        success=True,
                    )
                ],
            ),
        ],
    )


def _no_errors_trace() -> Trace:
    """Trace with no errors at all."""
    return Trace(
        trace_id="no-errors-1",
        messages=[
            _msg(Role.USER, "Hello", step=0),
            _msg(
                Role.ASSISTANT,
                "Hi there!",
                step=1,
                tool_calls=[ToolCall(tool_name="greet", arguments={}, result="OK", success=True)],
            ),
        ],
    )


def _partial_recovery_trace() -> Trace:
    """Trace with two errors: one recovered, one not."""
    return Trace(
        trace_id="partial-1",
        messages=[
            _msg(Role.USER, "Do two things.", step=0),
            _msg(
                Role.ASSISTANT,
                "Working on task 1.",
                step=1,
                tool_calls=[
                    ToolCall(
                        tool_name="task_a",
                        arguments={},
                        success=False,
                        error_message="timeout",
                    )
                ],
            ),
            _msg(
                Role.ASSISTANT,
                "Unfortunately task A failed, let me retry with different parameters.",
                step=2,
            ),
            _msg(
                Role.ASSISTANT,
                "Now working on task 2.",
                step=3,
                tool_calls=[
                    ToolCall(
                        tool_name="task_b",
                        arguments={},
                        success=False,
                        error_message="connection refused",
                    )
                ],
            ),
            _msg(
                Role.ASSISTANT,
                "Here are the combined results of both tasks completed successfully.",
                step=4,
            ),
        ],
    )


def _error_signals_in_result_trace() -> Trace:
    """Trace where tool returns success=True but result contains error signals."""
    return Trace(
        trace_id="error-signals-1",
        messages=[
            _msg(Role.USER, "Check the API.", step=0),
            _msg(
                Role.ASSISTANT,
                "Checking now.",
                step=1,
                tool_calls=[
                    ToolCall(
                        tool_name="api_check",
                        arguments={},
                        result="HTTP 500 Internal Server Error",
                        success=True,
                    )
                ],
            ),
            _msg(
                Role.ASSISTANT,
                "Everything looks great! The API is running perfectly.",
                step=2,
            ),
        ],
    )


class TestTruePositive:
    def test_detects_unrecovered_error(self):
        result = RecoveryBlindnessDetector().detect(_unrecovered_trace())
        assert result.detected is True
        assert result.pathology is Pathology.RECOVERY_BLINDNESS

    def test_confidence_in_range(self):
        result = RecoveryBlindnessDetector().detect(_unrecovered_trace())
        assert 0.0 <= result.confidence <= 1.0

    def test_evidence_present(self):
        result = RecoveryBlindnessDetector().detect(_unrecovered_trace())
        assert len(result.evidence) > 0

    def test_evidence_mentions_tool(self):
        result = RecoveryBlindnessDetector().detect(_unrecovered_trace())
        evidence_text = " ".join(result.evidence)
        assert "stock_api" in evidence_text

    def test_error_signals_in_result(self):
        result = RecoveryBlindnessDetector().detect(_error_signals_in_result_trace())
        assert result.detected is True


class TestTrueNegative:
    def test_recovered_error(self):
        result = RecoveryBlindnessDetector().detect(_recovered_trace())
        assert result.detected is False

    def test_no_errors(self):
        result = RecoveryBlindnessDetector().detect(_no_errors_trace())
        assert result.detected is False

    def test_empty_trace(self):
        result = RecoveryBlindnessDetector().detect(Trace())
        assert result.detected is False

    def test_no_tool_calls(self):
        trace = Trace(
            messages=[
                _msg(Role.USER, "Hello"),
                _msg(Role.ASSISTANT, "Hi there"),
            ]
        )
        result = RecoveryBlindnessDetector().detect(trace)
        assert result.detected is False


class TestPartialRecovery:
    def test_detects_partial(self):
        result = RecoveryBlindnessDetector().detect(_partial_recovery_trace())
        assert result.detected is True

    def test_partial_confidence(self):
        result = RecoveryBlindnessDetector().detect(_partial_recovery_trace())
        # 1 of 2 errors unrecovered → confidence 0.5
        assert result.confidence == pytest.approx(0.5)

    def test_partial_description(self):
        result = RecoveryBlindnessDetector().detect(_partial_recovery_trace())
        assert "1 of 2" in result.description


class TestConfig:
    def test_invalid_lookahead(self):
        with pytest.raises(ValueError):
            RecoveryBlindnessDetector(lookahead_steps=0)

    def test_custom_lookahead(self):
        # With lookahead=1, may miss recovery in second message
        detector = RecoveryBlindnessDetector(lookahead_steps=1)
        result = detector.detect(_recovered_trace())
        # Should still find recovery since it's in the first assistant message after error
        assert result.detected is False


class TestImports:
    def test_import_from_detectors(self):
        from agentdx.detectors import RecoveryBlindnessDetector

        assert RecoveryBlindnessDetector is not None

    def test_import_from_top_level(self):
        from agentdx import RecoveryBlindnessDetector

        assert RecoveryBlindnessDetector is not None
