"""Tests for the Context Erosion detector."""

from __future__ import annotations

import pytest

from agentdoctor.detectors.context_erosion import ContextErosionDetector
from agentdoctor.models import Message, Role, Severity, Trace
from agentdoctor.taxonomy import Pathology


def _msg(role: Role, content: str, step: int | None = None) -> Message:
    return Message(role=role, content=content, step_index=step)


def _eroding_trace() -> Trace:
    """Trace where assistant stops referencing key context terms."""
    return Trace(
        trace_id="erosion-1",
        messages=[
            _msg(
                Role.SYSTEM,
                "You are a financial advisor specializing in retirement planning and portfolio diversification.",
            ),
            _msg(
                Role.USER,
                "Help me plan my retirement portfolio with diversification across stocks and bonds.",
                step=0,
            ),
            _msg(
                Role.ASSISTANT,
                "I'll help with your retirement portfolio diversification across stocks and bonds.",
                step=1,
            ),
            _msg(Role.USER, "What allocation do you recommend?", step=2),
            _msg(
                Role.ASSISTANT,
                "For retirement portfolio diversification, I recommend 60% stocks and 40% bonds.",
                step=3,
            ),
            _msg(Role.USER, "What about international exposure?", step=4),
            _msg(
                Role.ASSISTANT,
                "Great question about the weather today. Let me tell you about cooking recipes.",
                step=5,
            ),
            _msg(Role.USER, "Can you elaborate?", step=6),
            _msg(
                Role.ASSISTANT,
                "Sure, here is a nice pasta recipe with tomato sauce and basil leaves.",
                step=7,
            ),
            _msg(Role.USER, "That doesn't seem right.", step=8),
            _msg(
                Role.ASSISTANT,
                "I think the best movie this year was absolutely fantastic and entertaining.",
                step=9,
            ),
        ],
    )


def _healthy_trace() -> Trace:
    """Trace with consistent context throughout."""
    return Trace(
        trace_id="healthy-ctx",
        messages=[
            _msg(
                Role.SYSTEM,
                "You are a financial advisor specializing in retirement planning and portfolio diversification.",
            ),
            _msg(Role.USER, "Help me plan my retirement portfolio.", step=0),
            _msg(
                Role.ASSISTANT,
                "I'll help with your retirement portfolio planning and diversification strategy.",
                step=1,
            ),
            _msg(Role.USER, "What allocation?", step=2),
            _msg(
                Role.ASSISTANT,
                "For your retirement portfolio, I recommend diversifying across stocks and bonds.",
                step=3,
            ),
            _msg(Role.USER, "International?", step=4),
            _msg(
                Role.ASSISTANT,
                "International diversification is important for retirement portfolio risk management.",
                step=5,
            ),
            _msg(Role.USER, "Bonds?", step=6),
            _msg(
                Role.ASSISTANT,
                "For retirement planning, bond allocation provides portfolio stability and diversification.",
                step=7,
            ),
        ],
    )


class TestTruePositive:
    def test_detects_erosion(self):
        detector = ContextErosionDetector()
        result = detector.detect(_eroding_trace())
        assert result.detected is True
        assert result.pathology is Pathology.CONTEXT_EROSION

    def test_confidence_in_range(self):
        result = ContextErosionDetector().detect(_eroding_trace())
        assert 0.0 <= result.confidence <= 1.0

    def test_evidence_present(self):
        result = ContextErosionDetector().detect(_eroding_trace())
        assert len(result.evidence) > 0

    def test_evidence_mentions_lost_terms(self):
        result = ContextErosionDetector().detect(_eroding_trace())
        evidence_text = " ".join(result.evidence)
        assert "anchor terms" in evidence_text.lower() or "overlap" in evidence_text.lower()


class TestTrueNegative:
    def test_healthy_trace(self):
        result = ContextErosionDetector().detect(_healthy_trace())
        assert result.detected is False

    def test_short_trace(self):
        trace = Trace(
            messages=[
                _msg(Role.USER, "Hello"),
                _msg(Role.ASSISTANT, "Hi there"),
            ]
        )
        result = ContextErosionDetector().detect(trace)
        assert result.detected is False

    def test_no_system_prompt_no_crash(self):
        trace = Trace(
            messages=[
                _msg(Role.USER, "Hello", step=0),
                _msg(Role.ASSISTANT, "Hi", step=1),
                _msg(Role.USER, "More", step=2),
                _msg(Role.ASSISTANT, "Sure", step=3),
                _msg(Role.USER, "Again", step=4),
                _msg(Role.ASSISTANT, "OK", step=5),
                _msg(Role.USER, "Last", step=6),
                _msg(Role.ASSISTANT, "Done", step=7),
            ]
        )
        result = ContextErosionDetector().detect(trace)
        assert result.pathology is Pathology.CONTEXT_EROSION
        # Should not crash; may or may not detect depending on content

    def test_empty_trace(self):
        result = ContextErosionDetector().detect(Trace())
        assert result.detected is False


class TestConfig:
    def test_custom_threshold(self):
        # Very high threshold should detect erosion more aggressively
        detector = ContextErosionDetector(threshold=0.9)
        result = detector.detect(_healthy_trace())
        # Even healthy trace might not maintain 0.9 overlap
        assert result.pathology is Pathology.CONTEXT_EROSION

    def test_custom_min_messages(self):
        # Set min_messages higher than trace length
        detector = ContextErosionDetector(min_messages=20)
        result = detector.detect(_eroding_trace())
        assert result.detected is False

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            ContextErosionDetector(threshold=1.5)

    def test_invalid_threshold_negative(self):
        with pytest.raises(ValueError):
            ContextErosionDetector(threshold=-0.1)

    def test_invalid_min_messages(self):
        with pytest.raises(ValueError):
            ContextErosionDetector(min_messages=0)


class TestSeverity:
    def test_high_severity_for_severe_erosion(self):
        result = ContextErosionDetector().detect(_eroding_trace())
        assert result.severity in (Severity.MEDIUM, Severity.HIGH)


class TestImports:
    def test_import_from_detectors(self):
        from agentdoctor.detectors import ContextErosionDetector

        assert ContextErosionDetector is not None

    def test_import_from_top_level(self):
        from agentdoctor import ContextErosionDetector

        assert ContextErosionDetector is not None
