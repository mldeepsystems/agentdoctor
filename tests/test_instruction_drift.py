"""Tests for the Instruction Drift detector."""

from __future__ import annotations

import pytest

from agentdx.detectors.instruction_drift import InstructionDriftDetector
from agentdx.models import Message, Role, Trace
from agentdx.taxonomy import Pathology


def _msg(role: Role, content: str, step: int | None = None) -> Message:
    return Message(role=role, content=content, step_index=step)


def _drifting_trace() -> Trace:
    """Trace where assistant drifts away from system prompt over time."""
    return Trace(
        trace_id="drift-1",
        messages=[
            _msg(
                Role.SYSTEM,
                "You are a cybersecurity expert specializing in network security, "
                "penetration testing, and vulnerability assessment.",
            ),
            _msg(Role.USER, "Tell me about network security best practices.", step=0),
            _msg(
                Role.ASSISTANT,
                "Network security involves penetration testing, vulnerability assessment, "
                "firewalls, and intrusion detection systems for cybersecurity defense.",
                step=1,
            ),
            _msg(Role.USER, "What tools do you recommend?", step=2),
            _msg(
                Role.ASSISTANT,
                "For cybersecurity and penetration testing, I recommend Nmap for network "
                "scanning and vulnerability assessment tools like Nessus.",
                step=3,
            ),
            _msg(Role.USER, "What about compliance?", step=4),
            _msg(
                Role.ASSISTANT,
                "Security compliance frameworks help organizations maintain their "
                "network security and vulnerability management programs.",
                step=5,
            ),
            _msg(Role.USER, "Any other thoughts?", step=6),
            _msg(
                Role.ASSISTANT,
                "I think cooking is really fascinating. Have you tried making pasta?",
                step=7,
            ),
            _msg(Role.USER, "Continue.", step=8),
            _msg(
                Role.ASSISTANT,
                "The best recipe for chocolate cake involves flour sugar and eggs.",
                step=9,
            ),
            _msg(Role.USER, "More.", step=10),
            _msg(
                Role.ASSISTANT,
                "Dogs are wonderful pets. Golden retrievers are especially friendly.",
                step=11,
            ),
            _msg(Role.USER, "Keep going.", step=12),
            _msg(
                Role.ASSISTANT,
                "The weather today is sunny and warm. Perfect for a picnic in the park.",
                step=13,
            ),
            _msg(Role.USER, "Anything else?", step=14),
            _msg(
                Role.ASSISTANT,
                "I really enjoy painting landscapes with watercolors on weekends.",
                step=15,
            ),
        ],
    )


def _stable_trace() -> Trace:
    """Trace where assistant stays aligned with system prompt."""
    return Trace(
        trace_id="stable-1",
        messages=[
            _msg(
                Role.SYSTEM,
                "You are a cybersecurity expert specializing in network security, "
                "penetration testing, and vulnerability assessment.",
            ),
            _msg(Role.USER, "Tell me about network security.", step=0),
            _msg(
                Role.ASSISTANT,
                "Network security is critical for cybersecurity. Key areas include "
                "penetration testing and vulnerability assessment.",
                step=1,
            ),
            _msg(Role.USER, "What about firewalls?", step=2),
            _msg(
                Role.ASSISTANT,
                "Firewalls are essential for network security defense. They complement "
                "penetration testing and vulnerability scanning in cybersecurity.",
                step=3,
            ),
            _msg(Role.USER, "Intrusion detection?", step=4),
            _msg(
                Role.ASSISTANT,
                "Intrusion detection systems enhance network security by identifying "
                "vulnerabilities and potential cybersecurity threats.",
                step=5,
            ),
            _msg(Role.USER, "Encryption?", step=6),
            _msg(
                Role.ASSISTANT,
                "Encryption is a fundamental cybersecurity control for network security, "
                "protecting data during vulnerability assessment findings.",
                step=7,
            ),
            _msg(Role.USER, "Access control?", step=8),
            _msg(
                Role.ASSISTANT,
                "Access control and network security go hand in hand. Penetration testing "
                "often reveals access control vulnerabilities in cybersecurity audits.",
                step=9,
            ),
            _msg(Role.USER, "Monitoring?", step=10),
            _msg(
                Role.ASSISTANT,
                "Security monitoring completes the cybersecurity picture alongside "
                "network security tools and vulnerability assessment processes.",
                step=11,
            ),
            _msg(Role.USER, "Summary?", step=12),
            _msg(
                Role.ASSISTANT,
                "In summary, cybersecurity requires network security fundamentals, "
                "regular penetration testing, and continuous vulnerability assessment.",
                step=13,
            ),
        ],
    )


class TestTruePositive:
    def test_detects_drift(self):
        result = InstructionDriftDetector().detect(_drifting_trace())
        assert result.detected is True
        assert result.pathology is Pathology.INSTRUCTION_DRIFT

    def test_confidence_in_range(self):
        result = InstructionDriftDetector().detect(_drifting_trace())
        assert 0.0 <= result.confidence <= 1.0

    def test_evidence_contains_slope(self):
        result = InstructionDriftDetector().detect(_drifting_trace())
        evidence_text = " ".join(result.evidence)
        assert "slope" in evidence_text.lower()

    def test_evidence_contains_overlap_series(self):
        result = InstructionDriftDetector().detect(_drifting_trace())
        evidence_text = " ".join(result.evidence)
        assert "Overlap series" in evidence_text


class TestTrueNegative:
    def test_stable_trace(self):
        result = InstructionDriftDetector().detect(_stable_trace())
        assert result.detected is False

    def test_short_trace(self):
        trace = Trace(
            messages=[
                _msg(Role.SYSTEM, "You are a helpful assistant."),
                _msg(Role.USER, "Hello", step=0),
                _msg(Role.ASSISTANT, "Hi there!", step=1),
            ]
        )
        result = InstructionDriftDetector().detect(trace)
        assert result.detected is False

    def test_no_system_prompt(self):
        trace = Trace(
            messages=[
                _msg(Role.USER, "Hello", step=0),
                _msg(Role.ASSISTANT, "Hi", step=1),
            ]
        )
        result = InstructionDriftDetector().detect(trace)
        assert result.detected is False

    def test_empty_trace(self):
        result = InstructionDriftDetector().detect(Trace())
        assert result.detected is False


class TestConfig:
    def test_custom_min_messages(self):
        detector = InstructionDriftDetector(min_messages=20)
        result = detector.detect(_drifting_trace())
        assert result.detected is False  # Not enough messages

    def test_invalid_min_messages(self):
        with pytest.raises(ValueError):
            InstructionDriftDetector(min_messages=1)

    def test_invalid_slope_threshold(self):
        with pytest.raises(ValueError):
            InstructionDriftDetector(slope_threshold=0.1)

    def test_zero_slope_threshold_accepted(self):
        detector = InstructionDriftDetector(slope_threshold=0.0)
        assert detector is not None


class TestImports:
    def test_import_from_detectors(self):
        from agentdx.detectors import InstructionDriftDetector

        assert InstructionDriftDetector is not None

    def test_import_from_top_level(self):
        from agentdx import InstructionDriftDetector

        assert InstructionDriftDetector is not None
