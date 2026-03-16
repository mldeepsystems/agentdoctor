"""Tests for the ToolThrashingDetector."""

from __future__ import annotations

import os

import pytest

from agentdoctor.detectors import BaseDetector
from agentdoctor.detectors.tool_thrashing import ToolThrashingDetector
from agentdoctor.models import Message, Role, Severity, ToolCall, Trace
from agentdoctor.parsers import JSONParser
from agentdoctor.taxonomy import Pathology

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "traces")
HEALTHY_TRACE = os.path.join(FIXTURES_DIR, "healthy_trace.json")
THRASHING_TRACE = os.path.join(FIXTURES_DIR, "thrashing_trace.json")


def _make_trace_with_tool_calls(
    tool_name: str, args_list: list[dict], role: Role = Role.ASSISTANT
) -> Trace:
    """Helper to build a Trace with one tool call per message."""
    messages = []
    for i, args in enumerate(args_list):
        messages.append(
            Message(
                role=role,
                content="",
                tool_calls=[ToolCall(tool_name=tool_name, arguments=args)],
                step_index=i,
            )
        )
    return Trace(messages=messages)


# ---------- Fixture-based tests ----------


class TestFixtureTraces:
    def test_thrashing_trace_detected(self):
        trace = JSONParser().parse(THRASHING_TRACE)
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is True

    def test_healthy_trace_not_detected(self):
        trace = JSONParser().parse(HEALTHY_TRACE)
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is False


# ---------- Programmatic tests ----------


class TestProgrammatic:
    def test_three_identical_calls_detected(self):
        trace = _make_trace_with_tool_calls(
            "search", [{"q": "python"}] * 3
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is True
        assert result.confidence == pytest.approx(0.6)
        assert result.severity is Severity.MEDIUM

    def test_five_identical_calls_critical(self):
        trace = _make_trace_with_tool_calls(
            "search", [{"q": "python"}] * 5
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is True
        assert result.confidence == pytest.approx(1.0)
        assert result.severity is Severity.CRITICAL

    def test_two_calls_not_detected(self):
        trace = _make_trace_with_tool_calls(
            "search", [{"q": "python"}] * 2
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is False

    def test_different_args_not_detected(self):
        trace = _make_trace_with_tool_calls(
            "search",
            [
                {"q": "alpha"},
                {"q": "bravo"},
                {"q": "charlie"},
            ],
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is False

    def test_no_tool_calls_not_detected(self):
        trace = Trace(
            messages=[
                Message(role=Role.USER, content="hi", step_index=0),
                Message(role=Role.ASSISTANT, content="hello", step_index=1),
            ]
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is False

    def test_four_calls_high_severity(self):
        trace = _make_trace_with_tool_calls(
            "search", [{"q": "python"}] * 4
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is True
        assert result.severity is Severity.HIGH

    def test_empty_args_treated_as_identical(self):
        trace = _make_trace_with_tool_calls("ping", [{}] * 3)
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is True


# ---------- Config tests ----------


class TestConfig:
    def test_custom_min_repeats(self):
        trace = _make_trace_with_tool_calls(
            "search", [{"q": "python"}] * 2
        )
        result = ToolThrashingDetector(min_repeats=2).detect(trace)
        assert result.detected is True

    def test_custom_threshold_strict(self):
        """With threshold=1.0, only exact matches count."""
        trace = _make_trace_with_tool_calls(
            "search",
            [
                {"q": "python testing"},
                {"q": "python testing framework"},
                {"q": "python testing best"},
            ],
        )
        result = ToolThrashingDetector(similarity_threshold=1.0).detect(trace)
        assert result.detected is False

    def test_custom_window(self):
        """Window of 3 should still catch 3 adjacent identical calls."""
        trace = _make_trace_with_tool_calls(
            "search", [{"q": "python"}] * 3
        )
        result = ToolThrashingDetector(window_size=3).detect(trace)
        assert result.detected is True


# ---------- Evidence ----------


class TestEvidence:
    def test_evidence_contains_tool_name(self):
        trace = _make_trace_with_tool_calls(
            "web_search", [{"q": "python"}] * 3
        )
        result = ToolThrashingDetector().detect(trace)
        assert any("web_search" in e for e in result.evidence)

    def test_evidence_contains_step_indices(self):
        trace = _make_trace_with_tool_calls(
            "search", [{"q": "python"}] * 3
        )
        result = ToolThrashingDetector().detect(trace)
        assert any("Step indices" in e for e in result.evidence)


# ---------- Type checks ----------


class TestTypeChecks:
    def test_pathology_property(self):
        detector = ToolThrashingDetector()
        assert detector.pathology is Pathology.TOOL_THRASHING

    def test_isinstance_base_detector(self):
        detector = ToolThrashingDetector()
        assert isinstance(detector, BaseDetector)
