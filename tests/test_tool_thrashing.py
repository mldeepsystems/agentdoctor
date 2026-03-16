"""Tests for the ToolThrashingDetector."""

from __future__ import annotations

import os

import pytest

from agentdoctor.detectors import BaseDetector
from agentdoctor.detectors.tool_thrashing import (
    ToolThrashingDetector,
    _UNKNOWN_STEP,
    _serialize_args,
)
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


# ---------- Numeric & mixed-type argument tests (false-positive regression) ----------


class TestNumericArgs:
    """Verify that numeric argument values are NOT silently dropped.

    Success: calls with identical tool name but different numeric args are
    recognised as distinct and NOT flagged as thrashing.
    Failure: numbers are stripped/ignored, causing false positives on
    pagination, retry-with-backoff, or ID-based lookups.
    """

    def test_pagination_not_flagged(self):
        """fetch_page(page=1), fetch_page(page=2), fetch_page(page=3) differ."""
        trace = _make_trace_with_tool_calls(
            "fetch_page",
            [{"page": 1}, {"page": 2}, {"page": 3}],
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is False

    def test_identical_numeric_args_detected(self):
        """fetch_page(page=1) repeated 3x is genuine thrashing."""
        trace = _make_trace_with_tool_calls(
            "fetch_page", [{"page": 1}] * 3
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is True

    def test_id_lookup_not_flagged(self):
        """get_user(id=101), get_user(id=102), get_user(id=103) differ."""
        trace = _make_trace_with_tool_calls(
            "get_user",
            [{"id": 101}, {"id": 102}, {"id": 103}],
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is False

    def test_mixed_string_and_int_differ_by_int(self):
        """Same query string but different limit → distinct calls."""
        trace = _make_trace_with_tool_calls(
            "search",
            [
                {"q": "python", "limit": 10},
                {"q": "python", "limit": 20},
                {"q": "python", "limit": 30},
            ],
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is False

    def test_mixed_string_and_int_same_values(self):
        """Same query AND same limit → genuine thrashing."""
        trace = _make_trace_with_tool_calls(
            "search",
            [{"q": "python", "limit": 10}] * 3,
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is True

    def test_boolean_args_differ(self):
        """enabled=True vs enabled=False are distinct."""
        trace = _make_trace_with_tool_calls(
            "toggle",
            [{"enabled": True}, {"enabled": False}, {"enabled": True}],
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is False

    def test_float_args_differ(self):
        """temperature=0.1 vs 0.5 vs 0.9 are distinct."""
        trace = _make_trace_with_tool_calls(
            "sample",
            [{"temperature": 0.1}, {"temperature": 0.5}, {"temperature": 0.9}],
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is False

    def test_none_args_identical(self):
        """Three calls with arg=None are identical → thrashing."""
        trace = _make_trace_with_tool_calls(
            "reset", [{"target": None}] * 3
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is True

    def test_nested_dict_args_differ(self):
        """Nested dicts with different values are distinct."""
        trace = _make_trace_with_tool_calls(
            "query",
            [
                {"filter": {"status": "active"}},
                {"filter": {"status": "inactive"}},
                {"filter": {"status": "pending"}},
            ],
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is False

    def test_nested_dict_args_identical(self):
        """Nested dicts with same values → thrashing."""
        trace = _make_trace_with_tool_calls(
            "query",
            [{"filter": {"status": "active"}}] * 3,
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is True


# ---------- Serialization unit tests ----------


class TestSerializeArgs:
    """Verify _serialize_args produces correct token sets.

    Success: numeric, bool, None, nested values are all preserved as
    distinct tokens; string values are tokenised into key=term pairs.
    Failure: values are dropped, collapsed, or misrepresented.
    """

    def test_int_preserved(self):
        assert _serialize_args({"page": 1}) == {"page=1"}

    def test_different_ints_distinct(self):
        assert _serialize_args({"page": 1}) != _serialize_args({"page": 2})

    def test_float_preserved(self):
        assert _serialize_args({"temp": 0.7}) == {"temp=0.7"}

    def test_bool_preserved(self):
        assert _serialize_args({"on": True}) == {"on=True"}
        assert _serialize_args({"on": False}) == {"on=False"}

    def test_none_preserved(self):
        assert _serialize_args({"val": None}) == {"val=None"}

    def test_string_tokenised(self):
        """String values become key=term pairs, not a single key=value."""
        tokens = _serialize_args({"q": "python framework"})
        assert "q=python" in tokens
        assert "q=framework" in tokens

    def test_nested_dict_serialised(self):
        tokens = _serialize_args({"filter": {"a": 1, "b": 2}})
        # Should be a single deterministic JSON token
        assert len(tokens) == 1
        token = next(iter(tokens))
        assert token.startswith("filter=")
        assert '"a": 1' in token

    def test_empty_args(self):
        assert _serialize_args({}) == set()


# ---------- Config validation ----------


class TestConfigValidation:
    """Verify invalid config values are rejected at construction time.

    Success: ValueError raised with descriptive message.
    Failure: nonsensical config silently accepted.
    """

    def test_min_repeats_too_low(self):
        with pytest.raises(ValueError, match="min_repeats must be >= 2"):
            ToolThrashingDetector(min_repeats=1)

    def test_min_repeats_zero(self):
        with pytest.raises(ValueError, match="min_repeats must be >= 2"):
            ToolThrashingDetector(min_repeats=0)

    def test_min_repeats_negative(self):
        with pytest.raises(ValueError, match="min_repeats must be >= 2"):
            ToolThrashingDetector(min_repeats=-5)

    def test_min_repeats_boundary_accepted(self):
        d = ToolThrashingDetector(min_repeats=2)
        assert d.min_repeats == 2

    def test_threshold_too_high(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            ToolThrashingDetector(similarity_threshold=1.1)

    def test_threshold_negative(self):
        with pytest.raises(ValueError, match="similarity_threshold"):
            ToolThrashingDetector(similarity_threshold=-0.1)

    def test_threshold_boundaries_accepted(self):
        d0 = ToolThrashingDetector(similarity_threshold=0.0)
        assert d0.similarity_threshold == 0.0
        d1 = ToolThrashingDetector(similarity_threshold=1.0)
        assert d1.similarity_threshold == 1.0

    def test_window_size_zero(self):
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            ToolThrashingDetector(window_size=0)

    def test_window_size_negative(self):
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            ToolThrashingDetector(window_size=-3)

    def test_window_size_one_accepted(self):
        d = ToolThrashingDetector(window_size=1)
        assert d.window_size == 1


# ---------- step_index sentinel ----------


class TestStepIndexSentinel:
    """Verify that missing step_index uses -1, not 0.

    Success: evidence shows -1 for unknown steps; never collides with
    real step 0.
    Failure: unknown step_index masquerades as step 0.
    """

    def test_none_step_index_uses_sentinel(self):
        """Messages without step_index get -1 in evidence."""
        trace = Trace(
            messages=[
                Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[ToolCall(tool_name="ping", arguments={})],
                    # step_index deliberately None
                )
                for _ in range(3)
            ]
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is True
        step_evidence = [e for e in result.evidence if "Step indices" in e][0]
        assert "-1" in step_evidence
        assert "0" not in step_evidence  # must NOT show 0

    def test_sentinel_value_is_negative_one(self):
        assert _UNKNOWN_STEP == -1

    def test_real_step_zero_not_confused(self):
        """A trace with real step 0 and None step should show distinct values."""
        trace = Trace(
            messages=[
                # Real step 0
                Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[ToolCall(tool_name="ping", arguments={})],
                    step_index=0,
                ),
                # None steps
                Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[ToolCall(tool_name="ping", arguments={})],
                ),
                Message(
                    role=Role.ASSISTANT,
                    content="",
                    tool_calls=[ToolCall(tool_name="ping", arguments={})],
                ),
            ]
        )
        result = ToolThrashingDetector().detect(trace)
        assert result.detected is True
        step_evidence = [e for e in result.evidence if "Step indices" in e][0]
        # Should contain both 0 (real) and -1 (sentinel)
        assert "-1" in step_evidence
        assert "0" in step_evidence


# ---------- Similarity threshold boundary tests ----------


class TestThresholdBoundary:
    """Verify the detector respects the similarity_threshold precisely.

    Success: calls at exactly the threshold are included; calls just below
    are excluded.
    Failure: boundary is off-by-one (>= vs >) or threshold is ignored.
    """

    def test_at_threshold_included(self):
        """Calls with Jaccard similarity exactly equal to threshold → similar.

        "alpha bravo charlie delta" vs "alpha bravo charlie delta echo"
        produces tokens with 4/5 Jaccard overlap = 0.8.  With threshold=0.8,
        these should be considered similar (>= comparison).
        """
        trace = _make_trace_with_tool_calls(
            "search",
            [
                {"q": "alpha bravo charlie delta"},
                {"q": "alpha bravo charlie delta echo"},
                {"q": "alpha bravo charlie delta"},
            ],
        )
        # threshold=0.8, similarity=0.8 → should be included
        result = ToolThrashingDetector(similarity_threshold=0.8).detect(trace)
        assert result.detected is True

    def test_below_threshold_excluded(self):
        """Calls with similarity below threshold → not similar.

        "alpha bravo charlie" (3 terms) vs "alpha bravo charlie delta echo"
        (5 terms) → Jaccard = 3/5 = 0.6.  With threshold=0.8, not similar.
        """
        trace = _make_trace_with_tool_calls(
            "search",
            [
                {"q": "alpha bravo charlie"},
                {"q": "alpha bravo charlie delta echo"},
                {"q": "alpha bravo charlie"},
            ],
        )
        # similarity between call 0 and 1 is 0.6 < 0.8
        result = ToolThrashingDetector(similarity_threshold=0.8).detect(trace)
        assert result.detected is False

    def test_threshold_zero_clusters_everything(self):
        """With threshold=0.0, even non-overlapping calls cluster together
        (any non-empty set has Jaccard >= 0 with any other)."""
        trace = _make_trace_with_tool_calls(
            "search",
            [
                {"q": "alpha"},
                {"q": "bravo"},
                {"q": "charlie"},
            ],
        )
        # similarity is 0.0 between these, and threshold is 0.0 → 0.0 >= 0.0
        result = ToolThrashingDetector(
            similarity_threshold=0.0, min_repeats=3
        ).detect(trace)
        assert result.detected is True

    def test_threshold_one_requires_exact_match(self):
        """With threshold=1.0, only identical token sets are similar."""
        trace = _make_trace_with_tool_calls(
            "search",
            [{"q": "python"}] * 3,
        )
        result = ToolThrashingDetector(similarity_threshold=1.0).detect(trace)
        assert result.detected is True

    def test_threshold_one_rejects_near_match(self):
        """Near-identical but not identical → not similar at threshold=1.0."""
        trace = _make_trace_with_tool_calls(
            "search",
            [
                {"q": "python framework"},
                {"q": "python framework testing"},
                {"q": "python framework"},
            ],
        )
        result = ToolThrashingDetector(similarity_threshold=1.0).detect(trace)
        assert result.detected is False


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
