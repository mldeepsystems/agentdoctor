"""Tests for the Silent Degradation detector."""

from __future__ import annotations

import pytest

from agentdoctor.detectors.silent_degradation import SilentDegradationDetector, _quality_score
from agentdoctor.models import Message, Role, Trace
from agentdoctor.taxonomy import Pathology


def _msg(role: Role, content: str, step: int | None = None) -> Message:
    return Message(role=role, content=content, step_index=step)


def _degrading_trace() -> Trace:
    """Trace where assistant responses get progressively shorter and less specific."""
    return Trace(
        trace_id="degrade-1",
        messages=[
            _msg(Role.SYSTEM, "You are a detailed technical writer."),
            _msg(Role.USER, "Explain databases.", step=0),
            _msg(
                Role.ASSISTANT,
                "Databases are sophisticated systems for storing, retrieving, and managing "
                "structured information. Relational databases use tables with predefined "
                "schemas, supporting complex queries through SQL. NoSQL databases offer "
                "flexible schemas for unstructured data, including document stores, "
                "key-value pairs, and graph databases. Performance optimization involves "
                "indexing strategies, query planning, and connection pooling.",
                step=1,
            ),
            _msg(Role.USER, "More about indexing?", step=2),
            _msg(
                Role.ASSISTANT,
                "Database indexing creates supplementary data structures that accelerate "
                "query performance. B-tree indexes handle range queries efficiently while "
                "hash indexes optimize exact-match lookups. Composite indexes combine "
                "multiple columns for complex query patterns.",
                step=3,
            ),
            _msg(Role.USER, "Transactions?", step=4),
            _msg(
                Role.ASSISTANT,
                "Transactions ensure database consistency through ACID properties. "
                "They group operations atomically and handle concurrent access.",
                step=5,
            ),
            _msg(Role.USER, "Replication?", step=6),
            _msg(
                Role.ASSISTANT,
                "Replication copies data across servers for availability.",
                step=7,
            ),
            _msg(Role.USER, "Sharding?", step=8),
            _msg(
                Role.ASSISTANT,
                "Sharding splits data.",
                step=9,
            ),
            _msg(Role.USER, "Caching?", step=10),
            _msg(
                Role.ASSISTANT,
                "Cache helps.",
                step=11,
            ),
        ],
    )


def _consistent_trace() -> Trace:
    """Trace with consistent quality throughout."""
    return Trace(
        trace_id="consistent-1",
        messages=[
            _msg(Role.SYSTEM, "You are a detailed technical writer."),
            _msg(Role.USER, "Topic 1?", step=0),
            _msg(
                Role.ASSISTANT,
                "Here is a detailed comprehensive explanation about this particular "
                "technical topic with sophisticated vocabulary and specific terminology.",
                step=1,
            ),
            _msg(Role.USER, "Topic 2?", step=2),
            _msg(
                Role.ASSISTANT,
                "This second topic involves intricate technical considerations "
                "requiring detailed analysis and comprehensive understanding.",
                step=3,
            ),
            _msg(Role.USER, "Topic 3?", step=4),
            _msg(
                Role.ASSISTANT,
                "The third technical subject encompasses sophisticated concepts "
                "that require thorough examination and detailed exploration.",
                step=5,
            ),
            _msg(Role.USER, "Topic 4?", step=6),
            _msg(
                Role.ASSISTANT,
                "This fourth area involves comprehensive technical knowledge "
                "with sophisticated methodologies and detailed implementation.",
                step=7,
            ),
            _msg(Role.USER, "Topic 5?", step=8),
            _msg(
                Role.ASSISTANT,
                "The fifth technical domain requires detailed understanding "
                "of sophisticated systems and comprehensive architectural design.",
                step=9,
            ),
            _msg(Role.USER, "Topic 6?", step=10),
            _msg(
                Role.ASSISTANT,
                "This sixth subject involves intricate technical specifications "
                "with detailed requirements and comprehensive documentation needs.",
                step=11,
            ),
        ],
    )


class TestQualityScore:
    def test_empty_text(self):
        assert _quality_score("") == 0.0

    def test_single_word(self):
        score = _quality_score("hello")
        assert 0.0 < score <= 1.0

    def test_longer_text_higher_score(self):
        short = _quality_score("OK")
        long = _quality_score(
            "This is a comprehensive and detailed explanation about sophisticated "
            "technical concepts involving multiple architectural considerations."
        )
        assert long > short

    def test_returns_float(self):
        assert isinstance(_quality_score("test text"), float)


class TestTruePositive:
    def test_detects_degradation(self):
        result = SilentDegradationDetector().detect(_degrading_trace())
        assert result.detected is True
        assert result.pathology is Pathology.SILENT_DEGRADATION

    def test_confidence_in_range(self):
        result = SilentDegradationDetector().detect(_degrading_trace())
        assert 0.0 <= result.confidence <= 1.0

    def test_evidence_contains_slope(self):
        result = SilentDegradationDetector().detect(_degrading_trace())
        evidence_text = " ".join(result.evidence)
        assert "slope" in evidence_text.lower()

    def test_evidence_contains_quality_scores(self):
        result = SilentDegradationDetector().detect(_degrading_trace())
        evidence_text = " ".join(result.evidence)
        assert "Quality scores" in evidence_text


class TestTrueNegative:
    def test_consistent_trace(self):
        result = SilentDegradationDetector().detect(_consistent_trace())
        assert result.detected is False

    def test_short_trace(self):
        trace = Trace(
            messages=[
                _msg(Role.USER, "Hi", step=0),
                _msg(Role.ASSISTANT, "Hello!", step=1),
            ]
        )
        result = SilentDegradationDetector().detect(trace)
        assert result.detected is False

    def test_empty_trace(self):
        result = SilentDegradationDetector().detect(Trace())
        assert result.detected is False

    def test_no_assistant_messages(self):
        trace = Trace(
            messages=[
                _msg(Role.USER, "Hello"),
                _msg(Role.USER, "Anyone there?"),
            ]
        )
        result = SilentDegradationDetector().detect(trace)
        assert result.detected is False


class TestConfig:
    def test_custom_min_messages(self):
        detector = SilentDegradationDetector(min_messages=20)
        result = detector.detect(_degrading_trace())
        assert result.detected is False

    def test_invalid_min_messages(self):
        with pytest.raises(ValueError):
            SilentDegradationDetector(min_messages=1)

    def test_invalid_slope_threshold(self):
        with pytest.raises(ValueError):
            SilentDegradationDetector(slope_threshold=0.1)

    def test_zero_slope_threshold_accepted(self):
        detector = SilentDegradationDetector(slope_threshold=0.0)
        assert detector is not None


class TestImports:
    def test_import_from_detectors(self):
        from agentdoctor.detectors import SilentDegradationDetector

        assert SilentDegradationDetector is not None

    def test_import_from_top_level(self):
        from agentdoctor import SilentDegradationDetector

        assert SilentDegradationDetector is not None
