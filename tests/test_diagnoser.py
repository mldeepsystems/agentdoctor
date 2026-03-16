"""Tests for the Diagnoser orchestrator."""

from __future__ import annotations

from agentdx.detectors import ALL_DETECTORS
from agentdx.detectors.base import BaseDetector
from agentdx.diagnoser import Diagnoser
from agentdx.models import DetectorResult, Severity, Trace
from agentdx.parsers import JSONParser
from agentdx.report import DiagnosticReport
from agentdx.taxonomy import Pathology


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AlwaysDetects(BaseDetector):
    """Stub detector that always reports a detection."""

    @property
    def pathology(self) -> Pathology:
        return Pathology.CONTEXT_EROSION

    def detect(self, trace: Trace) -> DetectorResult:
        return DetectorResult(
            pathology=self.pathology,
            detected=True,
            confidence=0.9,
            severity=Severity.HIGH,
            description="Stub detection",
        )


class _NeverDetects(BaseDetector):
    """Stub detector that never reports a detection."""

    @property
    def pathology(self) -> Pathology:
        return Pathology.INSTRUCTION_DRIFT

    def detect(self, trace: Trace) -> DetectorResult:
        return DetectorResult(
            pathology=self.pathology,
            detected=False,
            confidence=0.8,
        )


class _ExplodingDetector(BaseDetector):
    """Stub detector that raises an exception."""

    @property
    def pathology(self) -> Pathology:
        return Pathology.RECOVERY_BLINDNESS

    def detect(self, trace: Trace) -> DetectorResult:
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDefaultConstructor:
    def test_creates_all_registered_detectors(self):
        diagnoser = Diagnoser()
        assert len(diagnoser.detectors) == len(ALL_DETECTORS)

    def test_detector_types_match_registry(self):
        diagnoser = Diagnoser()
        actual_types = {type(d) for d in diagnoser.detectors}
        expected_types = set(ALL_DETECTORS)
        assert actual_types == expected_types


class TestCustomConstructor:
    def test_accepts_custom_detectors(self):
        custom = [_AlwaysDetects(), _NeverDetects()]
        diagnoser = Diagnoser(detectors=custom)
        assert len(diagnoser.detectors) == 2

    def test_empty_list(self):
        diagnoser = Diagnoser(detectors=[])
        assert diagnoser.detectors == []


class TestDiagnose:
    def test_returns_diagnostic_report(self):
        diagnoser = Diagnoser(detectors=[_NeverDetects()])
        trace = Trace(trace_id="test-1")
        report = diagnoser.diagnose(trace)
        assert isinstance(report, DiagnosticReport)

    def test_report_contains_correct_result_count(self):
        diagnoser = Diagnoser(detectors=[_AlwaysDetects(), _NeverDetects()])
        report = diagnoser.diagnose(Trace(trace_id="test-2"))
        assert len(report.results) == 2

    def test_report_trace_id_matches(self):
        diagnoser = Diagnoser(detectors=[_NeverDetects()])
        report = diagnoser.diagnose(Trace(trace_id="abc-123"))
        assert report.trace_id == "abc-123"

    def test_report_metadata_preserved(self):
        diagnoser = Diagnoser(detectors=[_NeverDetects()])
        trace = Trace(trace_id="t", metadata={"env": "test"})
        report = diagnoser.diagnose(trace)
        assert report.metadata == {"env": "test"}

    def test_healthy_trace_no_detections(self):
        diagnoser = Diagnoser(detectors=[_NeverDetects()])
        report = diagnoser.diagnose(Trace(trace_id="healthy"))
        assert report.detected_pathologies == []

    def test_pathology_detected(self):
        diagnoser = Diagnoser(detectors=[_AlwaysDetects()])
        report = diagnoser.diagnose(Trace(trace_id="sick"))
        detected = report.detected_pathologies
        assert len(detected) == 1
        assert detected[0].pathology is Pathology.CONTEXT_EROSION

    def test_mixed_results(self):
        diagnoser = Diagnoser(detectors=[_AlwaysDetects(), _NeverDetects()])
        report = diagnoser.diagnose(Trace(trace_id="mixed"))
        assert len(report.detected_pathologies) == 1
        assert len(report.results) == 2

    def test_empty_trace(self):
        diagnoser = Diagnoser(detectors=[_NeverDetects()])
        report = diagnoser.diagnose(Trace())
        assert isinstance(report, DiagnosticReport)
        assert report.trace_id is None

    def test_no_detectors_empty_results(self):
        diagnoser = Diagnoser(detectors=[])
        report = diagnoser.diagnose(Trace(trace_id="empty"))
        assert report.results == []
        assert report.detected_pathologies == []


class TestDetectorIsolation:
    def test_one_failure_does_not_block_others(self):
        diagnoser = Diagnoser(detectors=[_AlwaysDetects(), _ExplodingDetector(), _NeverDetects()])
        report = diagnoser.diagnose(Trace(trace_id="isolation"))
        # All 3 detectors should produce results
        assert len(report.results) == 3
        # The exploding one should produce a non-detected result
        exploded = [r for r in report.results if r.pathology is Pathology.RECOVERY_BLINDNESS]
        assert len(exploded) == 1
        assert exploded[0].detected is False
        assert "boom" in exploded[0].description

    def test_failed_detector_does_not_affect_detected(self):
        diagnoser = Diagnoser(detectors=[_AlwaysDetects(), _ExplodingDetector()])
        report = diagnoser.diagnose(Trace(trace_id="fail"))
        detected = report.detected_pathologies
        assert len(detected) == 1
        assert detected[0].pathology is Pathology.CONTEXT_EROSION


class TestEndToEnd:
    def test_with_json_parser_healthy(self):
        """Parse a healthy fixture and run default detectors."""
        parser = JSONParser()
        trace = parser.parse("tests/fixtures/traces/healthy_trace.json")
        diagnoser = Diagnoser()
        report = diagnoser.diagnose(trace)
        assert isinstance(report, DiagnosticReport)
        assert report.trace_id == "healthy-001"
        # Healthy trace should have no tool thrashing
        for r in report.results:
            if r.pathology is Pathology.TOOL_THRASHING:
                assert r.detected is False

    def test_with_json_parser_thrashing(self):
        """Parse a thrashing fixture and verify detection."""
        parser = JSONParser()
        trace = parser.parse("tests/fixtures/traces/thrashing_trace.json")
        diagnoser = Diagnoser()
        report = diagnoser.diagnose(trace)
        assert isinstance(report, DiagnosticReport)
        assert report.trace_id == "thrashing-001"
        # Thrashing trace should be detected
        thrashing = [r for r in report.results if r.pathology is Pathology.TOOL_THRASHING]
        assert len(thrashing) == 1
        assert thrashing[0].detected is True

    def test_full_pipeline_summary(self):
        """Parse → diagnose → summary round-trip."""
        parser = JSONParser()
        trace = parser.parse("tests/fixtures/traces/thrashing_trace.json")
        report = Diagnoser().diagnose(trace)
        summary = report.summary()
        assert "thrashing-001" in summary
        assert "Tool Thrashing" in summary


class TestImports:
    def test_import_from_top_level(self):
        from agentdx import Diagnoser, DiagnosticReport

        assert Diagnoser is not None
        assert DiagnosticReport is not None

    def test_all_detectors_exported(self):
        from agentdx import ALL_DETECTORS

        assert isinstance(ALL_DETECTORS, tuple)
        assert len(ALL_DETECTORS) >= 1
