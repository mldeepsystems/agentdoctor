"""Tests for the DiagnosticReport class."""

from __future__ import annotations

import json
import os
import tempfile

from agentdoctor.models import DetectorResult, Severity
from agentdoctor.report import DiagnosticReport
from agentdoctor.taxonomy import Pathology


def _make_result(
    pathology: Pathology,
    detected: bool,
    confidence: float = 0.5,
    severity: Severity = Severity.LOW,
    description: str = "",
    recommendation: str = "",
    evidence: list[str] | None = None,
) -> DetectorResult:
    return DetectorResult(
        pathology=pathology,
        detected=detected,
        confidence=confidence,
        severity=severity,
        description=description,
        recommendation=recommendation,
        evidence=evidence or [],
    )


class TestDetectedPathologies:
    def test_filters_only_detected(self):
        report = DiagnosticReport(
            trace_id="t-1",
            results=[
                _make_result(Pathology.TOOL_THRASHING, detected=True),
                _make_result(Pathology.CONTEXT_EROSION, detected=False),
                _make_result(Pathology.GOAL_HIJACKING, detected=True),
            ],
        )
        detected = report.detected_pathologies
        assert len(detected) == 2
        pathologies = {r.pathology for r in detected}
        assert pathologies == {Pathology.TOOL_THRASHING, Pathology.GOAL_HIJACKING}

    def test_empty_when_none_detected(self):
        report = DiagnosticReport(
            trace_id="t-2",
            results=[
                _make_result(Pathology.TOOL_THRASHING, detected=False),
                _make_result(Pathology.CONTEXT_EROSION, detected=False),
            ],
        )
        assert report.detected_pathologies == []

    def test_empty_results(self):
        report = DiagnosticReport(trace_id="t-3", results=[])
        assert report.detected_pathologies == []


class TestHighestSeverity:
    def test_returns_highest(self):
        report = DiagnosticReport(
            trace_id="t-4",
            results=[
                _make_result(Pathology.TOOL_THRASHING, detected=True, severity=Severity.LOW),
                _make_result(Pathology.GOAL_HIJACKING, detected=True, severity=Severity.CRITICAL),
                _make_result(Pathology.CONTEXT_EROSION, detected=True, severity=Severity.MEDIUM),
            ],
        )
        assert report.highest_severity is Severity.CRITICAL

    def test_returns_none_when_nothing_detected(self):
        report = DiagnosticReport(
            trace_id="t-5",
            results=[_make_result(Pathology.TOOL_THRASHING, detected=False)],
        )
        assert report.highest_severity is None

    def test_returns_none_for_empty_results(self):
        report = DiagnosticReport(trace_id="t-6", results=[])
        assert report.highest_severity is None

    def test_single_detected(self):
        report = DiagnosticReport(
            trace_id="t-7",
            results=[
                _make_result(Pathology.TOOL_THRASHING, detected=True, severity=Severity.HIGH),
            ],
        )
        assert report.highest_severity is Severity.HIGH

    def test_severity_ordering(self):
        """Verify CRITICAL > HIGH > MEDIUM > LOW."""
        for higher, lower in [
            (Severity.CRITICAL, Severity.HIGH),
            (Severity.HIGH, Severity.MEDIUM),
            (Severity.MEDIUM, Severity.LOW),
        ]:
            report = DiagnosticReport(
                trace_id="t-ord",
                results=[
                    _make_result(Pathology.TOOL_THRASHING, detected=True, severity=lower),
                    _make_result(Pathology.GOAL_HIJACKING, detected=True, severity=higher),
                ],
            )
            assert report.highest_severity is higher


class TestSummary:
    def test_contains_trace_id(self):
        report = DiagnosticReport(trace_id="my-trace", results=[])
        assert "my-trace" in report.summary()

    def test_unknown_trace_id_when_none(self):
        report = DiagnosticReport(trace_id=None, results=[])
        assert "unknown" in report.summary()

    def test_contains_detected_count(self):
        report = DiagnosticReport(
            trace_id="t-s",
            results=[
                _make_result(
                    Pathology.TOOL_THRASHING,
                    detected=True,
                    severity=Severity.HIGH,
                    description="5 repeated calls",
                ),
                _make_result(Pathology.CONTEXT_EROSION, detected=False),
            ],
        )
        s = report.summary()
        assert "1/2" in s
        assert "Tool Thrashing" in s
        assert "Context Erosion" in s

    def test_clean_report_no_detections(self):
        report = DiagnosticReport(
            trace_id="clean",
            results=[_make_result(Pathology.TOOL_THRASHING, detected=False)],
        )
        s = report.summary()
        assert "0/1" in s
        assert "No pathologies detected" in s

    def test_overall_severity_in_summary(self):
        report = DiagnosticReport(
            trace_id="sev",
            results=[
                _make_result(Pathology.TOOL_THRASHING, detected=True, severity=Severity.HIGH),
            ],
        )
        assert "HIGH" in report.summary()


class TestToDict:
    def test_produces_serializable_dict(self):
        report = DiagnosticReport(
            trace_id="d-1",
            results=[
                _make_result(
                    Pathology.TOOL_THRASHING,
                    detected=True,
                    confidence=0.85,
                    severity=Severity.HIGH,
                    description="test desc",
                    recommendation="test rec",
                    evidence=["evidence line 1"],
                ),
            ],
            metadata={"source": "test"},
        )
        d = report.to_dict()
        # Should be JSON-serializable (no enums left)
        json_str = json.dumps(d)
        loaded = json.loads(json_str)
        assert loaded["trace_id"] == "d-1"
        assert loaded["overall_severity"] == "high"
        assert len(loaded["results"]) == 1
        r = loaded["results"][0]
        assert r["pathology"] == "tool_thrashing"
        assert r["detected"] is True
        assert r["confidence"] == 0.85
        assert r["severity"] == "high"
        assert r["evidence"] == ["evidence line 1"]
        assert r["description"] == "test desc"
        assert r["recommendation"] == "test rec"
        assert loaded["metadata"] == {"source": "test"}

    def test_overall_severity_none_when_clean(self):
        report = DiagnosticReport(
            trace_id="d-2",
            results=[_make_result(Pathology.TOOL_THRASHING, detected=False)],
        )
        assert report.to_dict()["overall_severity"] is None

    def test_empty_results(self):
        report = DiagnosticReport(trace_id="d-3", results=[])
        d = report.to_dict()
        assert d["results"] == []
        assert d["overall_severity"] is None


class TestToJson:
    def test_returns_valid_json_string(self):
        report = DiagnosticReport(
            trace_id="j-1",
            results=[_make_result(Pathology.TOOL_THRASHING, detected=True)],
        )
        json_str = report.to_json()
        loaded = json.loads(json_str)
        assert loaded["trace_id"] == "j-1"

    def test_writes_file(self):
        report = DiagnosticReport(
            trace_id="j-2",
            results=[_make_result(Pathology.TOOL_THRASHING, detected=True)],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            report.to_json(path)
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            assert loaded["trace_id"] == "j-2"
        finally:
            os.unlink(path)

    def test_json_is_indented(self):
        report = DiagnosticReport(trace_id="j-3", results=[])
        json_str = report.to_json()
        assert "\n" in json_str  # indent=2 produces newlines


class TestToMarkdown:
    def test_contains_header(self):
        report = DiagnosticReport(
            trace_id="m-1",
            results=[_make_result(Pathology.TOOL_THRASHING, detected=True)],
        )
        md = report.to_markdown()
        assert "# AgentDoctor Diagnostic Report" in md

    def test_contains_summary_table(self):
        report = DiagnosticReport(
            trace_id="m-2",
            results=[
                _make_result(
                    Pathology.TOOL_THRASHING,
                    detected=True,
                    severity=Severity.HIGH,
                    confidence=0.9,
                ),
                _make_result(Pathology.CONTEXT_EROSION, detected=False, confidence=0.8),
            ],
        )
        md = report.to_markdown()
        assert "| Pathology |" in md
        assert "Tool Thrashing" in md
        assert "Context Erosion" in md
        assert "90%" in md
        assert "HIGH" in md

    def test_detected_pathologies_section(self):
        report = DiagnosticReport(
            trace_id="m-3",
            results=[
                _make_result(
                    Pathology.TOOL_THRASHING,
                    detected=True,
                    description="test description",
                    recommendation="try caching",
                    evidence=["evidence item 1"],
                ),
            ],
        )
        md = report.to_markdown()
        assert "## Detected Pathologies" in md
        assert "### Tool Thrashing" in md
        assert "test description" in md
        assert "try caching" in md
        assert "- evidence item 1" in md

    def test_no_detected_section_when_clean(self):
        report = DiagnosticReport(
            trace_id="m-4",
            results=[_make_result(Pathology.TOOL_THRASHING, detected=False)],
        )
        md = report.to_markdown()
        assert "## Detected Pathologies" not in md

    def test_writes_file(self):
        report = DiagnosticReport(
            trace_id="m-5",
            results=[_make_result(Pathology.TOOL_THRASHING, detected=True)],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = f.name
        try:
            report.to_markdown(path)
            with open(path, encoding="utf-8") as f:
                content = f.read()
            assert "# AgentDoctor Diagnostic Report" in content
        finally:
            os.unlink(path)


class TestFullReport:
    def test_all_pathologies_detected(self):
        results = [_make_result(p, detected=True, severity=Severity.HIGH) for p in Pathology]
        report = DiagnosticReport(trace_id="full", results=results)
        assert len(report.detected_pathologies) == 7
        assert report.highest_severity is Severity.HIGH
        s = report.summary()
        assert "7/7" in s

    def test_round_trip_json(self):
        """to_dict() → json.dumps → json.loads produces identical data."""
        results = [
            _make_result(
                Pathology.TOOL_THRASHING,
                detected=True,
                confidence=0.95,
                severity=Severity.CRITICAL,
                evidence=["e1", "e2"],
            ),
            _make_result(Pathology.CONTEXT_EROSION, detected=False, confidence=0.7),
        ]
        report = DiagnosticReport(trace_id="rt", results=results, metadata={"k": "v"})
        d1 = report.to_dict()
        d2 = json.loads(json.dumps(d1))
        assert d1 == d2
