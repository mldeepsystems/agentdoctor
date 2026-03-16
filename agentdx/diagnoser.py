"""Diagnoser orchestrator for agentdx."""

from __future__ import annotations

from agentdx.detectors import ALL_DETECTORS, BaseDetector
from agentdx.models import DetectorResult, Severity, Trace
from agentdx.report import DiagnosticReport


class Diagnoser:
    """Primary API entry point that runs detectors against a trace.

    By default, all registered detectors are used. Pass a custom list
    to run only a subset::

        diagnoser = Diagnoser()                          # all detectors
        diagnoser = Diagnoser(detectors=[MyDetector()])   # custom subset
    """

    def __init__(self, detectors: list[BaseDetector] | None = None) -> None:
        if detectors is not None:
            self._detectors = list(detectors)
        else:
            self._detectors = [cls() for cls in ALL_DETECTORS]

    @property
    def detectors(self) -> list[BaseDetector]:
        """The detector instances this diagnoser will run."""
        return list(self._detectors)

    def diagnose(self, trace: Trace) -> DiagnosticReport:
        """Run all detectors against *trace* and return a report.

        Each detector runs independently — a single detector raising an
        exception will not block the others.  Failed detectors produce a
        result with ``detected=False`` and a description noting the error.
        """
        results: list[DetectorResult] = []
        for detector in self._detectors:
            try:
                result = detector.detect(trace)
                results.append(result)
            except Exception as exc:
                results.append(
                    DetectorResult(
                        pathology=detector.pathology,
                        detected=False,
                        confidence=0.0,
                        severity=Severity.LOW,
                        description=f"Detector error: {exc}",
                    )
                )
        return DiagnosticReport(
            trace_id=trace.trace_id,
            results=results,
            metadata=trace.metadata,
        )
