"""Tests for the BaseDetector interface."""

from __future__ import annotations

import pytest

from agentdoctor.detectors.base import BaseDetector
from agentdoctor.models import DetectorResult, Severity, Trace
from agentdoctor.taxonomy import Pathology


class TestBaseDetector:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseDetector()

    def test_partial_subclass_only_pathology(self):
        class OnlyPathology(BaseDetector):
            @property
            def pathology(self) -> Pathology:
                return Pathology.TOOL_THRASHING

        with pytest.raises(TypeError):
            OnlyPathology()

    def test_partial_subclass_only_detect(self):
        class OnlyDetect(BaseDetector):
            def detect(self, trace: Trace) -> DetectorResult:
                return DetectorResult(
                    pathology=Pathology.TOOL_THRASHING, detected=False
                )

        with pytest.raises(TypeError):
            OnlyDetect()

    def test_complete_subclass(self):
        class Complete(BaseDetector):
            @property
            def pathology(self) -> Pathology:
                return Pathology.TOOL_THRASHING

            def detect(self, trace: Trace) -> DetectorResult:
                return DetectorResult(
                    pathology=self.pathology,
                    detected=True,
                    confidence=0.9,
                    severity=Severity.HIGH,
                )

        detector = Complete()
        assert detector.pathology is Pathology.TOOL_THRASHING

        result = detector.detect(Trace())
        assert isinstance(result, DetectorResult)
        assert result.detected is True
        assert result.confidence == 0.9
        assert result.severity is Severity.HIGH
        assert result.pathology is Pathology.TOOL_THRASHING
