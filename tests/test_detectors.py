"""Tests for the BaseDetector interface and detector registry."""

from __future__ import annotations

import pytest

from agentdx.detectors import ALL_DETECTORS
from agentdx.detectors.base import BaseDetector
from agentdx.models import DetectorResult, Severity, Trace
from agentdx.taxonomy import Pathology


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
                return DetectorResult(pathology=Pathology.TOOL_THRASHING, detected=False)

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


class TestAllDetectorsRegistry:
    def test_is_tuple(self):
        assert isinstance(ALL_DETECTORS, tuple)

    def test_immutable(self):
        with pytest.raises(AttributeError):
            ALL_DETECTORS.append(None)  # type: ignore[attr-defined]

    def test_all_are_base_detector_subclasses(self):
        for cls in ALL_DETECTORS:
            assert issubclass(cls, BaseDetector)
