"""Pathology detectors for AgentDoctor."""

from agentdoctor.detectors.base import BaseDetector
from agentdoctor.detectors.tool_thrashing import ToolThrashingDetector

ALL_DETECTORS: list[type[BaseDetector]] = [ToolThrashingDetector]
"""Detector classes registered for default use by :class:`Diagnoser`."""

__all__ = ["ALL_DETECTORS", "BaseDetector", "ToolThrashingDetector"]
