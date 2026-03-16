"""Pathology detectors for AgentDoctor."""

from agentdoctor.detectors.base import BaseDetector
from agentdoctor.detectors.tool_thrashing import ToolThrashingDetector

__all__ = ["BaseDetector", "ToolThrashingDetector"]
