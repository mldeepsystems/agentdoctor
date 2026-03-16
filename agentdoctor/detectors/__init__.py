"""Pathology detectors for AgentDoctor."""

from agentdoctor.detectors.base import BaseDetector
from agentdoctor.detectors.context_erosion import ContextErosionDetector
from agentdoctor.detectors.hallucinated_tool_success import HallucinatedToolSuccessDetector
from agentdoctor.detectors.recovery_blindness import RecoveryBlindnessDetector
from agentdoctor.detectors.tool_thrashing import ToolThrashingDetector

ALL_DETECTORS: tuple[type[BaseDetector], ...] = (
    ToolThrashingDetector,
    ContextErosionDetector,
    RecoveryBlindnessDetector,
    HallucinatedToolSuccessDetector,
)
"""Detector classes registered for default use by :class:`Diagnoser`."""

__all__ = [
    "ALL_DETECTORS",
    "BaseDetector",
    "ContextErosionDetector",
    "HallucinatedToolSuccessDetector",
    "RecoveryBlindnessDetector",
    "ToolThrashingDetector",
]
