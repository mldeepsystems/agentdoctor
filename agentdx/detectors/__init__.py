"""Pathology detectors for agentdx."""

from agentdx.detectors.base import BaseDetector
from agentdx.detectors.context_erosion import ContextErosionDetector
from agentdx.detectors.goal_hijacking import GoalHijackingDetector
from agentdx.detectors.hallucinated_tool_success import HallucinatedToolSuccessDetector
from agentdx.detectors.instruction_drift import InstructionDriftDetector
from agentdx.detectors.recovery_blindness import RecoveryBlindnessDetector
from agentdx.detectors.silent_degradation import SilentDegradationDetector
from agentdx.detectors.tool_thrashing import ToolThrashingDetector

ALL_DETECTORS: tuple[type[BaseDetector], ...] = (
    ToolThrashingDetector,
    ContextErosionDetector,
    RecoveryBlindnessDetector,
    HallucinatedToolSuccessDetector,
    InstructionDriftDetector,
    GoalHijackingDetector,
    SilentDegradationDetector,
)
"""Detector classes registered for default use by :class:`Diagnoser`."""

__all__ = [
    "ALL_DETECTORS",
    "BaseDetector",
    "ContextErosionDetector",
    "GoalHijackingDetector",
    "HallucinatedToolSuccessDetector",
    "InstructionDriftDetector",
    "RecoveryBlindnessDetector",
    "SilentDegradationDetector",
    "ToolThrashingDetector",
]
