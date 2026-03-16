"""Pathology detectors for AgentDoctor."""

from agentdoctor.detectors.base import BaseDetector
from agentdoctor.detectors.context_erosion import ContextErosionDetector
from agentdoctor.detectors.goal_hijacking import GoalHijackingDetector
from agentdoctor.detectors.hallucinated_tool_success import HallucinatedToolSuccessDetector
from agentdoctor.detectors.instruction_drift import InstructionDriftDetector
from agentdoctor.detectors.recovery_blindness import RecoveryBlindnessDetector
from agentdoctor.detectors.silent_degradation import SilentDegradationDetector
from agentdoctor.detectors.tool_thrashing import ToolThrashingDetector

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
