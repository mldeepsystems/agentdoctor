from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agentdx")
except PackageNotFoundError:
    __version__ = "0.1.0a1"  # fallback for uninstalled usage

from agentdx.models import (
    DetectorResult,
    Message,
    Role,
    Severity,
    ToolCall,
    Trace,
)
from agentdx.detectors import (
    ALL_DETECTORS,
    BaseDetector,
    ContextErosionDetector,
    GoalHijackingDetector,
    HallucinatedToolSuccessDetector,
    InstructionDriftDetector,
    RecoveryBlindnessDetector,
    SilentDegradationDetector,
    ToolThrashingDetector,
)
from agentdx.diagnoser import Diagnoser
from agentdx.parsers import BaseParser, JSONParser
from agentdx.report import DiagnosticReport
from agentdx.taxonomy import PATHOLOGY_REGISTRY, Pathology

__all__ = [
    "ALL_DETECTORS",
    "BaseDetector",
    "BaseParser",
    "ContextErosionDetector",
    "Diagnoser",
    "DiagnosticReport",
    "GoalHijackingDetector",
    "HallucinatedToolSuccessDetector",
    "InstructionDriftDetector",
    "RecoveryBlindnessDetector",
    "SilentDegradationDetector",
    "ToolThrashingDetector",
    "DetectorResult",
    "JSONParser",
    "Message",
    "PATHOLOGY_REGISTRY",
    "Pathology",
    "Role",
    "Severity",
    "ToolCall",
    "Trace",
]
