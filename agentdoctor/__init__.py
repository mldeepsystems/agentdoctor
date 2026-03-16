from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("agentdoctor")
except PackageNotFoundError:
    __version__ = "0.1.0"  # fallback for uninstalled usage

from agentdoctor.models import (
    DetectorResult,
    Message,
    Role,
    Severity,
    ToolCall,
    Trace,
)
from agentdoctor.detectors import ALL_DETECTORS, BaseDetector, ToolThrashingDetector
from agentdoctor.diagnoser import Diagnoser
from agentdoctor.parsers import BaseParser, JSONParser
from agentdoctor.report import DiagnosticReport
from agentdoctor.taxonomy import PATHOLOGY_REGISTRY, Pathology

__all__ = [
    "ALL_DETECTORS",
    "BaseDetector",
    "BaseParser",
    "Diagnoser",
    "DiagnosticReport",
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
