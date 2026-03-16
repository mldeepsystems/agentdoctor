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
from agentdoctor.taxonomy import PATHOLOGY_REGISTRY, Pathology

__all__ = [
    "DetectorResult",
    "Message",
    "PATHOLOGY_REGISTRY",
    "Pathology",
    "Role",
    "Severity",
    "ToolCall",
    "Trace",
]
