from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class Pathology(str, Enum):
    """The seven operational failure pathologies detected by AgentDoctor."""

    CONTEXT_EROSION = "context_erosion"
    TOOL_THRASHING = "tool_thrashing"
    INSTRUCTION_DRIFT = "instruction_drift"
    RECOVERY_BLINDNESS = "recovery_blindness"
    HALLUCINATED_TOOL_SUCCESS = "hallucinated_tool_success"
    GOAL_HIJACKING = "goal_hijacking"
    SILENT_DEGRADATION = "silent_degradation"


@dataclass(frozen=True)
class PathologyInfo:
    """Metadata for a single pathology."""

    pathology: Pathology
    name: str
    description: str
    owasp_mapping: str
    mast_mapping: str


__all__ = ["Pathology", "PathologyInfo", "PATHOLOGY_REGISTRY"]

PATHOLOGY_REGISTRY: MappingProxyType[Pathology, PathologyInfo] = MappingProxyType(
    {
        Pathology.CONTEXT_EROSION: PathologyInfo(
            pathology=Pathology.CONTEXT_EROSION,
            name="Context Erosion",
            description=(
                "Agent loses critical context over long conversations or multi-step "
                "tasks, leading to decisions that contradict earlier information."
            ),
            owasp_mapping="OWASP Agentic Top 10: A05 — Improper Multi-Agent Orchestration",
            mast_mapping="MAST: Task Derailment — context loss across planning steps",
        ),
        Pathology.TOOL_THRASHING: PathologyInfo(
            pathology=Pathology.TOOL_THRASHING,
            name="Tool Thrashing",
            description=(
                "Agent repeatedly calls tools with ineffective or contradictory "
                "parameters, wasting resources without making progress."
            ),
            owasp_mapping="OWASP Agentic Top 10: A04 — Unrestricted Tool Utilisation",
            mast_mapping="MAST: Execution Failure — repeated ineffective tool invocations",
        ),
        Pathology.INSTRUCTION_DRIFT: PathologyInfo(
            pathology=Pathology.INSTRUCTION_DRIFT,
            name="Instruction Drift",
            description=(
                "Agent gradually deviates from its original instructions or mandate, "
                "producing outputs that no longer align with the user's intent."
            ),
            owasp_mapping="OWASP Agentic Top 10: A02 — Privilege Compromise via Agent Hijacking",
            mast_mapping="MAST: Task Derailment — goal divergence from original instructions",
        ),
        Pathology.RECOVERY_BLINDNESS: PathologyInfo(
            pathology=Pathology.RECOVERY_BLINDNESS,
            name="Recovery Blindness",
            description=(
                "Agent fails to detect or recover from errors in its own execution, "
                "continuing to operate as though no failure occurred."
            ),
            owasp_mapping="OWASP Agentic Top 10: A08 — Inadequate Sandboxing & Containment",
            mast_mapping="MAST: Execution Failure — inability to recover from errors",
        ),
        Pathology.HALLUCINATED_TOOL_SUCCESS: PathologyInfo(
            pathology=Pathology.HALLUCINATED_TOOL_SUCCESS,
            name="Hallucinated Tool Success",
            description=(
                "Agent treats a failed tool call as successful and proceeds on false "
                "premises, propagating errors through the execution chain."
            ),
            owasp_mapping="OWASP Agentic Top 10: A06 — Overreliance on Agentic Systems",
            mast_mapping="MAST: Execution Failure — misinterpreted tool output",
        ),
        Pathology.GOAL_HIJACKING: PathologyInfo(
            pathology=Pathology.GOAL_HIJACKING,
            name="Goal Hijacking",
            description=(
                "Agent's objective is altered by adversarial input or environmental "
                "manipulation, causing it to pursue unintended goals."
            ),
            owasp_mapping="OWASP Agentic Top 10: A01 — Agentic Prompt Injection",
            mast_mapping="MAST: Task Derailment — adversarial goal redirection",
        ),
        Pathology.SILENT_DEGRADATION: PathologyInfo(
            pathology=Pathology.SILENT_DEGRADATION,
            name="Silent Degradation",
            description=(
                "Agent's output quality deteriorates over time without triggering "
                "explicit errors, making the failure difficult to detect."
            ),
            owasp_mapping="OWASP Agentic Top 10: A10 — Misaligned Behaviours & Emergent Misuse",
            mast_mapping="MAST: Output Quality — progressive decline in response quality",
        ),
    }
)
