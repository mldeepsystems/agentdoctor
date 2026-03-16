"""Tool Thrashing detector for AgentDoctor."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentdoctor._text_utils import extract_key_terms, term_overlap
from agentdoctor.detectors.base import BaseDetector
from agentdoctor.models import DetectorResult, Severity, Trace
from agentdoctor.taxonomy import Pathology


@dataclass
class _IndexedCall:
    """A tool call paired with its message step_index."""

    tool_name: str
    arguments: dict[str, Any]
    step_index: int


def _serialize_args(arguments: dict[str, Any]) -> set[str]:
    """Convert arguments dict to a set of terms for similarity comparison.

    Each key and value is tokenised into meaningful terms via
    :func:`extract_key_terms` so that Jaccard similarity captures
    partial overlap between near-identical argument sets.
    """
    parts: list[str] = []
    for key, value in arguments.items():
        parts.append(key)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, (int, float, bool)) or value is None:
            parts.append(str(value))
        else:
            parts.append(json.dumps(value, sort_keys=True))
    return extract_key_terms(" ".join(parts))


def _pairwise_similar(
    calls: list[_IndexedCall], threshold: float
) -> list[_IndexedCall]:
    """Find the largest cluster of mutually similar calls.

    Two calls are similar if their serialized argument sets have a
    ``term_overlap`` above *threshold*, or if both have empty arguments.
    """
    n = len(calls)
    if n == 0:
        return []

    serialized = [_serialize_args(c.arguments) for c in calls]

    # Build similarity matrix
    similar: list[list[bool]] = [[False] * n for _ in range(n)]
    for i in range(n):
        similar[i][i] = True
        for j in range(i + 1, n):
            if not serialized[i] and not serialized[j]:
                # Both empty → identical
                sim = 1.0
            else:
                sim = term_overlap(serialized[i], serialized[j])
            if sim > threshold:
                similar[i][j] = True
                similar[j][i] = True

    # Greedy largest clique: start from each node, grow greedily
    best_cluster: list[int] = []
    for start in range(n):
        cluster = [start]
        for candidate in range(n):
            if candidate == start:
                continue
            if all(similar[candidate][member] for member in cluster):
                cluster.append(candidate)
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    return [calls[i] for i in best_cluster]


class ToolThrashingDetector(BaseDetector):
    """Detect repeated tool calls with similar arguments (tool thrashing).

    Config:
        min_repeats: Minimum cluster size to flag as thrashing. Default 3.
        similarity_threshold: Minimum pairwise similarity for arguments
            to be considered "similar". Default 0.8.
        window_size: Sliding window size over per-tool calls. Default 5.
    """

    def __init__(
        self,
        min_repeats: int = 3,
        similarity_threshold: float = 0.8,
        window_size: int = 5,
    ) -> None:
        self.min_repeats = min_repeats
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size

    @property
    def pathology(self) -> Pathology:
        return Pathology.TOOL_THRASHING

    def detect(self, trace: Trace) -> DetectorResult:
        # Step 1: Collect all tool calls with their step indices
        indexed_calls: list[_IndexedCall] = []
        for msg in trace.messages:
            for tc in msg.tool_calls:
                indexed_calls.append(
                    _IndexedCall(
                        tool_name=tc.tool_name,
                        arguments=dict(tc.arguments),
                        step_index=msg.step_index if msg.step_index is not None else 0,
                    )
                )

        if not indexed_calls:
            return self._no_detection()

        # Step 2: Group by tool_name
        by_tool: dict[str, list[_IndexedCall]] = {}
        for call in indexed_calls:
            by_tool.setdefault(call.tool_name, []).append(call)

        # Step 3-7: Find worst thrashing across all tools
        worst_tool: str | None = None
        worst_cluster: list[_IndexedCall] = []

        for tool_name, calls in by_tool.items():
            # Slide window
            for start in range(len(calls)):
                end = min(start + self.window_size, len(calls))
                window = calls[start:end]
                if len(window) < self.min_repeats:
                    continue
                cluster = _pairwise_similar(window, self.similarity_threshold)
                if len(cluster) >= self.min_repeats and len(cluster) > len(
                    worst_cluster
                ):
                    worst_cluster = cluster
                    worst_tool = tool_name

        if not worst_cluster:
            return self._no_detection()

        repeat_count = len(worst_cluster)
        confidence = min(1.0, repeat_count / 5)
        severity = self._severity(repeat_count)
        step_indices = [c.step_index for c in worst_cluster]
        arg_values = [c.arguments for c in worst_cluster]

        return DetectorResult(
            pathology=Pathology.TOOL_THRASHING,
            detected=True,
            confidence=confidence,
            severity=severity,
            evidence=[
                f"Tool '{worst_tool}' called {repeat_count} times with similar arguments",
                f"Step indices: {step_indices}",
                f"Arguments: {arg_values}",
            ],
            description=(
                f"Tool thrashing detected: '{worst_tool}' was called "
                f"{repeat_count} times with near-identical arguments."
            ),
            recommendation=(
                "Consider varying the tool arguments, using a different tool, "
                "or adding logic to detect when a tool call is not making progress."
            ),
        )

    def _no_detection(self) -> DetectorResult:
        return DetectorResult(
            pathology=Pathology.TOOL_THRASHING,
            detected=False,
            confidence=0.0,
        )

    @staticmethod
    def _severity(repeat_count: int) -> Severity:
        if repeat_count >= 5:
            return Severity.CRITICAL
        if repeat_count >= 4:
            return Severity.HIGH
        return Severity.MEDIUM
