"""Tool Thrashing detector for agentdx."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agentdx._text_utils import extract_key_terms, term_overlap
from agentdx.detectors.base import BaseDetector
from agentdx.models import DetectorResult, Severity, Trace
from agentdx.taxonomy import Pathology

# Sentinel for messages where step_index was not assigned by the parser.
# Using -1 avoids collisions with real step indices (which start at 0).
_UNKNOWN_STEP: int = -1


@dataclass
class _IndexedCall:
    """A tool call paired with its message step_index."""

    tool_name: str
    arguments: dict[str, Any]
    step_index: int


def _serialize_args(arguments: dict[str, Any]) -> set[str]:
    """Convert arguments dict to a set of ``key=value`` tokens for comparison.

    Each argument becomes a single token ``"key=normalised_value"`` so that
    Jaccard similarity can distinguish calls that differ only in value —
    including numeric values like ``page=1`` vs ``page=2``.

    String values are additionally split into key terms (via
    :func:`extract_key_terms`) so that ``query="best python framework"``
    and ``query="top python framework"`` are recognised as partially
    overlapping rather than completely disjoint.
    """
    tokens: set[str] = set()
    for key, value in arguments.items():
        if isinstance(value, str):
            # For strings, include the key AND extract key terms from the
            # value so partial textual overlap is captured.
            for term in extract_key_terms(value):
                tokens.add(f"{key}={term}")
        elif isinstance(value, (int, float, bool)) or value is None:
            # Numeric/bool/None values are preserved verbatim so that
            # page=1 and page=2 produce distinct tokens.
            tokens.add(f"{key}={value}")
        else:
            # Nested structures: deterministic JSON serialization.
            tokens.add(f"{key}={json.dumps(value, sort_keys=True)}")
    return tokens


def _pairwise_similar(calls: list[_IndexedCall], threshold: float) -> list[_IndexedCall]:
    """Find the largest cluster of mutually similar calls.

    Two calls are similar if their serialized argument token sets have a
    Jaccard similarity (via :func:`term_overlap`) at or above *threshold*,
    or if both have empty arguments.

    Uses a greedy clique approximation: tries each node as a seed and
    greedily adds compatible nodes.  For the typical window sizes used
    here (<=5 nodes), this is effectively exhaustive.
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
                # Both empty → identical (e.g. ping() called repeatedly)
                sim = 1.0
            else:
                sim = term_overlap(serialized[i], serialized[j])
            if sim >= threshold:
                similar[i][j] = True
                similar[j][i] = True

    # Greedy clique approximation: for each start node, grow a maximal
    # clique by adding nodes that are similar to every current member.
    # With window_size <= 5 this explores at most C(5,k) subsets, making
    # the approximation gap negligible in practice.
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
        min_repeats: Minimum cluster size to flag as thrashing (>= 2).
            Default 3.
        similarity_threshold: Minimum pairwise Jaccard similarity for
            argument token sets to be considered "similar" (0.0–1.0).
            Default 0.8.
        window_size: Sliding window size over per-tool calls (>= 1).
            Detection is recency-based: only calls within the same
            window are compared.  Default 5.

    Confidence:
        ``min(1.0, cluster_size / 5)`` — linearly scales from 0.2 (one
        repeat above minimum) to 1.0 (five or more repeats), reflecting
        increasing certainty that the pattern is genuine thrashing rather
        than coincidence.

    Severity mapping:
        - 3 repeats → MEDIUM  (agent is struggling)
        - 4 repeats → HIGH    (significant resource waste)
        - 5+ repeats → CRITICAL (agent is stuck in a loop)
    """

    def __init__(
        self,
        min_repeats: int = 3,
        similarity_threshold: float = 0.8,
        window_size: int = 5,
    ) -> None:
        if min_repeats < 2:
            raise ValueError(f"min_repeats must be >= 2, got {min_repeats}")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be between 0.0 and 1.0, got {similarity_threshold}"
            )
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
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
                        step_index=msg.step_index if msg.step_index is not None else _UNKNOWN_STEP,
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
                if len(cluster) >= self.min_repeats and len(cluster) > len(worst_cluster):
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
