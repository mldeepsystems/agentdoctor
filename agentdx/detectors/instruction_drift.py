"""Instruction Drift detector for agentdx."""

from __future__ import annotations

from agentdx._text_utils import extract_key_terms, simple_linear_regression, term_overlap
from agentdx.detectors.base import BaseDetector
from agentdx.models import DetectorResult, Role, Severity, Trace
from agentdx.taxonomy import Pathology


class InstructionDriftDetector(BaseDetector):
    """Detect when an agent gradually deviates from its system prompt.

    Tracks thematic alignment between the system prompt and assistant
    responses over time using term overlap and linear regression trend
    analysis.

    Args:
        min_messages: Minimum assistant messages required before the
            detector activates.  Defaults to ``8``.
        slope_threshold: Maximum negative slope before flagging drift.
            Defaults to ``-0.05``.
    """

    def __init__(
        self,
        min_messages: int = 8,
        slope_threshold: float = -0.05,
    ) -> None:
        if min_messages < 2:
            raise ValueError(f"min_messages must be >= 2, got {min_messages}")
        if slope_threshold > 0:
            raise ValueError(f"slope_threshold must be <= 0, got {slope_threshold}")
        self._min_messages = min_messages
        self._slope_threshold = slope_threshold

    @property
    def pathology(self) -> Pathology:
        return Pathology.INSTRUCTION_DRIFT

    def detect(self, trace: Trace) -> DetectorResult:
        """Analyse *trace* for instruction drift."""
        # Extract instruction fingerprint from system prompt
        system_prompt = trace.system_prompt
        if not system_prompt:
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.7,
                description="No system prompt found to measure drift against.",
            )

        instruction_terms = extract_key_terms(system_prompt)
        if not instruction_terms:
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.7,
                description="No meaningful terms in system prompt.",
            )

        # Compute overlap time series for assistant messages
        assistant_msgs = [m for m in trace.messages if m.role is Role.ASSISTANT]
        if len(assistant_msgs) < self._min_messages:
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.8,
                description="Not enough assistant messages to assess drift.",
            )

        overlaps: list[float] = []
        for msg in assistant_msgs:
            msg_terms = extract_key_terms(msg.content)
            overlaps.append(term_overlap(instruction_terms, msg_terms))

        # Linear regression on overlap time series
        xs = [float(i) for i in range(len(overlaps))]
        slope, _ = simple_linear_regression(xs, overlaps)

        if slope >= self._slope_threshold:
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.6,
                description=f"No significant drift detected (slope={slope:.4f}).",
            )

        # Detected — compute confidence and evidence
        confidence = min(1.0, abs(slope) / abs(self._slope_threshold))

        evidence: list[str] = [
            f"Overlap trend slope: {slope:.4f} (threshold: {self._slope_threshold})",
        ]
        evidence.append(f"Overlap series: {', '.join(f'{o:.2f}' for o in overlaps)}")

        severity = Severity.HIGH if slope < self._slope_threshold * 2 else Severity.MEDIUM

        return DetectorResult(
            pathology=self.pathology,
            detected=True,
            confidence=confidence,
            severity=severity,
            evidence=evidence,
            description=(
                f"Instruction drift detected: alignment slope {slope:.4f} "
                f"below threshold {self._slope_threshold}."
            ),
            recommendation=(
                "Consider periodic system prompt reinforcement or shorter conversation windows."
            ),
        )
