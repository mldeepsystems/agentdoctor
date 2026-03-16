"""Silent Degradation detector for AgentDoctor."""

from __future__ import annotations

import re

from agentdoctor._text_utils import simple_linear_regression
from agentdoctor.detectors.base import BaseDetector
from agentdoctor.models import DetectorResult, Role, Severity, Trace
from agentdoctor.taxonomy import Pathology

_WORD_RE = re.compile(r"[a-zA-Z]+")


def _quality_score(text: str) -> float:
    """Compute a composite quality proxy for *text*.

    Combines three signals:
    - Length score: ``min(1.0, len(text) / 500)``
    - Vocabulary richness: unique words / total words
    - Specificity: ratio of words longer than 5 characters

    Returns 0.0 for empty text.
    """
    if not text:
        return 0.0

    words = _WORD_RE.findall(text)
    if not words:
        return 0.0

    total = len(words)
    unique = len(set(w.lower() for w in words))

    length_score = min(1.0, len(text) / 500.0)
    richness = unique / total
    specificity = sum(1 for w in words if len(w) > 5) / total

    return (length_score + richness + specificity) / 3.0


class SilentDegradationDetector(BaseDetector):
    """Detect when output quality deteriorates without explicit errors.

    Uses quality proxies (response length, vocabulary richness,
    specificity) and linear regression to detect declining quality
    trends.

    Args:
        min_messages: Minimum assistant messages required before the
            detector activates.  Defaults to ``6``.
        slope_threshold: Maximum negative slope before flagging
            degradation.  Defaults to ``-0.03``.
    """

    def __init__(
        self,
        min_messages: int = 6,
        slope_threshold: float = -0.03,
    ) -> None:
        if min_messages < 2:
            raise ValueError(f"min_messages must be >= 2, got {min_messages}")
        if slope_threshold > 0:
            raise ValueError(f"slope_threshold must be <= 0, got {slope_threshold}")
        self._min_messages = min_messages
        self._slope_threshold = slope_threshold

    @property
    def pathology(self) -> Pathology:
        return Pathology.SILENT_DEGRADATION

    def detect(self, trace: Trace) -> DetectorResult:
        """Analyse *trace* for silent quality degradation."""
        assistant_msgs = [m for m in trace.messages if m.role is Role.ASSISTANT]

        if len(assistant_msgs) < self._min_messages:
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.8,
                description="Not enough assistant messages to assess quality trend.",
            )

        scores = [_quality_score(msg.content) for msg in assistant_msgs]

        # All zero scores means no content to analyse
        if all(s == 0.0 for s in scores):
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.5,
                description="No scorable content in assistant messages.",
            )

        xs = [float(i) for i in range(len(scores))]
        slope, _ = simple_linear_regression(xs, scores)

        if slope >= self._slope_threshold:
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.6,
                description=f"No significant quality decline (slope={slope:.4f}).",
            )

        confidence = min(1.0, abs(slope) / abs(self._slope_threshold))

        evidence: list[str] = [
            f"Quality trend slope: {slope:.4f} (threshold: {self._slope_threshold})",
            f"Quality scores: {', '.join(f'{s:.2f}' for s in scores)}",
        ]

        severity = Severity.HIGH if slope < self._slope_threshold * 2 else Severity.MEDIUM

        return DetectorResult(
            pathology=self.pathology,
            detected=True,
            confidence=confidence,
            severity=severity,
            evidence=evidence,
            description=(
                f"Silent degradation detected: quality slope {slope:.4f} "
                f"below threshold {self._slope_threshold}."
            ),
            recommendation=(
                "Monitor output quality metrics. Consider shorter conversation "
                "windows or periodic quality checks."
            ),
        )
