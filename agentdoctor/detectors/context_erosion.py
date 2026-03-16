"""Context Erosion detector for AgentDoctor."""

from __future__ import annotations

from agentdoctor._text_utils import extract_key_terms, term_overlap
from agentdoctor.detectors.base import BaseDetector
from agentdoctor.models import DetectorResult, Message, Role, Severity, Trace
from agentdoctor.taxonomy import Pathology

_UNKNOWN_STEP: int = -1


class ContextErosionDetector(BaseDetector):
    """Detect when an agent loses critical context over a conversation.

    Extracts anchor terms from the system prompt and early user messages,
    then measures whether those terms continue to appear in later assistant
    responses.  A significant drop in overlap in the final third of the
    conversation indicates context erosion.

    Args:
        threshold: Minimum overlap ratio in the final third to consider
            context preserved.  Defaults to ``0.3``.
        min_messages: Minimum number of messages required before the
            detector activates.  Defaults to ``6``.
    """

    def __init__(
        self,
        threshold: float = 0.3,
        min_messages: int = 6,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")
        if min_messages < 1:
            raise ValueError(f"min_messages must be >= 1, got {min_messages}")
        self._threshold = threshold
        self._min_messages = min_messages

    @property
    def pathology(self) -> Pathology:
        return Pathology.CONTEXT_EROSION

    def detect(self, trace: Trace) -> DetectorResult:
        """Analyse *trace* for context erosion."""
        messages = trace.messages
        if len(messages) < self._min_messages:
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.8,
                description="Trace too short for context erosion analysis.",
            )

        # Step 1: Extract anchor terms from system prompt + early user messages
        anchor_terms = self._extract_anchors(messages)
        if not anchor_terms:
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.7,
                description="No anchor terms found in system prompt or early messages.",
            )

        # Step 2: Compute overlap for each assistant message
        assistant_msgs = [m for m in messages if m.role is Role.ASSISTANT]
        if len(assistant_msgs) < 2:
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.7,
                description="Not enough assistant messages to assess erosion.",
            )

        overlaps: list[tuple[int, float]] = []
        for msg in assistant_msgs:
            msg_terms = extract_key_terms(msg.content)
            overlap = term_overlap(anchor_terms, msg_terms)
            step = msg.step_index if msg.step_index is not None else _UNKNOWN_STEP
            overlaps.append((step, overlap))

        # Step 3: Check final third
        final_third_start = len(overlaps) - max(1, len(overlaps) // 3)
        final_overlaps = overlaps[final_third_start:]
        min_overlap = min(o for _, o in final_overlaps)

        if min_overlap >= self._threshold:
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.6,
                description="Context maintained throughout conversation.",
            )

        # Step 4: Detected — compute confidence and evidence
        confidence = min(1.0, 1.0 - min_overlap)

        # Find lost terms
        final_msg_terms: set[str] = set()
        for msg in assistant_msgs[final_third_start:]:
            final_msg_terms |= extract_key_terms(msg.content)
        lost_terms = anchor_terms - final_msg_terms

        evidence: list[str] = []
        if lost_terms:
            evidence.append(f"Lost anchor terms: {', '.join(sorted(lost_terms))}")
        for step, overlap in final_overlaps:
            if overlap < self._threshold:
                evidence.append(
                    f"Step {step}: overlap {overlap:.2f} below threshold {self._threshold}"
                )

        severity = Severity.HIGH if min_overlap < 0.1 else Severity.MEDIUM

        return DetectorResult(
            pathology=self.pathology,
            detected=True,
            confidence=confidence,
            severity=severity,
            evidence=evidence,
            description=(
                f"Context erosion detected: overlap dropped to {min_overlap:.2f} "
                f"in final third (threshold {self._threshold})."
            ),
            recommendation=(
                "Consider injecting key context reminders into the conversation "
                "or reducing conversation length."
            ),
        )

    def _extract_anchors(self, messages: list[Message]) -> set[str]:
        """Extract anchor terms from system prompt and early user messages."""
        terms: set[str] = set()

        # System prompt(s)
        for msg in messages:
            if msg.role is Role.SYSTEM:
                terms |= extract_key_terms(msg.content)

        # First 25% of user messages
        user_msgs = [m for m in messages if m.role is Role.USER]
        early_count = max(1, len(user_msgs) // 4)
        for msg in user_msgs[:early_count]:
            terms |= extract_key_terms(msg.content)

        return terms
