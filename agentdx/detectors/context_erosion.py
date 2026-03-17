"""Context Erosion detector for agentdx."""

from __future__ import annotations

from agentdx._text_utils import anchor_recall, extract_key_terms
from agentdx.detectors.base import BaseDetector
from agentdx.models import DetectorResult, Message, Role, Severity, Trace
from agentdx.taxonomy import Pathology

_UNKNOWN_STEP: int = -1


class ContextErosionDetector(BaseDetector):
    """Detect when an agent loses critical context over a conversation.

    Uses a two-gate approach:

    1. **Early engagement gate** — the agent must demonstrate meaningful
       anchor-term recall in its early responses.  If the agent never
       echoed anchor vocabulary, a low score later is the baseline
       vocabulary gap, not erosion.
    2. **Decline gate** — the mean anchor recall must drop by at least
       *threshold* between the first third and final third of assistant
       messages.

    Args:
        threshold: Minimum decline in mean anchor recall (early − late)
            required to flag erosion.  Defaults to ``0.35``.
        min_messages: Minimum number of messages required before the
            detector activates.  Defaults to ``6``.
        min_early_recall: Minimum mean anchor recall in the first third
            of assistant messages.  Below this the agent never demonstrated
            anchor engagement, so no erosion can be diagnosed.
            Defaults to ``0.30``.
    """

    def __init__(
        self,
        threshold: float = 0.35,
        min_messages: int = 6,
        min_early_recall: float = 0.30,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")
        if min_messages < 1:
            raise ValueError(f"min_messages must be >= 1, got {min_messages}")
        if not 0.0 <= min_early_recall <= 1.0:
            raise ValueError(
                f"min_early_recall must be between 0.0 and 1.0, got {min_early_recall}"
            )
        self._threshold = threshold
        self._min_messages = min_messages
        self._min_early_recall = min_early_recall

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

        # Step 2: Compute recall for each assistant message
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
            overlap = anchor_recall(anchor_terms, msg_terms)
            step = msg.step_index if msg.step_index is not None else _UNKNOWN_STEP
            overlaps.append((step, overlap))

        # Step 3: Compute early and late means
        n = len(overlaps)
        first_third_end = max(1, n // 3)
        final_third_start = n - max(1, n // 3)

        early_values = [o for _, o in overlaps[:first_third_end]]
        late_values = [o for _, o in overlaps[final_third_start:]]

        early_mean = sum(early_values) / len(early_values)
        late_mean = sum(late_values) / len(late_values)

        # Gate 1: Agent must have demonstrated anchor engagement early
        if early_mean < self._min_early_recall:
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.7,
                description=(
                    f"Agent never demonstrated strong anchor engagement "
                    f"(early recall {early_mean:.2f} < {self._min_early_recall})."
                ),
            )

        # Gate 2: Must show significant decline
        drop = early_mean - late_mean
        if drop < self._threshold:
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.6,
                description=(
                    f"No significant context decline "
                    f"(drop {drop:.2f}, threshold {self._threshold})."
                ),
            )

        # Step 4: Detected — compute confidence and evidence
        confidence = min(1.0, drop / self._threshold) if self._threshold > 0 else 1.0

        # Find lost terms
        final_msg_terms: set[str] = set()
        for msg in assistant_msgs[final_third_start:]:
            final_msg_terms |= extract_key_terms(msg.content)
        lost_terms = anchor_terms - final_msg_terms

        evidence: list[str] = []
        evidence.append(
            f"Early recall mean: {early_mean:.2f}, "
            f"late recall mean: {late_mean:.2f}, "
            f"drop: {drop:.2f}"
        )
        if lost_terms:
            evidence.append(f"Lost anchor terms: {', '.join(sorted(lost_terms))}")
        for step, overlap in overlaps[final_third_start:]:
            if overlap < self._min_early_recall:
                evidence.append(
                    f"Step {step}: recall {overlap:.2f} (early mean was {early_mean:.2f})"
                )

        severity = Severity.HIGH if late_mean < 0.05 else Severity.MEDIUM

        return DetectorResult(
            pathology=self.pathology,
            detected=True,
            confidence=confidence,
            severity=severity,
            evidence=evidence,
            description=(
                f"Context erosion detected: anchor recall dropped from "
                f"{early_mean:.2f} to {late_mean:.2f} "
                f"(drop {drop:.2f}, threshold {self._threshold})."
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
