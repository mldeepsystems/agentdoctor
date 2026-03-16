"""Recovery Blindness detector for AgentDoctor."""

from __future__ import annotations

import re

from agentdoctor._text_utils import contains_error_signals
from agentdoctor.detectors.base import BaseDetector
from agentdoctor.models import DetectorResult, Role, Severity, Trace
from agentdoctor.taxonomy import Pathology

_UNKNOWN_STEP: int = -1

_RECOVERY_PATTERN = re.compile(
    r"retry|retrying|retried"
    r"|try\s+again|trying\s+again"
    r"|alternative|different\s+approach"
    r"|sorry|apolog"
    r"|unfortunately|failed|couldn.t|unable"
    r"|error|issue|problem"
    r"|let\s+me\s+fix|fixing|correcting"
    r"|instead|fallback",
    re.IGNORECASE,
)


class RecoveryBlindnessDetector(BaseDetector):
    """Detect when an agent ignores errors and continues without recovery.

    Scans for error signals in tool calls, then checks whether the agent
    acknowledges the error and takes corrective action in subsequent
    messages.

    Args:
        lookahead_steps: Number of subsequent assistant messages to scan
            for recovery signals after an error.  Defaults to ``2``.
    """

    def __init__(self, lookahead_steps: int = 2) -> None:
        if lookahead_steps < 1:
            raise ValueError(f"lookahead_steps must be >= 1, got {lookahead_steps}")
        self._lookahead_steps = lookahead_steps

    @property
    def pathology(self) -> Pathology:
        return Pathology.RECOVERY_BLINDNESS

    def detect(self, trace: Trace) -> DetectorResult:
        """Analyse *trace* for unrecovered errors."""
        errors = self._find_errors(trace)
        if not errors:
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.9,
                description="No errors found in trace.",
            )

        messages = trace.messages
        unrecovered: list[tuple[int, str]] = []

        for error_idx, error_desc in errors:
            # Look ahead in messages after the error
            recovered = False
            assistant_seen = 0
            for msg in messages[error_idx + 1 :]:
                if msg.role is Role.ASSISTANT:
                    assistant_seen += 1
                    if _RECOVERY_PATTERN.search(msg.content):
                        recovered = True
                        break
                    if assistant_seen >= self._lookahead_steps:
                        break
            if not recovered:
                step = messages[error_idx].step_index
                if step is None:
                    step = _UNKNOWN_STEP
                unrecovered.append((step, error_desc))

        if not unrecovered:
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.8,
                description="All errors were properly addressed.",
            )

        confidence = min(1.0, len(unrecovered) / len(errors))

        evidence = [f"Step {step}: {desc} — no recovery detected" for step, desc in unrecovered]

        severity = Severity.HIGH if confidence >= 0.7 else Severity.MEDIUM

        return DetectorResult(
            pathology=self.pathology,
            detected=True,
            confidence=confidence,
            severity=severity,
            evidence=evidence,
            description=(f"{len(unrecovered)} of {len(errors)} errors went unrecovered."),
            recommendation=(
                "Ensure the agent acknowledges tool failures and takes "
                "corrective action (retry, alternative approach, or user notification)."
            ),
        )

    def _find_errors(self, trace: Trace) -> list[tuple[int, str]]:
        """Return ``(message_index, description)`` for each error found."""
        errors: list[tuple[int, str]] = []
        for i, msg in enumerate(trace.messages):
            for tc in msg.tool_calls:
                if not tc.success:
                    errors.append((i, f"{tc.tool_name} failed: {tc.error_message or 'unknown'}"))
                elif tc.error_message:
                    errors.append((i, f"{tc.tool_name} partial error: {tc.error_message}"))
                elif tc.result and contains_error_signals(tc.result):
                    errors.append((i, f"{tc.tool_name} returned error signals"))
        return errors
