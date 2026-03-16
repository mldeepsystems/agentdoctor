"""Hallucinated Tool Success detector for agentdx."""

from __future__ import annotations

import re

from agentdx._text_utils import contains_error_signals
from agentdx.detectors.base import BaseDetector
from agentdx.models import DetectorResult, Role, Severity, Trace
from agentdx.taxonomy import Pathology

_UNKNOWN_STEP: int = -1

_SUCCESS_INDICATORS = re.compile(
    r"the\s+results?\s+show"
    r"|according\s+to"
    r"|i\s+found"
    r"|here\s+are\s+the"
    r"|based\s+on\s+the\s+(?:results?|data|output|response)"
    r"|the\s+(?:search|query|api|tool)\s+returned"
    r"|successfully"
    r"|as\s+(?:shown|indicated|reported)",
    re.IGNORECASE,
)

_FAILURE_ACKNOWLEDGMENT = re.compile(
    r"fail|error|unable|couldn.t|unfortunately|sorry"
    r"|didn.t\s+work|no\s+results|timed?\s*out|not\s+found"
    r"|issue|problem|went\s+wrong",
    re.IGNORECASE,
)


class HallucinatedToolSuccessDetector(BaseDetector):
    """Detect when an agent fabricates results from a failed tool call.

    Identifies failed tool calls, then checks whether the following
    assistant message presents information as if the call had succeeded.
    """

    @property
    def pathology(self) -> Pathology:
        return Pathology.HALLUCINATED_TOOL_SUCCESS

    def detect(self, trace: Trace) -> DetectorResult:
        """Analyse *trace* for hallucinated tool success."""
        messages = trace.messages
        hallucinations: list[tuple[int, str, str]] = []  # (step, tool, phrase)

        for i, msg in enumerate(messages):
            for tc in msg.tool_calls:
                if not self._is_failed(tc):
                    continue

                # Find the next assistant message
                next_assistant = self._next_assistant(messages, i)
                if next_assistant is None:
                    continue

                content = next_assistant.content

                # Check for success indicators WITHOUT failure acknowledgment
                success_match = _SUCCESS_INDICATORS.search(content)
                acknowledges = _FAILURE_ACKNOWLEDGMENT.search(content)

                if success_match and not acknowledges:
                    step = msg.step_index if msg.step_index is not None else _UNKNOWN_STEP
                    hallucinations.append((step, tc.tool_name, success_match.group()))

        if not hallucinations:
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.8,
                description="No hallucinated tool successes detected.",
            )

        confidence = min(1.0, 0.7 + 0.1 * len(hallucinations))
        severity = Severity.CRITICAL if len(hallucinations) > 1 else Severity.HIGH

        evidence = [
            f'Step {step}: {tool} failed but assistant said "{phrase}"'
            for step, tool, phrase in hallucinations
        ]

        return DetectorResult(
            pathology=self.pathology,
            detected=True,
            confidence=confidence,
            severity=severity,
            evidence=evidence,
            description=(f"{len(hallucinations)} tool failure(s) treated as success."),
            recommendation=(
                "Ensure the agent validates tool call results before presenting "
                "them and acknowledges failures explicitly."
            ),
        )

    @staticmethod
    def _is_failed(tc) -> bool:
        """Return True if the tool call represents a failure."""
        if not tc.success:
            return True
        if tc.error_message:
            return True
        if tc.result and contains_error_signals(tc.result):
            return True
        return False

    @staticmethod
    def _next_assistant(messages, start_idx):
        """Find the next assistant message after *start_idx*."""
        for msg in messages[start_idx + 1 :]:
            if msg.role is Role.ASSISTANT:
                return msg
        return None
