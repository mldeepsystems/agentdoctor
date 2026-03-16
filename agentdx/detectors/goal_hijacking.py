"""Goal Hijacking detector for agentdx."""

from __future__ import annotations

import re

from agentdx._text_utils import extract_key_terms, term_overlap
from agentdx.detectors.base import BaseDetector
from agentdx.models import DetectorResult, Role, Severity, Trace
from agentdx.taxonomy import Pathology

_UNKNOWN_STEP: int = -1

_INJECTION_PATTERNS = re.compile(
    r"ignore\s+(?:all\s+)?previous\s+instructions"
    r"|disregard\s+(?:all\s+)?(?:previous|prior|above)"
    r"|new\s+instructions?\s*:"
    r"|you\s+are\s+now"
    r"|override\s+(?:your\s+)?(?:instructions?|system\s+prompt)"
    r"|forget\s+(?:everything|all|your)"
    r"|act\s+as\s+(?:if\s+you\s+are|a)"
    r"|system\s*:\s*you\s+are"
    r"|from\s+now\s+on\s+you",
    re.IGNORECASE,
)


class GoalHijackingDetector(BaseDetector):
    """Detect when an agent's objective is altered by adversarial input.

    Scans user messages and tool results for prompt injection patterns,
    then checks whether the agent's behavior shifts after the suspected
    injection.

    Args:
        topic_shift_threshold: Minimum Jaccard similarity between
            consecutive user messages.  A drop below this indicates
            a drastic topic shift.  Defaults to ``0.1``.
    """

    def __init__(self, topic_shift_threshold: float = 0.1) -> None:
        if not 0.0 <= topic_shift_threshold <= 1.0:
            raise ValueError(
                f"topic_shift_threshold must be between 0.0 and 1.0, got {topic_shift_threshold}"
            )
        self._topic_shift_threshold = topic_shift_threshold

    @property
    def pathology(self) -> Pathology:
        return Pathology.GOAL_HIJACKING

    def detect(self, trace: Trace) -> DetectorResult:
        """Analyse *trace* for goal hijacking signals."""
        messages = trace.messages
        findings: list[tuple[int, str, str]] = []  # (step, source, detail)

        # 1. Scan for injection patterns in user messages
        for msg in messages:
            if msg.role is Role.USER:
                match = _INJECTION_PATTERNS.search(msg.content)
                if match:
                    step = msg.step_index if msg.step_index is not None else _UNKNOWN_STEP
                    findings.append(
                        (step, "user message", f'Injection pattern: "{match.group()}"')
                    )

        # 2. Scan for injection patterns in tool results
        for msg in messages:
            for tc in msg.tool_calls:
                if tc.result and _INJECTION_PATTERNS.search(tc.result):
                    step = msg.step_index if msg.step_index is not None else _UNKNOWN_STEP
                    match = _INJECTION_PATTERNS.search(tc.result)
                    findings.append(
                        (
                            step,
                            f"tool result ({tc.tool_name})",
                            f'Indirect injection: "{match.group()}"',
                        )
                    )

        # 3. Detect drastic topic shifts in user messages
        user_msgs = [m for m in messages if m.role is Role.USER]
        if len(user_msgs) >= 2:
            first_terms = extract_key_terms(user_msgs[0].content)
            for i in range(1, len(user_msgs)):
                current_terms = extract_key_terms(user_msgs[i].content)
                prev_terms = extract_key_terms(user_msgs[i - 1].content)
                if prev_terms and current_terms:
                    sim = term_overlap(prev_terms, current_terms)
                    # Also check against original goal
                    goal_sim = term_overlap(first_terms, current_terms) if first_terms else 1.0
                    if (
                        sim < self._topic_shift_threshold
                        and goal_sim < self._topic_shift_threshold
                    ):
                        step = (
                            user_msgs[i].step_index
                            if user_msgs[i].step_index is not None
                            else _UNKNOWN_STEP
                        )
                        findings.append(
                            (step, "topic shift", f"Drastic topic shift (similarity={sim:.2f})")
                        )

        if not findings:
            return DetectorResult(
                pathology=self.pathology,
                detected=False,
                confidence=0.8,
                description="No goal hijacking signals detected.",
            )

        # Check for behavioral shift after injection
        has_injection = any(
            s in ("user message", "tool result") or "tool result" in s for _, s, _ in findings
        )
        confidence = min(1.0, 0.6 + 0.15 * len(findings))
        if has_injection:
            confidence = min(1.0, confidence + 0.1)

        severity = Severity.CRITICAL if has_injection else Severity.HIGH

        evidence = [f"Step {step}: [{source}] {detail}" for step, source, detail in findings]

        return DetectorResult(
            pathology=self.pathology,
            detected=True,
            confidence=confidence,
            severity=severity,
            evidence=evidence,
            description=f"{len(findings)} goal hijacking signal(s) detected.",
            recommendation=(
                "Review flagged messages for prompt injection attempts. "
                "Consider input sanitization and instruction anchoring."
            ),
        )
