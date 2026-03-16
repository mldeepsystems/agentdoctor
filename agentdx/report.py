"""Diagnostic report for agentdx analysis results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agentdx.models import DetectorResult, Severity
from agentdx.taxonomy import PATHOLOGY_REGISTRY

_SEVERITY_ORDER: tuple[Severity, ...] = tuple(Severity)


@dataclass
class DiagnosticReport:
    """Aggregated results from running detectors against a trace.

    Provides multiple output formats: human-readable summary, JSON for
    CI/CD integration, Markdown for documentation, and dict for
    programmatic access.
    """

    trace_id: str | None
    results: list[DetectorResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def detected_pathologies(self) -> list[DetectorResult]:
        """Return only results where the pathology was detected."""
        return [r for r in self.results if r.detected]

    @property
    def highest_severity(self) -> Severity | None:
        """Return the most severe level among detected pathologies.

        Returns ``None`` when no pathologies were detected.
        """
        detected = self.detected_pathologies
        if not detected:
            return None
        return max(
            (r.severity for r in detected),
            key=lambda s: _SEVERITY_ORDER.index(s),
        )

    def summary(self) -> str:
        """Return a human-readable summary string."""
        lines: list[str] = [
            "agentdx Diagnostic Report",
            f"Trace: {self.trace_id or 'unknown'}",
        ]

        severity = self.highest_severity
        lines.append(f"Overall Severity: {severity.value.upper() if severity else 'NONE'}")
        lines.append("")

        detected = self.detected_pathologies
        total = len(self.results)
        lines.append(f"Detected Pathologies ({len(detected)}/{total}):")

        if detected:
            for r in detected:
                info = PATHOLOGY_REGISTRY.get(r.pathology)
                name = info.name if info else r.pathology.value
                lines.append(f"  [{r.severity.value.upper():<8s}] {name} — {r.description}")
        else:
            lines.append("  No pathologies detected.")

        not_detected = [r for r in self.results if not r.detected]
        if not_detected:
            names = []
            for r in not_detected:
                info = PATHOLOGY_REGISTRY.get(r.pathology)
                names.append(info.name if info else r.pathology.value)
            lines.append("")
            lines.append(f"No issues found for: {', '.join(names)}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a fully JSON-serializable dictionary."""
        return {
            "trace_id": self.trace_id,
            "overall_severity": (self.highest_severity.value if self.highest_severity else None),
            "results": [
                {
                    "pathology": r.pathology.value,
                    "detected": r.detected,
                    "confidence": r.confidence,
                    "severity": r.severity.value,
                    "evidence": r.evidence,
                    "description": r.description,
                    "recommendation": r.recommendation,
                }
                for r in self.results
            ],
            "metadata": self.metadata,
        }

    def to_json(self, path: str | None = None) -> str:
        """Serialize the report to a JSON string.

        When *path* is given, the JSON is also written to that file.
        """
        data = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
        return data

    def to_markdown(self, path: str | None = None) -> str:
        """Render the report as a Markdown string.

        When *path* is given, the Markdown is also written to that file.
        """
        lines: list[str] = [
            "# agentdx Diagnostic Report",
            "",
            f"**Trace:** {self.trace_id or 'unknown'}",
        ]

        severity = self.highest_severity
        lines.append(f"**Overall Severity:** {severity.value.upper() if severity else 'NONE'}")
        lines.append("")

        # Summary table
        lines.append("## Summary")
        lines.append("")
        lines.append("| Pathology | Detected | Severity | Confidence |")
        lines.append("|-----------|----------|----------|------------|")
        for r in self.results:
            info = PATHOLOGY_REGISTRY.get(r.pathology)
            name = info.name if info else r.pathology.value
            detected_str = "Yes" if r.detected else "No"
            sev_str = r.severity.value.upper() if r.detected else "-"
            conf_str = f"{r.confidence:.0%}"
            lines.append(f"| {name} | {detected_str} | {sev_str} | {conf_str} |")

        # Details for detected pathologies
        detected = self.detected_pathologies
        if detected:
            lines.append("")
            lines.append("## Detected Pathologies")
            for r in detected:
                info = PATHOLOGY_REGISTRY.get(r.pathology)
                name = info.name if info else r.pathology.value
                lines.append("")
                lines.append(f"### {name}")
                lines.append("")
                lines.append(f"**Severity:** {r.severity.value.upper()}")
                lines.append(f"**Confidence:** {r.confidence:.0%}")
                if r.description:
                    lines.append(f"**Description:** {r.description}")
                if r.recommendation:
                    lines.append(f"**Recommendation:** {r.recommendation}")
                if r.evidence:
                    lines.append("")
                    lines.append("**Evidence:**")
                    for e in r.evidence:
                        lines.append(f"- {e}")

        md = "\n".join(lines) + "\n"
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(md)
        return md
