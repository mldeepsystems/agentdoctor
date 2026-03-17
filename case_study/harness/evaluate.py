"""Core evaluation logic: load ground truth, run diagnoser, classify results."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from agentdx import Diagnoser, JSONParser, Pathology, __version__

EXPECTED_VERSION = "0.1.0a3"

ALL_PATHOLOGY_KEYS = [p.value for p in Pathology]


@dataclass
class TraceResult:
    """Result of evaluating a single trace against ground truth."""

    trace_file: str
    trace_id: str
    difficulty: str
    description: str
    # Per-pathology classification: TP, FP, TN, FN
    classifications: dict[str, str] = field(default_factory=dict)
    # Raw detector outputs
    detector_outputs: dict[str, dict] = field(default_factory=dict)
    # Expected detections from ground truth
    expected: dict[str, dict] = field(default_factory=dict)
    parse_error: str | None = None


def load_ground_truth(manifest_path: str | Path) -> list[dict]:
    """Load ground truth manifest, filling defaults for unlisted pathologies.

    Unlisted pathologies default to detected=false.
    """
    path = Path(manifest_path)
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    traces = data.get("traces", [])
    for entry in traces:
        expected = entry.get("expected_detections", {})
        # Fill defaults: any pathology not listed is expected to NOT be detected
        for key in ALL_PATHOLOGY_KEYS:
            if key not in expected:
                expected[key] = {"detected": False}
            else:
                # Explicitly listed means expected to be detected
                expected[key]["detected"] = True
        entry["expected_detections"] = expected

    return traces


def evaluate_trace(
    trace_path: str | Path,
    expected_detections: dict[str, dict],
    diagnoser: Diagnoser | None = None,
    tolerated_detections: list[str] | None = None,
) -> dict[str, dict]:
    """Run diagnoser on a single trace, classify each pathology.

    Classifications:
    - ``TP`` — expected and detected
    - ``FP`` — not expected but detected (and not tolerated)
    - ``TN`` — not expected and not detected
    - ``FN`` — expected but not detected
    - ``TOLERATED`` — not expected but detected, and listed in *tolerated_detections*

    Returns dict mapping pathology_key -> {classification, confidence, severity,
    detected, expected_detected, evidence}.
    """
    if diagnoser is None:
        diagnoser = Diagnoser()
    tolerated = set(tolerated_detections or [])

    parser = JSONParser()
    trace = parser.parse(str(trace_path))
    report = diagnoser.diagnose(trace)

    results = {}
    for detector_result in report.results:
        key = detector_result.pathology.value
        expected = expected_detections.get(key, {"detected": False})
        expected_detected = expected.get("detected", False)
        actual_detected = detector_result.detected

        if actual_detected and expected_detected:
            classification = "TP"
        elif actual_detected and not expected_detected:
            classification = "TOLERATED" if key in tolerated else "FP"
        elif not actual_detected and expected_detected:
            classification = "FN"
        else:
            classification = "TN"

        results[key] = {
            "classification": classification,
            "detected": actual_detected,
            "expected_detected": expected_detected,
            "confidence": detector_result.confidence,
            "severity": detector_result.severity.value,
            "evidence": detector_result.evidence,
            "description": detector_result.description,
        }

        # Check severity range for TPs
        if classification == "TP":
            min_sev = expected.get("min_severity")
            max_sev = expected.get("max_severity")
            if min_sev or max_sev:
                results[key]["expected_min_severity"] = min_sev
                results[key]["expected_max_severity"] = max_sev

    return results


def run_evaluation(
    traces_dir: str | Path,
    manifest_path: str | Path,
) -> list[TraceResult]:
    """Run full evaluation suite over all traces in the manifest.

    Returns list of TraceResult objects.
    """
    if __version__ != EXPECTED_VERSION:
        warnings.warn(
            f"agentdx version {__version__} does not match expected {EXPECTED_VERSION}",
            stacklevel=2,
        )

    traces_dir = Path(traces_dir)
    ground_truth = load_ground_truth(manifest_path)
    diagnoser = Diagnoser()

    results = []
    for entry in ground_truth:
        trace_file = entry["trace_file"]
        trace_path = traces_dir / trace_file

        tr = TraceResult(
            trace_file=trace_file,
            trace_id=entry.get("trace_id", ""),
            difficulty=entry.get("difficulty", "unknown"),
            description=entry.get("description", ""),
            expected=entry["expected_detections"],
        )

        try:
            tolerated = entry.get("tolerated_detections", [])
            classifications = evaluate_trace(
                trace_path, entry["expected_detections"], diagnoser, tolerated
            )
            tr.detector_outputs = classifications
            tr.classifications = {k: v["classification"] for k, v in classifications.items()}
        except Exception as e:
            tr.parse_error = str(e)

        results.append(tr)

    return results
