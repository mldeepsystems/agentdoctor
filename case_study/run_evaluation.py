#!/usr/bin/env python3
"""Entry point for running the agentdx case study evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from case_study.harness.evaluate import run_evaluation  # noqa: E402
from case_study.harness.report_generator import generate_json_report, generate_markdown_report  # noqa: E402

CASE_STUDY_DIR = Path(__file__).resolve().parent
TRACES_DIR = CASE_STUDY_DIR / "traces"
MANIFEST_PATH = TRACES_DIR / "ground_truth.json"
RESULTS_DIR = CASE_STUDY_DIR / "results"


def main() -> int:
    print("Running agentdx evaluation...")
    print(f"  Traces: {TRACES_DIR}")
    print(f"  Ground truth: {MANIFEST_PATH}")
    print()

    if not MANIFEST_PATH.exists():
        print(f"ERROR: Ground truth manifest not found: {MANIFEST_PATH}")
        return 1

    results = run_evaluation(TRACES_DIR, MANIFEST_PATH)

    parse_errors = [r for r in results if r.parse_error]
    if parse_errors:
        print(f"WARNING: {len(parse_errors)} trace(s) failed to parse:")
        for r in parse_errors:
            print(f"  - {r.trace_file}: {r.parse_error}")
        print()

    # Generate reports
    json_path = RESULTS_DIR / "baseline_results.json"
    md_path = RESULTS_DIR / "evaluation_report.md"

    json_report = generate_json_report(results, json_path)
    print(f"JSON report written to: {json_path}")

    generate_markdown_report(results, md_path)
    print(f"Markdown report written to: {md_path}")

    # Print summary
    agg = json_report["aggregate_metrics"]
    print()
    print("=== Aggregate Metrics ===")
    print(
        f"  Macro P/R/F1: {agg['macro_precision']:.3f} / {agg['macro_recall']:.3f} / {agg['macro_f1']:.3f}"
    )
    print(
        f"  Micro P/R/F1: {agg['micro_precision']:.3f} / {agg['micro_recall']:.3f} / {agg['micro_f1']:.3f}"
    )
    print()

    # Per-detector summary
    print("=== Per-Detector Detection Rates ===")
    for key, det in json_report["per_detector"].items():
        status = "OK" if det["recall"] > 0 or (det["tp"] + det["fn"]) == 0 else "WARN"
        print(
            f"  [{status}] {key}: P={det['precision']:.2f} R={det['recall']:.2f} F1={det['f1']:.2f} (TP={det['tp']} FP={det['fp']} TN={det['tn']} FN={det['fn']})"
        )

    # Check for 0% recall/precision
    any_zero = False
    for key, det in json_report["per_detector"].items():
        if (det["tp"] + det["fn"]) > 0 and det["recall"] == 0:
            print(f"\nWARNING: {key} has 0% recall!")
            any_zero = True
        if (det["tp"] + det["fp"]) > 0 and det["precision"] == 0:
            print(f"\nWARNING: {key} has 0% precision!")
            any_zero = True

    if any_zero:
        print("\nSome detectors have 0% recall or precision. Review traces and ground truth.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
