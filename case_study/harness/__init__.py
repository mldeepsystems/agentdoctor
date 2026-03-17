from __future__ import annotations

from case_study.harness.evaluate import evaluate_trace, load_ground_truth, run_evaluation
from case_study.harness.metrics import (
    DetectorMetrics,
    clopper_pearson_ci,
    compute_aggregate_metrics,
    compute_cross_detector_interference,
    compute_per_detector_metrics,
    compute_severity_accuracy,
    compute_strict_metrics,
)
from case_study.harness.report_generator import generate_json_report, generate_markdown_report

__all__ = [
    "DetectorMetrics",
    "clopper_pearson_ci",
    "compute_aggregate_metrics",
    "compute_cross_detector_interference",
    "compute_per_detector_metrics",
    "compute_severity_accuracy",
    "compute_strict_metrics",
    "evaluate_trace",
    "generate_json_report",
    "generate_markdown_report",
    "load_ground_truth",
    "run_evaluation",
]
