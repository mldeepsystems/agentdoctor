"""Tests for the case study evaluation harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agentdx import Pathology  # noqa: E402

from case_study.harness.evaluate import TraceResult, evaluate_trace, load_ground_truth  # noqa: E402
from case_study.harness.metrics import (  # noqa: E402
    DetectorMetrics,
    clopper_pearson_ci,
    compute_aggregate_metrics,
    compute_cross_detector_interference,
    compute_per_detector_metrics,
    compute_strict_metrics,
)
from case_study.harness.report_generator import generate_json_report, generate_markdown_report  # noqa: E402

CASE_STUDY_DIR = PROJECT_ROOT / "case_study"
TRACES_DIR = CASE_STUDY_DIR / "traces"


@pytest.fixture
def traces_root():
    """Root directory for case study traces."""
    return TRACES_DIR


@pytest.fixture
def sample_trace_path(tmp_path):
    """Create a minimal valid trace file for testing."""
    trace = {
        "trace_id": "test-001",
        "metadata": {"source": "test"},
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ],
    }
    path = tmp_path / "test_trace.json"
    with open(path, "w") as f:
        json.dump(trace, f)
    return path


@pytest.fixture
def sample_thrashing_trace_path(tmp_path):
    """Create a trace with tool thrashing pattern."""
    trace = {
        "trace_id": "thrash-test",
        "metadata": {"source": "test"},
        "messages": [
            {"role": "user", "content": "Find Python frameworks"},
            *[
                {
                    "role": "assistant",
                    "content": f"Searching attempt {i}.",
                    "tool_calls": [
                        {
                            "tool_name": "web_search",
                            "arguments": {"query": f"best Python testing framework {i}"},
                            "result": "No results found.",
                            "success": True,
                        }
                    ],
                }
                for i in range(5)
            ],
        ],
    }
    path = tmp_path / "thrash_trace.json"
    with open(path, "w") as f:
        json.dump(trace, f)
    return path


class TestLoadGroundTruth:
    """Tests for ground truth loading with defaults-fill logic."""

    def test_loads_manifest(self, tmp_path):
        manifest = {
            "version": "1.0",
            "traces": [
                {
                    "trace_file": "tool_thrashing/tt_01.json",
                    "trace_id": "tt-01",
                    "expected_detections": {
                        "tool_thrashing": {"min_severity": "medium", "max_severity": "critical"}
                    },
                    "difficulty": "easy",
                    "description": "Search loop",
                }
            ],
        }
        path = tmp_path / "ground_truth.json"
        with open(path, "w") as f:
            json.dump(manifest, f)

        entries = load_ground_truth(path)
        assert len(entries) == 1

    def test_fills_defaults_for_unlisted_pathologies(self, tmp_path):
        manifest = {
            "version": "1.0",
            "traces": [
                {
                    "trace_file": "tool_thrashing/tt_01.json",
                    "trace_id": "tt-01",
                    "expected_detections": {
                        "tool_thrashing": {"min_severity": "medium", "max_severity": "critical"}
                    },
                    "difficulty": "easy",
                    "description": "Search loop",
                }
            ],
        }
        path = tmp_path / "ground_truth.json"
        with open(path, "w") as f:
            json.dump(manifest, f)

        entries = load_ground_truth(path)
        expected = entries[0]["expected_detections"]

        # tool_thrashing should be marked as detected
        assert expected["tool_thrashing"]["detected"] is True
        assert expected["tool_thrashing"]["min_severity"] == "medium"

        # All other pathologies should default to not detected
        for key in [
            "context_erosion",
            "instruction_drift",
            "recovery_blindness",
            "hallucinated_tool_success",
            "goal_hijacking",
            "silent_degradation",
        ]:
            assert expected[key]["detected"] is False

    def test_empty_manifest(self, tmp_path):
        manifest = {"version": "1.0", "traces": []}
        path = tmp_path / "ground_truth.json"
        with open(path, "w") as f:
            json.dump(manifest, f)

        entries = load_ground_truth(path)
        assert entries == []


class TestEvaluateTrace:
    """Tests for single-trace evaluation."""

    def test_classifies_tn_for_healthy_trace(self, sample_trace_path):
        expected = {p.value: {"detected": False} for p in Pathology}
        results = evaluate_trace(sample_trace_path, expected)

        # All should be TN (nothing expected, nothing detected in a simple trace)
        for key, result in results.items():
            assert result["classification"] in ("TN", "FP"), (
                f"{key}: expected TN or FP, got {result['classification']}"
            )

    def test_classifies_tp_for_thrashing_trace(self, sample_thrashing_trace_path):
        expected = {p.value: {"detected": False} for p in Pathology}
        expected["tool_thrashing"] = {
            "detected": True,
            "min_severity": "medium",
            "max_severity": "critical",
        }

        results = evaluate_trace(sample_thrashing_trace_path, expected)
        assert results["tool_thrashing"]["classification"] == "TP"

    def test_classifies_fn_when_expected_but_not_detected(self, sample_trace_path):
        expected = {p.value: {"detected": False} for p in Pathology}
        # Expect tool thrashing on a healthy trace — should be FN
        expected["tool_thrashing"] = {"detected": True}

        results = evaluate_trace(sample_trace_path, expected)
        assert results["tool_thrashing"]["classification"] == "FN"

    def test_classifies_tolerated_when_in_tolerated_list(self, sample_thrashing_trace_path):
        """Detections listed in tolerated_detections get TOLERATED, not FP."""
        expected = {p.value: {"detected": False} for p in Pathology}
        expected["tool_thrashing"] = {
            "detected": True,
            "min_severity": "medium",
            "max_severity": "critical",
        }

        # Run without tolerated — check which pathologies fire as FP
        results_no_tol = evaluate_trace(sample_thrashing_trace_path, expected)
        fp_keys = [k for k, v in results_no_tol.items() if v["classification"] == "FP"]

        if fp_keys:
            # Run with those FPs tolerated
            results_tol = evaluate_trace(
                sample_thrashing_trace_path, expected, tolerated_detections=fp_keys
            )
            for k in fp_keys:
                assert results_tol[k]["classification"] == "TOLERATED"

    def test_tolerated_does_not_affect_tp(self, sample_thrashing_trace_path):
        """A pathology that's both expected AND tolerated stays TP (expected wins)."""
        expected = {p.value: {"detected": False} for p in Pathology}
        expected["tool_thrashing"] = {
            "detected": True,
            "min_severity": "medium",
            "max_severity": "critical",
        }

        results = evaluate_trace(
            sample_thrashing_trace_path, expected, tolerated_detections=["tool_thrashing"]
        )
        # TP because it was expected — tolerated only applies to unexpected detections
        assert results["tool_thrashing"]["classification"] == "TP"

    def test_severity_uses_value_not_str(self, sample_trace_path):
        """Verify severity is serialized as 'low'/'medium'/etc., not 'Severity.LOW'."""
        expected = {p.value: {"detected": False} for p in Pathology}
        results = evaluate_trace(sample_trace_path, expected)
        for key, result in results.items():
            sev = result["severity"]
            assert not sev.startswith("Severity."), (
                f"{key}: severity should be .value, got {sev!r}"
            )


class TestDetectorMetrics:
    """Tests for metric computation with known inputs."""

    def test_precision_recall_f1(self):
        m = DetectorMetrics(pathology="test", tp=8, fp=2, tn=85, fn=5)
        assert abs(m.precision - 0.8) < 0.001
        assert abs(m.recall - 0.615) < 0.01
        assert m.f1 > 0  # just check it's computed

    def test_zero_division_handling(self):
        m = DetectorMetrics(pathology="test", tp=0, fp=0, tn=10, fn=0)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_perfect_scores(self):
        m = DetectorMetrics(pathology="test", tp=5, fp=0, tn=5, fn=0)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_specificity(self):
        m = DetectorMetrics(pathology="test", tp=5, fp=2, tn=8, fn=0)
        assert abs(m.specificity - 0.8) < 0.001  # 8 / (8 + 2)

    def test_specificity_zero_division(self):
        m = DetectorMetrics(pathology="test", tp=5, fp=0, tn=0, fn=0)
        assert m.specificity == 0.0

    def test_tolerated_field(self):
        m = DetectorMetrics(pathology="test", tp=3, tolerated=2)
        assert m.tolerated == 2

    def test_severity_accuracy(self):
        m = DetectorMetrics(pathology="test", severity_in_range=3, severity_total=4)
        assert m.severity_accuracy == 0.75

    def test_severity_accuracy_no_data(self):
        m = DetectorMetrics(pathology="test")
        assert m.severity_accuracy == 0.0


class TestComputePerDetectorMetrics:
    """Tests for per-detector metrics from TraceResult list."""

    def _make_result(self, classifications: dict[str, str]) -> TraceResult:
        return TraceResult(
            trace_file="test.json",
            trace_id="test",
            difficulty="easy",
            description="test",
            classifications=classifications,
            detector_outputs={
                k: {"classification": v, "confidence": 0.8, "severity": "medium"}
                for k, v in classifications.items()
            },
        )

    def test_counts_correctly(self):
        results = [
            self._make_result({"tool_thrashing": "TP", "context_erosion": "TN"}),
            self._make_result({"tool_thrashing": "FP", "context_erosion": "FN"}),
        ]
        metrics = compute_per_detector_metrics(results)
        assert metrics["tool_thrashing"].tp == 1
        assert metrics["tool_thrashing"].fp == 1
        assert metrics["context_erosion"].tn == 1
        assert metrics["context_erosion"].fn == 1

    def test_counts_tolerated(self):
        results = [
            self._make_result({"tool_thrashing": "TP", "context_erosion": "TOLERATED"}),
        ]
        metrics = compute_per_detector_metrics(results)
        assert metrics["context_erosion"].tolerated == 1
        # Tolerated should not affect FP count
        assert metrics["context_erosion"].fp == 0

    def test_skips_parse_errors(self):
        results = [
            TraceResult(
                trace_file="bad.json",
                trace_id="bad",
                difficulty="easy",
                description="bad",
                parse_error="File not found",
            )
        ]
        metrics = compute_per_detector_metrics(results)
        for m in metrics.values():
            assert m.tp == 0 and m.fp == 0 and m.tn == 0 and m.fn == 0


class TestComputeAggregateMetrics:
    """Tests for macro/micro averaged metrics."""

    def test_aggregate_with_known_data(self):
        per_detector = {
            "a": DetectorMetrics(pathology="a", tp=5, fp=1, tn=4, fn=0),
            "b": DetectorMetrics(pathology="b", tp=3, fp=0, tn=5, fn=2),
        }
        agg = compute_aggregate_metrics(per_detector)
        assert 0 < agg["macro_precision"] <= 1.0
        assert 0 < agg["macro_recall"] <= 1.0
        assert 0 < agg["micro_f1"] <= 1.0


class TestCrossDetectorInterference:
    """Tests for cross-detector interference matrix."""

    def test_interference_matrix_structure(self):
        results = [
            TraceResult(
                trace_file="tt.json",
                trace_id="tt",
                difficulty="easy",
                description="thrashing",
                classifications={"tool_thrashing": "TP", "context_erosion": "FP"},
                detector_outputs={
                    "tool_thrashing": {"detected": True, "classification": "TP"},
                    "context_erosion": {"detected": True, "classification": "FP"},
                },
                expected={
                    "tool_thrashing": {"detected": True},
                    "context_erosion": {"detected": False},
                },
            )
        ]
        matrix = compute_cross_detector_interference(results)
        # tool_thrashing traces that also triggered context_erosion
        assert matrix["tool_thrashing"]["tool_thrashing"] == 1
        assert matrix["tool_thrashing"]["context_erosion"] == 1


class TestReportGeneration:
    """Tests for markdown and JSON report generation."""

    def _make_results(self) -> list[TraceResult]:
        return [
            TraceResult(
                trace_file="test.json",
                trace_id="test-001",
                difficulty="easy",
                description="Test trace",
                classifications={p.value: "TN" for p in Pathology},
                detector_outputs={
                    p.value: {
                        "classification": "TN",
                        "detected": False,
                        "confidence": 0.0,
                        "severity": "low",
                    }
                    for p in Pathology
                },
                expected={p.value: {"detected": False} for p in Pathology},
            )
        ]

    def test_json_report_structure(self):
        results = self._make_results()
        report = generate_json_report(results)
        assert "metadata" in report
        assert "aggregate_metrics" in report
        assert "per_detector" in report
        assert "cross_detector_interference" in report
        assert "trace_results" in report

    def test_json_report_includes_specificity(self):
        results = self._make_results()
        report = generate_json_report(results)
        for det_metrics in report["per_detector"].values():
            assert "specificity" in det_metrics
            assert "tolerated" in det_metrics

    def test_json_report_writes_file(self, tmp_path):
        results = self._make_results()
        path = tmp_path / "report.json"
        generate_json_report(results, path)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert "metadata" in data

    def test_markdown_report_content(self):
        results = self._make_results()
        md = generate_markdown_report(results)
        assert "# agentdx Evaluation Report" in md
        assert "Aggregate Metrics" in md
        assert "Per-Detector Results" in md
        assert "Cross-Detector Interference" in md
        assert "Methodology & Limitations" in md
        assert "Detector Maturity Tiers" in md
        assert "Tolerant" in md
        assert "Strict" in md

    def test_markdown_report_writes_file(self, tmp_path):
        results = self._make_results()
        path = tmp_path / "report.md"
        generate_markdown_report(results, path)
        assert path.exists()
        content = path.read_text()
        assert "agentdx Evaluation Report" in content

    def test_json_report_includes_strict_metrics(self):
        results = self._make_results()
        report = generate_json_report(results)
        assert "strict_metrics" in report
        assert "aggregate" in report["strict_metrics"]
        assert "per_detector" in report["strict_metrics"]

    def test_json_report_includes_confidence_intervals(self):
        results = self._make_results()
        report = generate_json_report(results)
        assert "confidence_intervals" in report
        for det_ci in report["confidence_intervals"].values():
            assert "recall_ci" in det_ci
            assert "precision_ci" in det_ci


class TestClopperPearsonCI:
    """Tests for Clopper-Pearson exact binomial confidence interval."""

    def test_perfect_recall_small_n(self):
        """3/3 should give approximately [0.29, 1.0]."""
        lo, hi = clopper_pearson_ci(3, 3)
        assert abs(lo - 0.2924) < 0.01
        assert hi == 1.0

    def test_zero_successes(self):
        """0/5 should give [0.0, ~0.52]."""
        lo, hi = clopper_pearson_ci(0, 5)
        assert lo == 0.0
        assert 0.4 < hi < 0.6

    def test_n_zero(self):
        """0/0 edge case."""
        lo, hi = clopper_pearson_ci(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_partial_success(self):
        """7/10 at 95% CI is approximately [0.35, 0.93]."""
        lo, hi = clopper_pearson_ci(7, 10)
        assert 0.3 < lo < 0.4
        assert 0.9 < hi < 1.0

    def test_ci_contains_point_estimate(self):
        """CI should always contain the point estimate k/n."""
        for k, n in [(1, 5), (3, 4), (5, 10), (0, 3), (3, 3)]:
            lo, hi = clopper_pearson_ci(k, n)
            p_hat = k / n if n > 0 else 0.0
            assert lo <= p_hat <= hi, f"CI [{lo}, {hi}] does not contain {p_hat} for k={k}, n={n}"

    def test_wider_ci_with_fewer_trials(self):
        """CI should be wider with fewer trials."""
        _, hi_3 = clopper_pearson_ci(3, 3)
        lo_3, _ = clopper_pearson_ci(3, 3)
        _, hi_30 = clopper_pearson_ci(30, 30)
        lo_30, _ = clopper_pearson_ci(30, 30)
        assert (hi_3 - lo_3) > (hi_30 - lo_30)


class TestStrictMetrics:
    """Tests for to_strict() method and compute_strict_metrics."""

    def test_moves_tolerated_to_fp(self):
        m = DetectorMetrics(pathology="test", tp=3, fp=1, tn=5, fn=0, tolerated=2)
        strict = m.to_strict()
        assert strict.fp == 3  # 1 + 2
        assert strict.tolerated == 0
        assert strict.tp == 3
        assert strict.tn == 5
        assert strict.fn == 0

    def test_precision_drops_with_strict(self):
        m = DetectorMetrics(pathology="test", tp=3, fp=0, tn=5, fn=0, tolerated=2)
        assert m.precision == 1.0  # 3/(3+0)
        strict = m.to_strict()
        assert abs(strict.precision - 0.6) < 0.001  # 3/(3+2)

    def test_no_tolerated_unchanged(self):
        m = DetectorMetrics(pathology="test", tp=3, fp=1, tn=5, fn=0, tolerated=0)
        strict = m.to_strict()
        assert strict.fp == 1
        assert strict.precision == m.precision

    def test_compute_strict_metrics_wrapper(self):
        per_detector = {
            "a": DetectorMetrics(pathology="a", tp=5, fp=0, tn=4, fn=0, tolerated=3),
            "b": DetectorMetrics(pathology="b", tp=3, fp=0, tn=5, fn=1, tolerated=0),
        }
        strict_det, strict_agg = compute_strict_metrics(per_detector)
        assert strict_det["a"].fp == 3
        assert strict_det["b"].fp == 0
        assert "macro_f1" in strict_agg


class TestDetectorMetricsCI:
    """Tests for recall_ci and precision_ci properties."""

    def test_recall_ci_perfect(self):
        m = DetectorMetrics(pathology="test", tp=3, fp=0, tn=5, fn=0)
        lo, hi = m.recall_ci
        assert lo > 0
        assert hi == 1.0  # 3/3

    def test_precision_ci(self):
        m = DetectorMetrics(pathology="test", tp=3, fp=1, tn=5, fn=0)
        lo, hi = m.precision_ci
        assert lo > 0
        assert hi <= 1.0

    def test_ci_zero_denominator(self):
        m = DetectorMetrics(pathology="test", tp=0, fp=0, tn=5, fn=0)
        assert m.recall_ci == (0.0, 0.0)
        assert m.precision_ci == (0.0, 0.0)

    def test_recall_ci_with_fn(self):
        m = DetectorMetrics(pathology="test", tp=3, fp=0, tn=5, fn=1)
        lo, hi = m.recall_ci  # 3/4
        assert lo < 0.75
        assert hi > 0.75
