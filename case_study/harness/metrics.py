"""Metrics computation: P/R/F1, cross-detector analysis, severity accuracy."""

from __future__ import annotations

import math
from dataclasses import dataclass

from agentdx import Pathology

from case_study.harness.evaluate import TraceResult

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


# ---------------------------------------------------------------------------
# Clopper-Pearson exact binomial CI (stdlib math only)
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta function."""
    _TINY = 1e-30
    _EPS = 1e-12
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _TINY:
        d = _TINY
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _TINY:
            d = _TINY
        c = 1.0 + aa / c
        if abs(c) < _TINY:
            c = _TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= _EPS:
            break
    return h


def _betainc(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _beta_quantile(p: float, a: float, b: float) -> float:
    """Inverse of I_x(a, b) = p via bisection."""
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _betainc(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson exact binomial confidence interval.

    Args:
        k: Number of successes.
        n: Number of trials.
        alpha: Significance level (default 0.05 for 95% CI).

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    if n == 0:
        return (0.0, 1.0)
    lower = 0.0 if k == 0 else _beta_quantile(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else _beta_quantile(1.0 - alpha / 2, k + 1, n - k)
    return (lower, upper)


# ---------------------------------------------------------------------------


@dataclass
class DetectorMetrics:
    """Per-detector confusion matrix and derived metrics."""

    pathology: str
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    tolerated: int = 0
    severity_in_range: int = 0
    severity_total: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def specificity(self) -> float:
        """TN / (TN + FP) — how well the detector avoids false alarms."""
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else 0.0

    @property
    def detection_rate(self) -> float:
        """Proportion of expected positives actually detected."""
        return self.recall

    @property
    def severity_accuracy(self) -> float:
        return self.severity_in_range / self.severity_total if self.severity_total > 0 else 0.0

    def to_strict(self) -> DetectorMetrics:
        """Return new instance where tolerated counts are moved to FP."""
        return DetectorMetrics(
            pathology=self.pathology,
            tp=self.tp,
            fp=self.fp + self.tolerated,
            tn=self.tn,
            fn=self.fn,
            tolerated=0,
            severity_in_range=self.severity_in_range,
            severity_total=self.severity_total,
        )

    @property
    def recall_ci(self) -> tuple[float, float]:
        """95% Clopper-Pearson CI for recall."""
        n = self.tp + self.fn
        return clopper_pearson_ci(self.tp, n) if n > 0 else (0.0, 0.0)

    @property
    def precision_ci(self) -> tuple[float, float]:
        """95% Clopper-Pearson CI for precision."""
        n = self.tp + self.fp
        return clopper_pearson_ci(self.tp, n) if n > 0 else (0.0, 0.0)


def compute_per_detector_metrics(results: list[TraceResult]) -> dict[str, DetectorMetrics]:
    """Compute per-detector metrics from evaluation results."""
    metrics: dict[str, DetectorMetrics] = {}
    for p in Pathology:
        metrics[p.value] = DetectorMetrics(pathology=p.value)

    for tr in results:
        if tr.parse_error:
            continue
        for key, classification in tr.classifications.items():
            if key not in metrics:
                continue
            m = metrics[key]
            if classification == "TP":
                m.tp += 1
                # Check severity range
                output = tr.detector_outputs.get(key, {})
                actual_sev = output.get("severity", "low")
                min_sev = output.get("expected_min_severity")
                max_sev = output.get("expected_max_severity")
                if min_sev and max_sev:
                    m.severity_total += 1
                    actual_ord = SEVERITY_ORDER.get(actual_sev, 0)
                    min_ord = SEVERITY_ORDER.get(min_sev, 0)
                    max_ord = SEVERITY_ORDER.get(max_sev, 3)
                    if min_ord <= actual_ord <= max_ord:
                        m.severity_in_range += 1
            elif classification == "FP":
                m.fp += 1
            elif classification == "TN":
                m.tn += 1
            elif classification == "FN":
                m.fn += 1
            elif classification == "TOLERATED":
                m.tolerated += 1

    return metrics


def compute_aggregate_metrics(
    per_detector: dict[str, DetectorMetrics],
) -> dict[str, float]:
    """Compute macro-averaged and micro-averaged metrics."""
    # Macro: average of per-detector metrics
    precisions = [m.precision for m in per_detector.values() if (m.tp + m.fp) > 0]
    recalls = [m.recall for m in per_detector.values() if (m.tp + m.fn) > 0]
    f1s = [m.f1 for m in per_detector.values() if (m.tp + m.fp + m.fn) > 0]

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    # Micro: sum counts then compute
    total_tp = sum(m.tp for m in per_detector.values())
    total_fp = sum(m.fp for m in per_detector.values())
    total_fn = sum(m.fn for m in per_detector.values())

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
    }


def compute_cross_detector_interference(
    results: list[TraceResult],
) -> dict[str, dict[str, int]]:
    """Compute cross-detector interference matrix.

    For each pathology P, count how many traces *designed for P* also trigger
    detector Q (where Q != P). High interference suggests detector overlap.

    Returns: {target_pathology: {firing_pathology: count}}
    """
    pathology_keys = [p.value for p in Pathology]
    matrix: dict[str, dict[str, int]] = {p: {q: 0 for q in pathology_keys} for p in pathology_keys}

    for tr in results:
        if tr.parse_error:
            continue
        # Determine which pathology this trace was designed for (primary expected)
        primary_pathologies = [k for k, v in tr.expected.items() if v.get("detected", False)]
        if not primary_pathologies:
            continue

        for target in primary_pathologies:
            for key, output in tr.detector_outputs.items():
                if output.get("detected", False):
                    matrix[target][key] += 1

    return matrix


def compute_severity_accuracy(results: list[TraceResult]) -> dict[str, dict]:
    """For TPs, check if detected severity falls within expected range."""
    accuracy: dict[str, dict] = {}
    for p in Pathology:
        accuracy[p.value] = {"in_range": 0, "out_of_range": 0, "no_range_specified": 0}

    for tr in results:
        if tr.parse_error:
            continue
        for key, output in tr.detector_outputs.items():
            if output.get("classification") != "TP":
                continue
            min_sev = output.get("expected_min_severity")
            max_sev = output.get("expected_max_severity")
            if not min_sev or not max_sev:
                accuracy[key]["no_range_specified"] += 1
                continue

            actual_sev = output.get("severity", "low")
            actual_ord = SEVERITY_ORDER.get(actual_sev, 0)
            min_ord = SEVERITY_ORDER.get(min_sev, 0)
            max_ord = SEVERITY_ORDER.get(max_sev, 3)
            if min_ord <= actual_ord <= max_ord:
                accuracy[key]["in_range"] += 1
            else:
                accuracy[key]["out_of_range"] += 1

    return accuracy


def compute_strict_metrics(
    per_detector: dict[str, DetectorMetrics],
) -> tuple[dict[str, DetectorMetrics], dict[str, float]]:
    """Compute metrics with tolerated detections counted as FP.

    Returns (strict_per_detector, strict_aggregate).
    """
    strict = {k: m.to_strict() for k, m in per_detector.items()}
    aggregate = compute_aggregate_metrics(strict)
    return strict, aggregate
