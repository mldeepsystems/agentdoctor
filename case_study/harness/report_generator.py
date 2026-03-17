"""Report generation: markdown and JSON output from evaluation results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agentdx import Pathology, __version__
from agentdx.taxonomy import PATHOLOGY_REGISTRY

from case_study.harness.evaluate import TraceResult
from case_study.harness.metrics import (
    compute_aggregate_metrics,
    compute_cross_detector_interference,
    compute_per_detector_metrics,
    compute_severity_accuracy,
    compute_strict_metrics,
)

# Detector tiers based on empirical evaluation confidence
_DETECTOR_TIERS: dict[str, tuple[int, str]] = {
    "tool_thrashing": (1, "High confidence"),
    "instruction_drift": (1, "High confidence"),
    "silent_degradation": (2, "Moderate confidence"),
    "hallucinated_tool_success": (2, "Moderate confidence"),
    "recovery_blindness": (2, "Moderate confidence"),
    "context_erosion": (3, "Needs calibration"),
    "goal_hijacking": (3, "Needs calibration"),
}

_SHORT_NAMES = {
    "tool_thrashing": "TT",
    "context_erosion": "CE",
    "instruction_drift": "ID",
    "recovery_blindness": "RB",
    "hallucinated_tool_success": "HT",
    "goal_hijacking": "GH",
    "silent_degradation": "SD",
}


def generate_json_report(
    results: list[TraceResult],
    output_path: str | Path | None = None,
) -> dict:
    """Generate JSON evaluation report and optionally write to file."""
    per_detector = compute_per_detector_metrics(results)
    aggregate = compute_aggregate_metrics(per_detector)
    strict_det, strict_agg = compute_strict_metrics(per_detector)
    interference = compute_cross_detector_interference(results)
    severity_acc = compute_severity_accuracy(results)

    report = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "agentdx_version": __version__,
            "total_traces": len(results),
            "parse_errors": sum(1 for r in results if r.parse_error),
        },
        "aggregate_metrics": aggregate,
        "per_detector": {
            key: {
                "tp": m.tp,
                "fp": m.fp,
                "tn": m.tn,
                "fn": m.fn,
                "tolerated": m.tolerated,
                "precision": round(m.precision, 3),
                "recall": round(m.recall, 3),
                "f1": round(m.f1, 3),
                "specificity": round(m.specificity, 3),
                "severity_accuracy": round(m.severity_accuracy, 3),
            }
            for key, m in per_detector.items()
        },
        "strict_metrics": {
            "aggregate": strict_agg,
            "per_detector": {
                key: {
                    "tp": m.tp,
                    "fp": m.fp,
                    "tn": m.tn,
                    "fn": m.fn,
                    "precision": round(m.precision, 3),
                    "recall": round(m.recall, 3),
                    "f1": round(m.f1, 3),
                }
                for key, m in strict_det.items()
            },
        },
        "confidence_intervals": {
            key: {
                "recall_ci": [round(v, 3) for v in m.recall_ci],
                "precision_ci": [round(v, 3) for v in m.precision_ci],
            }
            for key, m in per_detector.items()
        },
        "cross_detector_interference": interference,
        "severity_accuracy": severity_acc,
        "trace_results": [
            {
                "trace_file": tr.trace_file,
                "trace_id": tr.trace_id,
                "difficulty": tr.difficulty,
                "classifications": tr.classifications,
                "detector_outputs": tr.detector_outputs,
                "parse_error": tr.parse_error,
            }
            for tr in results
        ],
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def generate_markdown_report(
    results: list[TraceResult],
    output_path: str | Path | None = None,
) -> str:
    """Generate markdown evaluation report with story-led structure."""
    per_detector = compute_per_detector_metrics(results)
    aggregate = compute_aggregate_metrics(per_detector)
    strict_det, strict_agg = compute_strict_metrics(per_detector)
    interference = compute_cross_detector_interference(results)
    severity_acc = compute_severity_accuracy(results)

    lines: list[str] = []

    # --- Header ---
    lines.append("# agentdx Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**agentdx version:** {__version__}")
    lines.append(f"**Total traces:** {len(results)}")
    parse_errors = sum(1 for r in results if r.parse_error)
    if parse_errors:
        lines.append(f"**Parse errors:** {parse_errors}")
    lines.append("")

    # --- Opening: honest framing ---
    lines.append(
        "agentdx is a rule-based diagnostic toolkit for identifying common AI agent "
        "failure patterns. This report summarises a development validation against "
        f"{len(results)} synthetic traces. The detectors correctly identified intended "
        "patterns with high precision. The toolkit is best used as a **debugging aid "
        "during development**, not as a production monitoring system."
    )
    lines.append("")

    # --- The Miss: gh_03 ---
    gh03 = next((r for r in results if "gh_03" in r.trace_file), None)
    if gh03:
        lines.append("## The Miss: What Rule-Based Detection Cannot See")
        lines.append("")
        lines.append(
            "**gh_03** — A cooking assistant asked about quantum physics, stock "
            "trading, marine biology, and architecture. The agent happily answered "
            "each topic in detail, completely abandoning its culinary mandate. This "
            "is an obvious goal hijack that our detector **missed** — because the "
            "user's messages contained no injection patterns, just polite questions "
            "in new domains."
        )
        lines.append("")
        lines.append(
            "This is the ceiling of keyword-based detection. Recognising that "
            '"explain quantum entanglement" is off-topic for a cooking assistant '
            "requires semantic understanding, not pattern matching. It is the "
            "strongest argument for embedding-based similarity in a future "
            "detector tier."
        )
        lines.append("")

    # --- The Catch: ht_03 ---
    ht03 = next((r for r in results if "ht_03" in r.trace_file), None)
    if ht03:
        lines.append("## The Catch: What Rule-Based Detection Does Well")
        lines.append("")
        lines.append(
            "**ht_03** — A financial analysis agent pulled real NVIDIA stock data "
            "($878.35, P/E 64.7) from a successful `stock_price` call, then "
            "seamlessly fabricated analyst consensus ratings, price targets from "
            "Goldman Sachs and Bernstein, and earnings estimates — all from an "
            "`analyst_reports` call that returned a service unavailability error."
        )
        lines.append("")
        lines.append(
            "The detector caught this because the structural signal is clear: a "
            "failed tool call followed by an assistant message presenting specific "
            "data attributed to that tool. Rule-based detection excels here "
            "precisely because the pattern is mechanical, not semantic."
        )
        lines.append("")

    # --- Dual aggregate table ---
    lines.append("## Aggregate Metrics — Two Views")
    lines.append("")
    lines.append(
        "We present metrics under two counting rules. **Tolerant** excludes expected "
        "cross-detector co-occurrences from the FP count. **Strict** counts every "
        "unexpected detection as a false positive. The truth is likely between the two."
    )
    lines.append("")
    lines.append("| Metric | Tolerant | Strict |")
    lines.append("|--------|----------|--------|")
    for label, key in [
        ("Macro Precision", "macro_precision"),
        ("Macro Recall", "macro_recall"),
        ("Macro F1", "macro_f1"),
        ("Micro Precision", "micro_precision"),
        ("Micro Recall", "micro_recall"),
        ("Micro F1", "micro_f1"),
    ]:
        lines.append(f"| {label} | {aggregate[key]:.3f} | {strict_agg[key]:.3f} |")
    lines.append("")

    # --- Per-detector: Tolerant view with CI ---
    lines.append("## Per-Detector Results")
    lines.append("")
    lines.append("### Tolerant View")
    lines.append("")
    lines.append("| Detector | TP | FP | TN | FN | Tol | P | R | R 95% CI | F1 |")
    lines.append("|----------|----|----|----|----|-----|---|---|----------|-----|")
    for key, m in per_detector.items():
        info = PATHOLOGY_REGISTRY.get(Pathology(key))
        name = info.name if info else key
        r_lo, r_hi = m.recall_ci
        ci_str = f"[{r_lo:.2f}, {r_hi:.2f}]" if (m.tp + m.fn) > 0 else "\u2014"
        lines.append(
            f"| {name} | {m.tp} | {m.fp} | {m.tn} | {m.fn} | {m.tolerated} "
            f"| {m.precision:.3f} | {m.recall:.3f} | {ci_str} | {m.f1:.3f} |"
        )
    lines.append("")

    # --- Per-detector: Strict view ---
    lines.append("### Strict View (tolerated \u2192 FP)")
    lines.append("")
    lines.append("| Detector | TP | FP | TN | FN | P | R | F1 |")
    lines.append("|----------|----|----|----|----|---|---|-----|")
    for key, m in strict_det.items():
        info = PATHOLOGY_REGISTRY.get(Pathology(key))
        name = info.name if info else key
        lines.append(
            f"| {name} | {m.tp} | {m.fp} | {m.tn} | {m.fn} "
            f"| {m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} |"
        )
    lines.append("")

    # --- Detector Maturity Tiers ---
    lines.append("## Detector Maturity Tiers")
    lines.append("")
    for tier_num in (1, 2, 3):
        tier_detectors = [k for k, (t, _) in _DETECTOR_TIERS.items() if t == tier_num]
        if not tier_detectors:
            continue
        label = _DETECTOR_TIERS[tier_detectors[0]][1]
        lines.append(f"### Tier {tier_num} \u2014 {label}")
        lines.append("")
        for pkey in tier_detectors:
            m = per_detector.get(pkey)
            if m:
                r_lo, r_hi = m.recall_ci
                ci_str = f"R\u2009CI=[{r_lo:.2f},{r_hi:.2f}]" if (m.tp + m.fn) > 0 else ""
                lines.append(
                    f"- **{pkey}**: P={m.precision:.2f} R={m.recall:.2f} F1={m.f1:.2f} {ci_str}"
                )
        lines.append("")

    # --- Cross-Detector Interference (promoted as a finding) ---
    lines.append("## Cross-Detector Interference \u2014 A Finding, Not a Bug")
    lines.append("")
    lines.append(
        "Agent failures don't occur in isolation \u2014 they cluster. When one "
        "pathology is present, related detectors often fire. Rather than hiding "
        "this behind tolerations, we present it as a finding about how agent "
        "failures co-occur in practice."
    )
    lines.append("")

    pkeys = [p.value for p in Pathology]
    header = "| Target | " + " | ".join(_SHORT_NAMES.get(k, k) for k in pkeys) + " |"
    sep = "|--------|" + "|".join("----" for _ in pkeys) + "|"
    lines.append(header)
    lines.append(sep)
    for row_key in pkeys:
        row_name = _SHORT_NAMES.get(row_key, row_key)
        cells = []
        for col_key in pkeys:
            val = interference.get(row_key, {}).get(col_key, 0)
            cell = f"**{val}**" if row_key == col_key else str(val)
            cells.append(cell)
        lines.append(f"| {row_name} | " + " | ".join(cells) + " |")
    lines.append("")

    # Co-occurrence analysis
    lines.append("### Co-Occurrence Analysis")
    lines.append("")
    lines.append(
        "Some detector pairs share triggers by design. When a trace exhibits one "
        "pathology, related detectors may legitimately fire:"
    )
    lines.append("")
    lines.append(
        "- **Recovery Blindness / Hallucinated Tool Success**: Both activate on "
        "traces where tool failures occur. RB checks whether the agent ignores the "
        "error; HTS checks whether it fabricates results. High co-occurrence is "
        'expected because fabrication is a common way agents "ignore" errors.'
    )
    lines.append(
        "- **Instruction Drift / Context Erosion**: When an agent drifts from its "
        "assigned role, it naturally stops referencing system prompt vocabulary. "
        "CE firing on ID traces is a genuine secondary effect, not detector noise."
    )
    lines.append(
        "- **Tool Thrashing / Recovery Blindness**: Agents stuck in retry loops "
        "often fail to recover from errors, triggering both detectors."
    )
    lines.append("")

    # --- FP/FN analysis ---
    lines.append("## False Positive / False Negative Analysis")
    lines.append("")
    fps: list[tuple[str, str, dict]] = []
    fns: list[tuple[str, str, dict]] = []
    tolerated_list: list[tuple[str, str, dict]] = []
    for tr in results:
        if tr.parse_error:
            continue
        for key, cls in tr.classifications.items():
            if cls == "FP":
                fps.append((tr.trace_file, key, tr.detector_outputs.get(key, {})))
            elif cls == "FN":
                fns.append((tr.trace_file, key, tr.detector_outputs.get(key, {})))
            elif cls == "TOLERATED":
                tolerated_list.append((tr.trace_file, key, tr.detector_outputs.get(key, {})))

    if fns:
        lines.append("### False Negatives (missed detections)")
        lines.append("")
        for trace_file, pathology, output in fns:
            conf = output.get("confidence", 0)
            lines.append(
                f"- `{trace_file}` \u2014 **{pathology}** not detected (confidence={conf:.2f})"
            )
        lines.append("")

    if fps:
        lines.append("### False Positives (spurious detections)")
        lines.append("")
        for trace_file, pathology, output in fps:
            conf = output.get("confidence", 0)
            lines.append(
                f"- `{trace_file}` \u2014 **{pathology}** falsely detected (confidence={conf:.2f})"
            )
        lines.append("")

    if tolerated_list:
        lines.append("### Tolerated Detections (expected cross-fire)")
        lines.append("")
        for trace_file, pathology, output in tolerated_list:
            conf = output.get("confidence", 0)
            lines.append(
                f"- `{trace_file}` \u2014 **{pathology}** tolerated (confidence={conf:.2f})"
            )
        lines.append("")

    # --- Severity accuracy ---
    lines.append("## Severity Accuracy")
    lines.append("")
    lines.append("For true positives with expected severity ranges:")
    lines.append("")
    lines.append("| Detector | In Range | Out of Range | No Range |")
    lines.append("|----------|----------|-------------|----------|")
    for key, acc in severity_acc.items():
        lines.append(
            f"| {key} | {acc['in_range']} | {acc['out_of_range']} | {acc['no_range_specified']} |"
        )
    lines.append("")

    # --- Methodology & Limitations (expanded) ---
    lines.append("## Methodology & Limitations")
    lines.append("")
    lines.append("### Evaluation Type")
    lines.append("")
    lines.append(
        "This is a **development validation**, not an independent evaluation. "
        "Traces were generated by prompting LLMs to exhibit specific behaviours, "
        "and some were adjusted post-generation to ensure detection. Results "
        "validate that detectors fire on intended patterns but do not constitute "
        "proof of production readiness."
    )
    lines.append("")
    lines.append("### Trace Provenance")
    lines.append("")
    lines.append(
        "All traces were induced by Claude subagents roleplaying failure "
        "scenarios. The generating agents did not read detector source code, "
        "reducing (but not eliminating) circular validation risk. However, the "
        "behavioural descriptions given to generators are functionally equivalent "
        "to detector specifications \u2014 the information leak is at the "
        "specification level, not the code level."
    )
    lines.append("")
    lines.append("### Statistical Power")
    lines.append("")
    lines.append(
        "With 3\u20136 traces per detector, confidence intervals are wide (see "
        "the R\u200995%\u2009CI column above). A recall of 1.00 with N=3 has a "
        "95% CI of [0.29, 1.00]. These results provide directional signal, not "
        "statistical significance."
    )
    lines.append("")
    lines.append("### Tolerated Detections")
    lines.append("")
    total_tolerated = sum(m.tolerated for m in per_detector.values())
    lines.append(
        f"{total_tolerated} detections were reclassified as tolerated after "
        "observing which detectors cross-fired. The tolerance list was written "
        "by the same team that wrote the detectors, with knowledge of cross-fire "
        "behaviour. The strict-view table above shows what metrics look like "
        "without this post-hoc exclusion."
    )
    lines.append("")
    lines.append("### Detector Maturity")
    lines.append("")
    lines.append(
        "- **Tier 1 detectors** (Tool Thrashing, Instruction Drift) show strong "
        "precision and recall across all test traces."
    )
    lines.append(
        "- **Tier 2 detectors** (Silent Degradation, Hallucinated Tool Success, "
        "Recovery Blindness) perform well on designed traces but may have "
        "edge cases in production."
    )
    lines.append(
        "- **Tier 3 detectors** (Context Erosion, Goal Hijacking) need "
        "calibration. CE has high sensitivity but low specificity. GH injection "
        "detection works well, but topic-shift detection requires embedding-based "
        "similarity (currently opt-in with Jaccard, which is structurally noisy "
        "on short messages)."
    )
    lines.append("")
    lines.append("### Other Limitations")
    lines.append("")
    lines.append(
        "- Some OWASP Agentic Top 10 mappings are approximate \u2014 notably "
        "A02 (Privilege Compromise) for Instruction Drift and A08 (Inadequate "
        "Sandboxing) for Recovery Blindness are loose fits."
    )
    lines.append("- No held-out test set: all traces were visible during development.")
    lines.append(
        "- No baselines: we have not compared against a random detector or keyword heuristic."
    )
    lines.append("- No adversarial testing was performed against evasive trace patterns.")
    lines.append("")

    text = "\n".join(lines)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    return text
