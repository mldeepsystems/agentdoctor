# Case Study Verdict — Internal Review

**Date:** 2026-03-16
**Status:** Ready for reframing, not ready for publication as-is

---

## Current Numbers

```
Aggregate: Macro P/R/F1 = 1.000 / 0.964 / 0.980
           Micro P/R/F1 = 1.000 / 0.962 / 0.980

Per-Detector:
  Context Erosion:           P=1.00  R=1.00  F1=1.00  (TP=4, FP=0, TN=35, FN=0)
  Tool Thrashing:            P=1.00  R=1.00  F1=1.00  (TP=4, FP=0, TN=38, FN=0)
  Instruction Drift:         P=1.00  R=1.00  F1=1.00  (TP=3, FP=0, TN=39, FN=0)
  Recovery Blindness:        P=1.00  R=1.00  F1=1.00  (TP=4, FP=0, TN=33, FN=0)
  Hallucinated Tool Success: P=1.00  R=1.00  F1=1.00  (TP=4, FP=0, TN=35, FN=0)
  Goal Hijacking:            P=1.00  R=0.75  F1=0.86  (TP=3, FP=0, TN=38, FN=1)
  Silent Degradation:        P=1.00  R=1.00  F1=1.00  (TP=3, FP=0, TN=38, FN=0)

Severity accuracy: 100% across all detectors
Tolerated cross-fire: 12 detections (CE:3, HTS:3, RB:5, SD:1)
```

## Bottom Line

**Good enough for an alpha case study. Not good enough to lead with the numbers.**

The SDK engineering is strong (zero-dep, clean API, 374 tests, good architecture).
The 7-pathology taxonomy with OWASP/MAST mappings is the real intellectual contribution.
The evaluation validates that detectors fire on intended patterns — but that's all it validates.

---

## What a Skeptical User Would Flag

### 1. Near-perfect scores on synthetic data are a red flag, not a green flag

Macro F1 of 0.980 with a single FN across 42 traces and zero FPs looks like
testing regexes against strings written to match those regexes. The one FN (gh_03)
was deliberately included as a "known limitation" — meaning even the sole blemish
was pre-approved.

### 2. N=3-4 per detector is statistically meaningless

With 3 positive examples for Instruction Drift, the 95% Clopper-Pearson confidence
interval on recall at 3/3 is [0.44, 1.00]. Claiming R=1.00 with N=3 is not a
statistical statement. The report acknowledges this but still headlines 0.980 F1.

### 3. Tolerated detections are a post-hoc exclusion mechanism

12 detections reclassified after seeing which detectors cross-fired. Without
tolerations, Recovery Blindness would have 5 FPs, HTS 3, CE 3. The tolerance
list was written by the same people who wrote the detectors, with knowledge of
cross-fire behavior. A reviewer would ask whether this list was determined before
or after running the evaluation.

### 4. "Adjusted post-generation to ensure detection" = overfitting admission

The methodology says this explicitly. When you adjust test data to pass your tests,
you're measuring your ability to craft inputs, not detector quality. The tool
thrashing traces have literally identical queries. Real LLM agents vary more subtly.

### 5. Circular validation

The generating agents received behavioral descriptions functionally equivalent to
detector specifications. "Generate a trace where the agent retries search with
identical queries" describes exactly what the Jaccard-based detector looks for.
The information leak is at the behavioral specification level, not the code level.

### 6. Missing evaluation components

- No held-out test set (all 42 traces visible during development)
- No baselines (what does a random detector or keyword heuristic achieve?)
- No sensitivity analysis (how do results change with threshold tuning?)
- No inter-annotator agreement (single labeler)
- No adversarial negatives beyond 3 challenging traces

---

## What's Genuinely Strong

- **The taxonomy itself.** 7 failure modes mapped to OWASP Agentic Top 10 and
  UC Berkeley MAST. This is useful framing regardless of detector quality.

- **Individual trace examples.** ht_03 (mixing real stock data with fabricated
  analyst ratings) is a compelling demonstration. The RB/HTS co-occurrence
  pattern is a genuine finding about how agent failures cluster.

- **The SDK engineering.** Zero-dependency, stdlib-only, clean API, well-tested.
  For an alpha, this substantially exceeds typical open-source ML tool quality.

- **gh_03 as an honest limitation.** A cooking assistant answering quantum physics
  questions is an obvious goal hijack that the detector misses. This honestly
  shows what rule-based detection can and cannot do.

- **The cross-detector interference matrix.** Shows pathologies co-occur in
  practice. This is more interesting than the per-detector metrics.

---

## Recommended Reframing

### Lead with the taxonomy and stories, not the numbers

The strongest honest claim:

> "agentdx provides a rule-based diagnostic toolkit for identifying 7 common AI
> agent failure patterns. In development testing with 42 synthetic traces, the
> detectors correctly identified intended patterns with high precision. The
> toolkit is best used as a debugging aid during agent development, not as a
> production monitoring system."

### Concrete changes for the case study

1. **Drop aggregate F1 from any prominent position.** Don't put 0.980 in the
   README, blog title, or summary. Replace with per-detector examples showing
   real trace snippets and what was detected.

2. **Present two metric sets.** Show both strict (tolerations counted as FP) and
   tolerant (current) numbers side by side. Let the reader decide which is more
   relevant to their use case. Transparency builds more trust than clean numbers.

3. **Add confidence intervals.** Showing [0.44, 1.00] for 3/3 recall is more
   credible than showing 1.00. Wide intervals are honest — they say "we need
   more data" which is appropriate for alpha.

4. **Frame cross-detector interference as a finding.** "Agents don't fail in one
   way — they fail in correlated ways. Our toolkit reveals these co-occurrence
   patterns." This is more interesting than hiding cross-fire behind tolerations.

5. **Lead with gh_03 as the motivating limitation.** It honestly shows the ceiling
   of rule-based detection and motivates community contributions of embedding-based
   approaches.

6. **Restructure the walkthrough.** Instead of "here are 7 detectors with perfect
   scores," make it "here are 7 ways agents fail — watch us catch each one, and
   watch us miss one." The miss is more memorable than the catches.

### What NOT to claim

- Don't claim "98% F1" in marketing or README badges
- Don't claim "production-ready detection"
- Don't claim the detectors "detect" pathologies the way an ML model detects anomalies — they detect surface-level textual patterns that correlate with pathologies
- Don't claim the taxonomy is exhaustive (missing: cost explosion, data leakage, authority escalation)
- Don't claim independence between trace generation and evaluation

---

## Next Steps (Prioritized)

### For the case study reframe (do before publishing)

1. Rewrite walkthrough.ipynb: lead with examples, add strict/tolerant comparison,
   add confidence intervals, restructure around stories not scores
2. Update report_generator.py: add CI computation, dual-metric presentation
3. Update README case study section to match new framing

### For evaluation credibility (do when feasible)

4. Source 10-20 external traces (SWE-bench, WebArena, or anonymized production)
   and run as a held-out set
5. Add parameter sensitivity sweeps (threshold curves per detector)
6. Add a null-detector baseline to show detectors outperform random chance
7. Expand adversarial negatives (traces designed to fool specific detectors)

### For detector improvement (longer term)

8. Context Erosion: explore embedding-based similarity (would require optional
   dependency or external service)
9. Goal Hijacking: same — topic shift detection needs embeddings to be useful
10. Recovery Blindness: reduce sensitivity to incidental numeric codes in tool
    results (the error_signals regex fix helps but edge cases remain)
11. Add framework-specific parsers (LangChain, CrewAI) so users can run on
    real traces without manual JSON conversion

---

## Detector-Specific Notes

### Tier 1 — High confidence in the heuristic

- **Tool Thrashing**: Sliding-window Jaccard on argument tokens. Catches obvious
  repetition. Will miss subtle variations. Solid for debugging.
- **Instruction Drift**: Linear regression on instruction adherence. Works when
  drift is vocabulary-measurable. Good signal.

### Tier 2 — Useful but with known blind spots

- **Hallucinated Tool Success**: Regex for success-indicating phrases after failed
  tools. Will miss agents that fabricate without using indicator phrases.
- **Recovery Blindness**: Checks for error acknowledgment after tool failures.
  Good structural signal. Error regex now requires contextual anchors for HTTP codes.
- **Silent Degradation**: Quality score based on length + vocabulary richness.
  Catches obvious quality decline. Cannot distinguish intentionally concise from
  degraded.

### Tier 3 — Needs fundamental improvement

- **Context Erosion**: Two-gate (early engagement + decline) works on synthetic
  traces. Untested on real-world vocabulary patterns. The core question — "is
  the agent still on task?" — probably requires semantic similarity.
- **Goal Hijacking**: Injection regex scanner is solid. Topic shift detection is
  structurally inadequate with Jaccard on short messages. Opt-in default is the
  right call.
