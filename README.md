# agentdx

**Open-source diagnostic SDK for detecting failure pathologies in AI agent systems.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/agentdx.svg)](https://pypi.org/project/agentdx/)

---

agentdx detects operational failure modes in AI agent systems — the kind of failures that observability tools miss because they happen at the reasoning level, not the infrastructure level. It analyses agent execution traces and produces structured diagnostic reports identifying specific pathologies.

## Why agentdx?

Existing agent observability tools (LangSmith, Arize, Datadog LLM) excel at **tracing**: they show you what happened. agentdx does **diagnosis**: it tells you what went wrong and why.

| Tool Type | What it answers | Examples |
|-----------|----------------|---------|
| Tracing / Observability | "What calls did the agent make?" | LangSmith, Arize, Datadog |
| **Diagnostics** | **"Why did the agent fail?"** | **agentdx** |

## The Seven Pathologies

agentdx detects seven operational failure modes, designed to align with the [OWASP Top 10 for Agentic Applications](https://genai.owasp.org/) and [UC Berkeley's MAST framework](https://arxiv.org/abs/2503.13657):

| Pathology | What it means |
|-----------|--------------|
| **Context Erosion** | Agent loses critical context over long conversations or multi-step tasks |
| **Tool Thrashing** | Agent repeatedly calls tools with ineffective or contradictory parameters |
| **Instruction Drift** | Agent gradually deviates from its original instructions or mandate |
| **Recovery Blindness** | Agent fails to detect or recover from errors in its own execution |
| **Hallucinated Tool Success** | Agent treats a failed tool call as successful and proceeds on false premises |
| **Goal Hijacking** | Agent's objective is altered by adversarial input or environmental manipulation |
| **Silent Degradation** | Agent's output quality deteriorates without triggering explicit errors |

## Quick Start

```bash
pip install agentdx
```

```python
from agentdx import Diagnoser, JSONParser

# Parse an agent execution trace (file path, dict, or message list)
parser = JSONParser()
trace = parser.parse("path/to/trace.json")

# Run all detectors
diagnoser = Diagnoser()
report = diagnoser.diagnose(trace)

# View results
print(report.summary())
report.to_json("diagnostic_report.json")
report.to_markdown("diagnostic_report.md")
```

The trace file is a JSON object with a `messages` array. Each message has a `role`, `content`, and optional `tool_calls`:

```json
{
  "trace_id": "my-trace",
  "messages": [
    {"role": "user", "content": "Find the best Python testing framework."},
    {
      "role": "assistant",
      "content": "Let me search for that.",
      "tool_calls": [
        {
          "tool_name": "web_search",
          "arguments": {"query": "best Python testing framework"},
          "result": "No relevant results found.",
          "success": true
        }
      ]
    }
  ]
}
```

## Architecture

agentdx uses a three-tier detection architecture:

```
Trace Input → Parser → Normalised Trace → Detectors → Diagnostic Report
                                              │
                                   ┌──────────┼──────────┐
                                   │          │          │
                              Rule-Based   ML Model   LLM-as-Judge
                              (< 10ms)    (< 100ms)   (async)
```

**Tier 1 — Rule-Based** (v0.1.0): Deterministic pattern matching on trace structure. Fast, interpretable, zero external dependencies.

**Tier 2 — ML Classifier** (planned): Trained classifiers for pathologies that require statistical pattern recognition.

**Tier 3 — LLM-as-Judge** (planned): Asynchronous LLM evaluation for complex, context-dependent pathology detection.

## Evaluation

We validated the Tier 1 detectors against 42 synthetic agent traces covering all 7 pathologies. The evaluation is a **development validation** — it confirms detectors fire on intended patterns, not that they generalise to production traces.

**What the evaluation shows:**
- Detectors correctly identify the target pathology in traces designed to exhibit it
- Cross-detector interference reveals that agent failures cluster (e.g., Recovery Blindness and Hallucinated Tool Success co-fire on failed tool calls)
- One known false negative (gh_03): a cooking assistant answering quantum physics questions — topic shifts without injection keywords are invisible to rule-based detection

**What it does not show:**
- Performance on production or adversarial traces (all traces are synthetic)
- Statistical significance (3–6 traces per detector; 95% CI for 3/3 recall is [0.29, 1.00])
- Comparison against baselines or alternative approaches

agentdx is best used as a **debugging aid during agent development**. See [`case_study/walkthrough.ipynb`](case_study/walkthrough.ipynb) for the full evaluation with dual metrics, confidence intervals, and per-detector analysis.

## Supported Frameworks

| Framework | Parser | Status |
|-----------|--------|--------|
| Raw JSON traces | `JSONParser` | v0.1.0 |
| LangChain / LangGraph | `LangChainParser` | Planned |
| CrewAI | `CrewAIParser` | Planned |
| AutoGen / AG2 | `AutoGenParser` | Planned |
| OpenAI Agents SDK | `OpenAIAgentParser` | Planned |

## Output Formats

- **JSON** — machine-readable diagnostic report
- **Markdown** — human-readable summary with severity ratings

## Research

agentdx is developed by [MLDeep Systems](https://mldeep.io), an AI and data consulting firm specialising in agent reliability. See [mldeep.io/research](https://mldeep.io/research) for related publications.

The failure pathology taxonomy draws on:

- OWASP GenAI Security Project. (2025). *Top 10 for Agentic Applications.* [genai.owasp.org](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
- Cemri, M., Pan, M., et al. (2025). *Why Do Multi-Agent LLM Systems Fail?* arXiv:2503.13657. [arxiv.org](https://arxiv.org/abs/2503.13657)

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use agentdx in your research, please cite:

```bibtex
@software{mldeepsystemsagentdx,
  author    = {Parimoo, Anmol},
  title     = {agentdx: An Open-Source Diagnostic SDK for AI Agent Reliability},
  year      = {2026},
  url       = {https://github.com/mldeepsystems/agentdx},
  license   = {MIT}
}
```

## License

MIT. See [LICENSE](LICENSE) for details.
