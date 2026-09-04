"""Top-level convenience API for agentdx."""

from __future__ import annotations

from agentdx.detectors import BaseDetector
from agentdx.diagnoser import Diagnoser
from agentdx.models import Trace
from agentdx.parsers import JSONParser
from agentdx.report import DiagnosticReport


def diagnose(
    source: str | dict | list | Trace,
    detectors: list[BaseDetector] | None = None,
) -> DiagnosticReport:
    """Parse *source* if needed and run detectors against it.

    The one-call form of the usual two steps::

        report = agentdx.diagnose("trace.json")

        # equivalent to
        report = Diagnoser().diagnose(JSONParser().parse("trace.json"))

    ``source`` accepts the three forms :class:`~agentdx.parsers.JSONParser`
    already handles — a file path, a dict with a ``messages`` key, or a bare
    list of message dicts — and additionally an already-parsed :class:`Trace`,
    which is returned through unchanged rather than rejected.

    ``detectors`` is passed to :class:`~agentdx.diagnoser.Diagnoser`, so
    ``None`` runs every registered detector and a list runs only that subset.

    :class:`Diagnoser` and :class:`JSONParser` remain the API for anything
    beyond this: reusing a parser, inspecting the :class:`Trace` before
    diagnosing it, or running the same diagnoser over many traces.
    """
    trace = source if isinstance(source, Trace) else JSONParser().parse(source)
    return Diagnoser(detectors=detectors).diagnose(trace)
