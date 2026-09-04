"""Tests for the top-level agentdx.diagnose() convenience."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

import agentdx
from agentdx import Diagnoser, JSONParser
from agentdx.detectors import ToolThrashingDetector
from agentdx.models import Trace
from agentdx.report import DiagnosticReport

_MESSAGES = [
    {"role": "user", "content": "Find the best Python testing framework."},
    {
        "role": "assistant",
        "content": "Let me search.",
        "tool_calls": [
            {
                "tool_name": "web_search",
                "arguments": {"query": "best python testing framework"},
                "result": "No relevant results found.",
                "success": True,
            }
        ],
    },
]

_TRACE_DICT = {"trace_id": "t-api", "messages": _MESSAGES}


class TestAcceptsEveryParserForm:
    def test_dict(self):
        report = agentdx.diagnose(_TRACE_DICT)
        assert isinstance(report, DiagnosticReport)
        assert report.trace_id == "t-api"

    def test_bare_message_list(self):
        report = agentdx.diagnose(_MESSAGES)
        assert isinstance(report, DiagnosticReport)

    def test_file_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trace.json")
            with open(path, "w") as fh:
                json.dump(_TRACE_DICT, fh)
            report = agentdx.diagnose(path)
        assert report.trace_id == "t-api"

    def test_already_parsed_trace_passes_through(self):
        # JSONParser rejects a Trace outright, so without the passthrough the
        # natural call after inspecting a trace would be a TypeError.
        trace = JSONParser().parse(_TRACE_DICT)
        assert isinstance(trace, Trace)
        report = agentdx.diagnose(trace)
        assert report.trace_id == "t-api"

    def test_unsupported_type_still_raises(self):
        # The passthrough must not swallow genuinely bad input.
        with pytest.raises(TypeError):
            agentdx.diagnose(42)


class TestEquivalentToTheTwoStepPath:
    def test_same_results_as_diagnoser_over_parser(self):
        # The whole claim of the convenience: it is the two-step call, not a
        # differently-behaving shortcut.
        one_call = agentdx.diagnose(_TRACE_DICT)
        two_step = Diagnoser().diagnose(JSONParser().parse(_TRACE_DICT))

        assert one_call.trace_id == two_step.trace_id
        assert [(r.pathology, r.detected) for r in one_call.results] == [
            (r.pathology, r.detected) for r in two_step.results
        ]

    def test_runs_every_detector_by_default(self):
        report = agentdx.diagnose(_TRACE_DICT)
        assert len(report.results) == len(Diagnoser().detectors)


class TestDetectorsArgument:
    def test_subset_is_respected(self):
        report = agentdx.diagnose(_TRACE_DICT, detectors=[ToolThrashingDetector()])
        assert len(report.results) == 1
        assert report.results[0].pathology is ToolThrashingDetector().pathology

    def test_none_means_all(self):
        assert len(agentdx.diagnose(_TRACE_DICT, detectors=None).results) == len(
            Diagnoser().detectors
        )

    def test_empty_list_runs_nothing(self):
        # An explicit empty list is not the same as None, and must not be
        # silently upgraded to "all detectors".
        assert agentdx.diagnose(_TRACE_DICT, detectors=[]).results == []


class TestExported:
    def test_is_exported_at_top_level(self):
        assert "diagnose" in agentdx.__all__
        assert agentdx.diagnose is not None

    def test_does_not_shadow_the_diagnoser_module(self):
        # `agentdx.diagnose` is a function; `agentdx.diagnoser` stays a module.
        from agentdx import diagnoser

        assert callable(agentdx.diagnose)
        assert diagnoser.Diagnoser is Diagnoser
