"""Base detector interface for AgentDoctor pathology detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentdoctor.models import DetectorResult, Trace
from agentdoctor.taxonomy import Pathology


class BaseDetector(ABC):
    """Abstract base class for pathology detectors.

    Subclasses must implement :attr:`pathology` and :meth:`detect`.
    """

    @property
    @abstractmethod
    def pathology(self) -> Pathology:
        """The pathology this detector targets."""

    @abstractmethod
    def detect(self, trace: Trace) -> DetectorResult:
        """Analyse *trace* and return a :class:`DetectorResult`."""
