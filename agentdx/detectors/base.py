"""Base detector interface for agentdx pathology detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentdx.models import DetectorResult, Trace
from agentdx.taxonomy import Pathology


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
