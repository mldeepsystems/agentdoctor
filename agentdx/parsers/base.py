"""Base parser interface for agentdx trace parsers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from agentdx.models import Trace


class BaseParser(ABC):
    """Abstract base class for trace parsers.

    Subclasses must implement :meth:`parse` to convert a raw source into a
    :class:`Trace`.
    """

    @abstractmethod
    def parse(self, source: str | dict | list) -> Trace:
        """Parse *source* into a :class:`Trace`.

        Args:
            source: A file path (str), a dict with a ``messages`` key,
                or a bare list of message dicts.

        Returns:
            A fully populated :class:`Trace`.
        """

    @staticmethod
    def _load_file(path: str) -> dict | list:
        """Load and return JSON data from *path*.

        Raises:
            FileNotFoundError: If *path* does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(path, encoding="utf-8") as f:
            return json.load(f)
