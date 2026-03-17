"""Shared text-processing and math utilities for detectors."""

from __future__ import annotations

import re

STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "even",
        "few",
        "for",
        "from",
        "further",
        "get",
        "got",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "know",
        "let",
        "like",
        "make",
        "many",
        "may",
        "me",
        "might",
        "more",
        "most",
        "much",
        "must",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "per",
        "same",
        "she",
        "should",
        "so",
        "some",
        "still",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "upon",
        "us",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)

_WORD_SPLIT_RE = re.compile(r"[^a-z]+")

_ERROR_SIGNALS = re.compile(
    r"error|failed|failure|exception|traceback|timeout|timed\s*out"
    r"|stacktrace|fatal|panic|abort"
    r"|(?:HTTP[/ ]*[\d.]*\s*|(?:status|code)\s*:?\s*|:\s*)(?:400|403|404|429|500|502|503)\b"
    r"|refused|denied|unauthorized|forbidden|not\s*found",
    re.IGNORECASE,
)


def extract_key_terms(text: str) -> set[str]:
    """Extract meaningful terms from *text*.

    Lowercases, splits on non-alpha characters, drops tokens shorter than 3
    characters, and removes common English stop words.
    """
    tokens = _WORD_SPLIT_RE.split(text.lower())
    return {t for t in tokens if len(t) >= 3 and t not in STOP_WORDS}


def term_overlap(terms_a: set[str], terms_b: set[str]) -> float:
    """Return the Jaccard similarity between two term sets.

    Returns 0.0 when either set is empty, indicating no information rather
    than perfect agreement.
    """
    if not terms_a or not terms_b:
        return 0.0
    intersection = terms_a & terms_b
    union = terms_a | terms_b
    return len(intersection) / len(union)


def anchor_recall(anchor_terms: set[str], message_terms: set[str]) -> float:
    """Fraction of *anchor_terms* present in *message_terms*.

    Unlike :func:`term_overlap` (Jaccard), this is asymmetric: it measures
    how much of the anchor vocabulary the message still references, without
    penalising vocabulary expansion in the message.

    Returns ``0.0`` when *anchor_terms* is empty.
    """
    if not anchor_terms:
        return 0.0
    return len(anchor_terms & message_terms) / len(anchor_terms)


def simple_linear_regression(
    xs: list[float] | tuple[float, ...],
    ys: list[float] | tuple[float, ...],
) -> tuple[float, float]:
    """Return (slope, intercept) for a simple linear regression.

    Raises ``ValueError`` when *xs* and *ys* differ in length or contain
    fewer than two points.  Returns ``(0.0, mean_y)`` when x has zero
    variance (constant x), treating the best-fit line as horizontal at the
    mean of y.
    """
    n = len(xs)
    if n != len(ys):
        raise ValueError("xs and ys must have the same length")
    if n < 2:
        raise ValueError("Need at least two data points")

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    ss_xx = sum((x - mean_x) ** 2 for x in xs)
    if ss_xx == 0.0:
        return 0.0, mean_y

    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x
    return slope, intercept


def contains_error_signals(text: str) -> bool:
    """Return True if *text* contains common error-related keywords."""
    return bool(_ERROR_SIGNALS.search(text))
