import pytest

from agentdoctor._text_utils import (
    STOP_WORDS,
    contains_error_signals,
    extract_key_terms,
    simple_linear_regression,
    term_overlap,
)


class TestExtractKeyTerms:
    def test_basic_extraction(self):
        terms = extract_key_terms("The agent failed to recover from the error")
        assert "agent" in terms
        assert "failed" in terms
        assert "recover" in terms
        assert "error" in terms

    def test_filters_short_words(self):
        terms = extract_key_terms("I am a go to it")
        assert "am" not in terms  # < 3 chars
        assert "go" not in terms  # < 3 chars
        assert len(terms) == 0  # all words are short or stop words

    def test_filters_stop_words(self):
        terms = extract_key_terms("this should be removed because they are stop words")
        assert "removed" in terms
        assert "stop" in terms
        assert "words" in terms
        assert "this" not in terms
        assert "should" not in terms

    def test_empty_input(self):
        assert extract_key_terms("") == set()

    def test_non_alpha_splitting(self):
        terms = extract_key_terms("tool_call.result=success")
        assert "tool" in terms
        assert "call" in terms
        assert "result" in terms
        assert "success" in terms

    def test_three_char_technical_terms(self):
        terms = extract_key_terms("The API uses SQL and git")
        assert "api" in terms
        assert "sql" in terms
        assert "git" in terms

    def test_unicode_input(self):
        terms = extract_key_terms("café résumé naïve")
        # non-ascii stripped by [^a-z]+ split; remaining fragments kept if >= 3
        assert "caf" in terms
        assert "sum" in terms
        assert "na" not in terms

    def test_digit_only_input(self):
        assert extract_key_terms("12345 678") == set()

    def test_all_stop_words(self):
        assert extract_key_terms("this that the those them they") == set()


class TestTermOverlap:
    def test_identical_sets(self):
        s = {"agent", "error", "failed"}
        assert term_overlap(s, s) == 1.0

    def test_disjoint_sets(self):
        assert term_overlap({"alpha", "beta"}, {"gamma", "delta"}) == 0.0

    def test_partial_overlap(self):
        result = term_overlap({"agent", "error"}, {"agent", "success"})
        assert result == pytest.approx(1 / 3)  # 1 shared out of 3 union

    def test_empty_first_set(self):
        assert term_overlap(set(), {"agent"}) == 0.0

    def test_empty_second_set(self):
        assert term_overlap({"agent"}, set()) == 0.0

    def test_both_empty(self):
        assert term_overlap(set(), set()) == 0.0


class TestSimpleLinearRegression:
    def test_perfect_positive_slope(self):
        slope, intercept = simple_linear_regression([1, 2, 3], [2, 4, 6])
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(0.0)

    def test_flat_line(self):
        slope, intercept = simple_linear_regression([1, 2, 3], [5, 5, 5])
        assert slope == pytest.approx(0.0)
        assert intercept == pytest.approx(5.0)

    def test_constant_x(self):
        slope, intercept = simple_linear_regression([3, 3, 3], [1, 2, 3])
        assert slope == 0.0
        assert intercept == pytest.approx(2.0)

    def test_two_points(self):
        slope, intercept = simple_linear_regression([0, 1], [0, 1])
        assert slope == pytest.approx(1.0)
        assert intercept == pytest.approx(0.0)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            simple_linear_regression([1, 2], [1])

    def test_single_point_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            simple_linear_regression([1], [1])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            simple_linear_regression([], [])

    def test_negative_slope(self):
        slope, intercept = simple_linear_regression([1, 2, 3], [6, 4, 2])
        assert slope == pytest.approx(-2.0)
        assert intercept == pytest.approx(8.0)

    def test_float_inputs(self):
        slope, intercept = simple_linear_regression([0.5, 1.5, 2.5], [1.0, 2.0, 3.0])
        assert slope == pytest.approx(1.0)
        assert intercept == pytest.approx(0.5)


class TestContainsErrorSignals:
    @pytest.mark.parametrize(
        "text",
        [
            "An error occurred",
            "The request FAILED",
            "Traceback (most recent call last)",
            "ConnectionTimeout reached",
            "HTTP 404 Not Found",
            "Status code: 500",
            "Connection refused",
            "Access denied",
            "exception raised",
            "Request timed out",
            "fatal: not a git repository",
            "panic: runtime error",
            "Aborted (core dumped)",
            "HTTP 400 Bad Request",
            "HTTP 403 Forbidden",
            "Rate limited: 429",
            "Go stacktrace dump",
        ],
    )
    def test_detects_error_signals(self, text):
        assert contains_error_signals(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Everything is fine",
            "The operation completed successfully",
            "",
            "Hello world",
        ],
    )
    def test_no_false_positives(self, text):
        assert contains_error_signals(text) is False

    def test_known_substring_matches(self):
        # Documents known behavior: substring matching means these trigger.
        # This is intentional — detectors refine context around matches.
        assert contains_error_signals("error_handler class works") is True
        assert contains_error_signals("The failed_jobs table is empty") is True
