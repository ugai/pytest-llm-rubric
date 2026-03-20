"""Tests for the calibration module."""

from __future__ import annotations

from pytest_rubric_grader.calibration import GOLDEN_TESTS, calibrate


class FakeLLMPass:
    """Returns the expected verdict by replaying golden test answers in order."""

    def __init__(self):
        self._index = 0

    def complete(self, messages: list[dict], max_tokens: int = 256) -> str:
        expected = GOLDEN_TESTS[self._index]["expected"]
        self._index += 1
        return expected


class FakeLLMFail:
    """Always returns PASS regardless of content."""

    def complete(self, messages: list[dict], max_tokens: int = 256) -> str:
        return "PASS"


class TestCalibrate:
    def test_perfect_judge(self):
        result = calibrate(FakeLLMPass())
        assert result.passed is True
        assert result.correct == result.total

    def test_bad_judge_fails_calibration(self):
        result = calibrate(FakeLLMFail())
        assert result.passed is False
        assert result.correct < result.total

    def test_result_has_details(self):
        result = calibrate(FakeLLMFail())
        assert len(result.details) == result.total
        for detail in result.details:
            assert "criterion" in detail
            assert "expected" in detail
            assert "actual" in detail
            assert "correct" in detail

    def test_golden_tests_are_balanced(self):
        """Ensure equal number of PASS and FAIL golden tests."""
        pass_count = sum(1 for t in GOLDEN_TESTS if t["expected"] == "PASS")
        fail_count = sum(1 for t in GOLDEN_TESTS if t["expected"] == "FAIL")
        assert pass_count == fail_count
