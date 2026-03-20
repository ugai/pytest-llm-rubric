"""Tests for the calibration module."""

from __future__ import annotations

from pytest_rubric_grader.calibration import GOLDEN_TESTS, JUDGE_SYSTEM_PROMPT, calibrate


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


class FakeLLMEmpty:
    """Returns empty string."""

    def complete(self, messages: list[dict], max_tokens: int = 256) -> str:
        return ""


class FakeLLMPartial:
    """Returns 'PASSING' — should be INVALID under strict parsing."""

    def complete(self, messages: list[dict], max_tokens: int = 256) -> str:
        return "PASSING"


class FakeLLMJunk:
    """Returns arbitrary junk."""

    def complete(self, messages: list[dict], max_tokens: int = 256) -> str:
        return "JUNK"


class FakeLLMCapture:
    """Captures the system prompt from the first call."""

    def __init__(self):
        self.captured_prompt: str | None = None
        self._index = 0

    def complete(self, messages: list[dict], max_tokens: int = 256) -> str:
        if self.captured_prompt is None:
            self.captured_prompt = messages[0]["content"]
        expected = GOLDEN_TESTS[self._index]["expected"]
        self._index += 1
        return expected


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

    def test_empty_response_is_invalid(self):
        result = calibrate(FakeLLMEmpty())
        assert result.passed is False
        assert all(d["actual"].startswith("INVALID") for d in result.details)

    def test_partial_match_is_invalid(self):
        """'PASSING' should not be accepted as PASS under strict parsing."""
        result = calibrate(FakeLLMPartial())
        assert result.passed is False
        assert all(d["actual"].startswith("INVALID") for d in result.details)

    def test_junk_response_is_invalid(self):
        result = calibrate(FakeLLMJunk())
        assert result.passed is False
        assert all(d["actual"].startswith("INVALID") for d in result.details)

    def test_custom_system_prompt(self):
        """calibrate() uses the custom system_prompt when provided."""
        llm = FakeLLMCapture()
        custom = "You are a custom grader."
        result = calibrate(llm, system_prompt=custom)
        assert result.passed is True
        assert llm.captured_prompt == custom

    def test_default_system_prompt(self):
        """calibrate() uses JUDGE_SYSTEM_PROMPT when system_prompt is None."""
        llm = FakeLLMCapture()
        result = calibrate(llm)
        assert result.passed is True
        assert llm.captured_prompt == JUDGE_SYSTEM_PROMPT
