"""Tests for the calibration module."""

from __future__ import annotations

from collections.abc import Callable

from pytest_llm_rubric.calibration import GOLDEN_TESTS, JUDGE_SYSTEM_PROMPT, calibrate


class FakeLLM:
    """Returns a fixed response for every call."""

    def __init__(self, response: str):
        self._response = response

    def complete(self, messages: list[dict], max_output_tokens: int = 256) -> str:
        return self._response


class ReplayLLM:
    """Replays golden test expected verdicts in order, with optional transform."""

    def __init__(self, transform: Callable[[str], str] = lambda v: v):
        self._index = 0
        self._transform = transform
        self.captured_prompts: list[str] = []

    def complete(self, messages: list[dict], max_output_tokens: int = 256) -> str:
        self.captured_prompts.append(messages[0]["content"])
        expected = GOLDEN_TESTS[self._index]["expected"]
        self._index += 1
        return self._transform(expected)


class TestCalibrate:
    def test_perfect_judge(self):
        result = calibrate(ReplayLLM())
        assert result.passed is True
        assert result.correct == result.total

    def test_bad_judge_fails_calibration(self):
        result = calibrate(FakeLLM("PASS"))
        assert result.passed is False
        assert result.correct < result.total

    def test_result_has_details(self):
        result = calibrate(FakeLLM("PASS"))
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
        result = calibrate(FakeLLM(""))
        assert result.passed is False
        assert all(d["actual"].startswith("INVALID") for d in result.details)

    def test_partial_match_is_invalid(self):
        """'PASSING' should not be accepted as PASS."""
        result = calibrate(FakeLLM("PASSING"))
        assert result.passed is False
        assert all(d["actual"].startswith("INVALID") for d in result.details)

    def test_junk_response_is_invalid(self):
        result = calibrate(FakeLLM("JUNK"))
        assert result.passed is False
        assert all(d["actual"].startswith("INVALID") for d in result.details)

    def test_decorated_response_accepted(self):
        """Verdicts with trailing punctuation or markdown decoration should be accepted."""
        templates = ["{}.", "**{}**", " {}\n\nBecause...", "{}."]
        counter = {"i": 0}

        def decorate(verdict: str) -> str:
            template = templates[counter["i"] % len(templates)]
            counter["i"] += 1
            return template.format(verdict)

        result = calibrate(ReplayLLM(transform=decorate))
        assert result.passed is True

    def test_failed_is_invalid(self):
        """'FAILED' should not be accepted as FAIL."""
        result = calibrate(FakeLLM("FAILED"))
        assert result.passed is False
        assert all(d["actual"].startswith("INVALID") for d in result.details)

    def test_custom_system_prompt(self):
        """calibrate() uses the custom system_prompt when provided."""
        llm = ReplayLLM()
        custom = "You are a custom judge."
        result = calibrate(llm, system_prompt=custom)
        assert result.passed is True
        assert llm.captured_prompts[0] == custom

    def test_default_system_prompt(self):
        """calibrate() uses JUDGE_SYSTEM_PROMPT when system_prompt is None."""
        llm = ReplayLLM()
        result = calibrate(llm)
        assert result.passed is True
        assert llm.captured_prompts[0] == JUDGE_SYSTEM_PROMPT
