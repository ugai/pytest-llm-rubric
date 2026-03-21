"""Calibration: verify an LLM backend can reliably judge rubric criteria.

Runs a small set of golden test pairs (known pass/fail) against the backend.
If the backend fails to match expected verdicts, it is considered unreliable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pytest_llm_rubric.golden_tests import GOLDEN_TESTS

if TYPE_CHECKING:
    from pytest_llm_rubric.plugin import JudgeLLM

# Accepts "PASS" / "FAIL" after optional non-word prefix (markdown, punctuation, whitespace).
# \b prevents partial matches like "PASSING" or "FAILED".
_VERDICT_RE = re.compile(r"\W*(PASS|FAIL)\b")

JUDGE_SYSTEM_PROMPT = """\
You are a rubric grader. You will be given a DOCUMENT and a CRITERION.
Determine whether the document expresses the criterion.
Respond with a single word: "PASS" or "FAIL".
Your response must be exactly one word. Do not explain."""


@dataclass
class CalibrationResult:
    passed: bool
    total: int
    correct: int
    details: list[dict]
    stopped_early: bool = False


def calibrate(llm: JudgeLLM, system_prompt: str | None = None) -> CalibrationResult:
    """Run golden tests against the LLM and return results.

    Stops early on the first incorrect answer since all tests must pass.

    Parameters:
        llm: Any object implementing the JudgeLLM protocol.
        system_prompt: Custom system prompt. Defaults to JUDGE_SYSTEM_PROMPT.
    """
    prompt = system_prompt if system_prompt is not None else JUDGE_SYSTEM_PROMPT
    details: list[dict] = []
    correct = 0
    stopped_early = False

    for test in GOLDEN_TESTS:
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (f"DOCUMENT:\n{test['document']}\n\nCRITERION:\n{test['criterion']}"),
            },
        ]
        raw_response = ""
        try:
            raw_response = llm.complete(messages, max_output_tokens=16).strip()
            normalized = raw_response.upper()
            m = _VERDICT_RE.match(normalized)
            verdict = m.group(1) if m else f"INVALID: {normalized[:50]}"
        except Exception as e:
            verdict = f"ERROR: {e}"

        is_correct = verdict == test["expected"]
        if is_correct:
            correct += 1

        details.append(
            {
                "criterion": test["criterion"],
                "expected": test["expected"],
                "actual": verdict,
                "correct": is_correct,
                "raw_response": raw_response,
            }
        )

        if not is_correct:
            stopped_early = len(details) < len(GOLDEN_TESTS)
            break

    return CalibrationResult(
        passed=correct == len(GOLDEN_TESTS),
        total=len(GOLDEN_TESTS),
        correct=correct,
        details=details,
        stopped_early=stopped_early,
    )
