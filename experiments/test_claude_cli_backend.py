"""Throwaway test: verify CLI backends work with judge()."""

import re
import subprocess

import pytest

from pytest_llm_rubric import AnyLLMJudge


class ClaudeCLIBackend(AnyLLMJudge):
    def complete(self, messages, max_output_tokens=256, response_format=None):
        prompt = messages[-1]["content"]
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            timeout=300,
        )
        return result.stdout.decode("utf-8")


class CopilotCLIBackend(AnyLLMJudge):
    def complete(self, messages, max_output_tokens=256, response_format=None):
        prompt = messages[-1]["content"]
        result = subprocess.run(
            ["copilot", "-p", prompt],
            capture_output=True,
            timeout=300,
        )
        output = result.stdout.decode("utf-8")
        # Strip trailing metadata (Total usage est, API time spent, etc.)
        output = re.split(r"\n\s*Total usage est:", output)[0]
        return output.strip()


@pytest.fixture(scope="session", params=["claude", "copilot"])
def judge_llm(request):
    if request.param == "claude":
        return ClaudeCLIBackend("claude-cli", "local")
    else:
        return CopilotCLIBackend("copilot-cli", "local")


def test_judge_pass(judge_llm):
    text = "The report is due by March 31st."
    assert judge_llm.judge(text, "The delivery deadline is mentioned.")


def test_judge_fail(judge_llm):
    text = "The weather is nice today."
    assert not judge_llm.judge(text, "The delivery deadline is mentioned.")
