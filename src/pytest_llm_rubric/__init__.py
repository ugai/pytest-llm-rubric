"""pytest-llm-rubric: rubric-based LLM-as-judge testing for pytest."""

# Lazy imports to avoid circular dependency between plugin.py and preflight.py.
# preflight.py uses JudgeLLM from plugin.py under TYPE_CHECKING only;
# importing both at module level here could break if that changes.


def __getattr__(name: str):  # noqa: ANN001
    if name in ("PreflightResult", "Verdict", "preflight", "parse_verdict", "JUDGE_SYSTEM_PROMPT"):
        from pytest_llm_rubric.preflight import (
            JUDGE_SYSTEM_PROMPT,
            PreflightResult,
            Verdict,
            parse_verdict,
            preflight,
        )

        _exports = {
            "PreflightResult": PreflightResult,
            "Verdict": Verdict,
            "preflight": preflight,
            "parse_verdict": parse_verdict,
            "JUDGE_SYSTEM_PROMPT": JUDGE_SYSTEM_PROMPT,
        }
        return _exports[name]
    if name in ("JudgeLLM", "AnyLLMJudge"):
        from pytest_llm_rubric.plugin import AnyLLMJudge, JudgeLLM

        _exports = {"JudgeLLM": JudgeLLM, "AnyLLMJudge": AnyLLMJudge}
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnyLLMJudge",
    "JUDGE_SYSTEM_PROMPT",
    "JudgeLLM",
    "PreflightResult",
    "Verdict",
    "parse_verdict",
    "preflight",
]
