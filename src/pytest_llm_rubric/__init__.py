"""pytest-llm-rubric: rubric-based LLM-as-judge testing for pytest."""

# Lazy imports to avoid circular dependency between plugin.py and preflight.py.
# preflight.py uses JudgeLLM from plugin.py under TYPE_CHECKING only;
# importing both at module level here could break if that changes.


def __getattr__(name: str):  # noqa: ANN001
    if name in (
        "GoldenTest",
        "PreflightDetail",
        "PreflightResult",
        "Verdict",
        "preflight",
        "parse_verdict",
        "JUDGE_SYSTEM_PROMPT",
    ):
        from pytest_llm_rubric.golden_tests import GoldenTest
        from pytest_llm_rubric.preflight import (
            JUDGE_SYSTEM_PROMPT,
            PreflightDetail,
            PreflightResult,
            Verdict,
            parse_verdict,
            preflight,
        )

        _exports = {
            "GoldenTest": GoldenTest,
            "PreflightDetail": PreflightDetail,
            "PreflightResult": PreflightResult,
            "Verdict": Verdict,
            "preflight": preflight,
            "parse_verdict": parse_verdict,
            "JUDGE_SYSTEM_PROMPT": JUDGE_SYSTEM_PROMPT,
        }
        globals().update(_exports)
        return _exports[name]
    if name in ("JudgeLLM", "AnyLLMJudge", "JudgmentRecord", "register_judge", "get_shared_tmp"):
        from pytest_llm_rubric.plugin import (
            AnyLLMJudge,
            JudgeLLM,
            JudgmentRecord,
            get_shared_tmp,
            register_judge,
        )

        _exports = {
            "JudgeLLM": JudgeLLM,
            "AnyLLMJudge": AnyLLMJudge,
            "JudgmentRecord": JudgmentRecord,
            "get_shared_tmp": get_shared_tmp,
            "register_judge": register_judge,
        }
        globals().update(_exports)
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnyLLMJudge",
    "JUDGE_SYSTEM_PROMPT",
    "GoldenTest",
    "JudgeLLM",
    "JudgmentRecord",
    "PreflightDetail",
    "PreflightResult",
    "Verdict",
    "get_shared_tmp",
    "parse_verdict",
    "preflight",
    "register_judge",
]
