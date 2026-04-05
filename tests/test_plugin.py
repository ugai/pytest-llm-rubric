"""Tests for pytest-llm-rubric plugin."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from pytest_llm_rubric.plugin import (
    AnyLLMJudge,
    _make_judge,
    _parse_model,
)

# ---------------------------------------------------------------------------
# _parse_model tests
# ---------------------------------------------------------------------------


class TestParseModel:
    def test_anthropic_prefix(self):
        assert _parse_model("anthropic:claude-haiku-4-5") == ("anthropic", "claude-haiku-4-5")

    def test_ollama_prefix_with_tag(self):
        """ollama:qwen3.5:9b → provider='ollama', model='qwen3.5:9b'."""
        assert _parse_model("ollama:qwen3.5:9b") == ("ollama", "qwen3.5:9b")

    def test_openai_prefix(self):
        assert _parse_model("openai:gpt-4o") == ("openai", "gpt-4o")

    def test_case_insensitive(self):
        assert _parse_model("Anthropic:claude-haiku-4-5") == ("anthropic", "claude-haiku-4-5")

    def test_no_colon_raises(self):
        with pytest.raises(ValueError, match="Invalid model format"):
            _parse_model("gpt-4o")

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            _parse_model("mycompany:custom-model")

    def test_bare_ollama_tag_raises(self):
        """Bare 'qwen3.5:9b' has no known provider prefix → error."""
        with pytest.raises(ValueError, match="Unknown provider"):
            _parse_model("qwen3.5:9b")


# ---------------------------------------------------------------------------
# _make_judge tests
# ---------------------------------------------------------------------------


class TestMakeJudge:
    def test_ollama_returns_reason_when_package_missing(self, monkeypatch):
        import builtins
        import importlib

        real_import = builtins.__import__

        def _block_ollama(name, *args, **kwargs):
            if name == "ollama":
                raise ImportError("mocked: no ollama")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_ollama)
        if "ollama" in importlib.sys.modules:
            monkeypatch.delitem(importlib.sys.modules, "ollama")

        result = _make_judge("ollama", "qwen3.5:9b")
        assert isinstance(result, str)
        assert "not installed" in result

    def test_ollama_returns_judge_when_running(self):
        result = _make_judge("ollama", "")
        if isinstance(result, str):
            pytest.skip(f"Ollama is not running: {result}")
        assert isinstance(result, AnyLLMJudge)

    def test_ollama_returns_reason_when_unreachable(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:19999")
        result = _make_judge("ollama", "qwen3.5:9b")
        assert isinstance(result, str)
        assert "Could not connect" in result

    def test_ollama_returns_reason_when_model_not_found(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "some-real-model:latest"}]}
        with patch("httpx.get", return_value=mock_resp):
            result = _make_judge("ollama", "nonexistent-model-xyz")
        assert isinstance(result, str)
        assert "not found" in result

    def test_anthropic_returns_reason_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = _make_judge("anthropic", "claude-haiku-4-5")
        assert isinstance(result, str)
        assert "ANTHROPIC_API_KEY" in result

    def test_anthropic_returns_judge_with_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        judge = _make_judge("anthropic", "claude-haiku-4-5")
        assert isinstance(judge, AnyLLMJudge)

    def test_openai_returns_reason_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = _make_judge("openai", "gpt-4o")
        assert isinstance(result, str)
        assert "OPENAI_API_KEY" in result

    def test_openai_returns_judge_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        judge = _make_judge("openai", "gpt-4o")
        assert isinstance(judge, AnyLLMJudge)

    def test_passthrough_provider_returns_judge(self):
        judge = _make_judge("groq", "llama-3.3-70b")
        assert isinstance(judge, AnyLLMJudge)


# ---------------------------------------------------------------------------
# _run_preflight_check tests
# ---------------------------------------------------------------------------


class TestRunPreflightCheck:
    def test_import_error_returns_actionable_result(self):
        """Missing provider SDK → result dict with import_error key."""
        from pytest_llm_rubric.plugin import _run_preflight_check

        class _ImportErrorLLM:
            def complete(self, messages, max_output_tokens=256, response_format=None):
                raise ImportError(
                    "anthropic required packages are not installed. "
                    "Please install them with `pip install any-llm-sdk[anthropic]`."
                )

        data = _run_preflight_check(_ImportErrorLLM())
        assert data["passed"] is False
        assert "import_error" in data
        assert "anthropic" in data["import_error"]
        assert "pip install" in data["import_error"]
        assert data["summary"] == "FAILED (missing provider SDK)"


# ---------------------------------------------------------------------------
# _resolve_models tests
# ---------------------------------------------------------------------------


class TestResolveModels:
    def test_comma_separated_env_var(self, pytester, monkeypatch):
        """Comma-separated PYTEST_LLM_RUBRIC_MODELS tries each in order."""
        monkeypatch.setenv(
            "PYTEST_LLM_RUBRIC_MODELS",
            "anthropic:claude-haiku-4-5, openai:gpt-4o",
        )
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("PYTHONUTF8", "1")
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(errors=1)
        # Should only try the two models from env var, not the defaults.
        result.stdout.fnmatch_lines(
            [
                "*anthropic:claude-haiku-4-5: ANTHROPIC_API_KEY*",
                "*openai:gpt-4o: OPENAI_API_KEY*",
            ]
        )
        # Failure reasons should NOT include an ollama entry (not in the list).
        result.stdout.no_fnmatch_line("*ollama:*")


# ---------------------------------------------------------------------------
# AnyLLMJudge unit tests
# ---------------------------------------------------------------------------


class TestAnyLLMJudge:
    def test_complete_passes_max_tokens(self):
        """max_output_tokens is forwarded as max_tokens to any_llm.completion()."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="PASS"))]

        with patch("any_llm.completion", return_value=mock_response) as mock_comp:
            judge = AnyLLMJudge("test-model", "openai", api_key="k")
            result = judge.complete([{"role": "user", "content": "hi"}], max_output_tokens=100)

        assert result == "PASS"
        kwargs = mock_comp.call_args.kwargs
        assert kwargs["max_tokens"] == 100
        assert kwargs["model"] == "test-model"
        assert kwargs["provider"] == "openai"
        assert kwargs["stream"] is False

    def test_complete_forwards_api_base(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="FAIL"))]

        with patch("any_llm.completion", return_value=mock_response) as mock_comp:
            judge = AnyLLMJudge("m", "ollama", api_base="http://host:11434")
            judge.complete([{"role": "user", "content": "hi"}])

        assert mock_comp.call_args.kwargs["api_base"] == "http://host:11434"

    def test_complete_omits_none_api_base(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="OK"))]

        with patch("any_llm.completion", return_value=mock_response) as mock_comp:
            judge = AnyLLMJudge("m", "openai", api_key="k")
            judge.complete([{"role": "user", "content": "hi"}])

        assert "api_base" not in mock_comp.call_args.kwargs

    def test_complete_forwards_response_format(self):
        """response_format is forwarded to any_llm.completion()."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"result": "PASS"}'))]

        from pydantic import BaseModel

        class MyFormat(BaseModel):
            result: str

        with patch("any_llm.completion", return_value=mock_response) as mock_comp:
            judge = AnyLLMJudge("m", "openai", api_key="k")
            judge.complete([{"role": "user", "content": "hi"}], response_format=MyFormat)

        assert mock_comp.call_args.kwargs["response_format"] is MyFormat

    def test_complete_omits_none_response_format(self):
        """response_format is not passed when None."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="PASS"))]

        with patch("any_llm.completion", return_value=mock_response) as mock_comp:
            judge = AnyLLMJudge("m", "openai", api_key="k")
            judge.complete([{"role": "user", "content": "hi"}])

        assert "response_format" not in mock_comp.call_args.kwargs

    def test_complete_retries_on_empty_response(self):
        """Empty responses are retried up to _MAX_EMPTY_RETRIES times."""
        empty = MagicMock(choices=[MagicMock(message=MagicMock(content=""))])
        ok = MagicMock(choices=[MagicMock(message=MagicMock(content="PASS"))])

        with patch("any_llm.completion", side_effect=[empty, ok]) as mock_comp:
            judge = AnyLLMJudge("m", "ollama", api_base="http://localhost:11434")
            result = judge.complete([{"role": "user", "content": "hi"}])

        assert result == "PASS"
        assert mock_comp.call_count == 2

    def test_complete_returns_empty_after_all_retries_exhausted(self):
        """Returns empty string when all retries produce empty responses."""
        empty = MagicMock(choices=[MagicMock(message=MagicMock(content=""))])

        with patch("any_llm.completion", return_value=empty) as mock_comp:
            judge = AnyLLMJudge("m", "ollama", api_base="http://localhost:11434")
            result = judge.complete([{"role": "user", "content": "hi"}])

        assert result == ""
        assert mock_comp.call_count == 1 + AnyLLMJudge._MAX_EMPTY_RETRIES

    def test_complete_no_retry_on_nonempty_response(self):
        """Non-empty responses are returned immediately without retry."""
        ok = MagicMock(choices=[MagicMock(message=MagicMock(content="FAIL"))])

        with patch("any_llm.completion", return_value=ok) as mock_comp:
            judge = AnyLLMJudge("m", "openai", api_key="k")
            result = judge.complete([{"role": "user", "content": "hi"}])

        assert result == "FAIL"
        assert mock_comp.call_count == 1


class TestJudgeMethod:
    """Tests for AnyLLMJudge.judge() convenience method."""

    def test_judge_returns_true_on_pass(self):
        mock_response = MagicMock(choices=[MagicMock(message=MagicMock(content="PASS"))])
        with patch("any_llm.completion", return_value=mock_response):
            judge = AnyLLMJudge("m", "openai", api_key="k")
            assert judge.judge("The deadline is Friday.", "mentions a deadline") is True

    def test_judge_returns_false_on_fail(self):
        mock_response = MagicMock(choices=[MagicMock(message=MagicMock(content="FAIL"))])
        with patch("any_llm.completion", return_value=mock_response):
            judge = AnyLLMJudge("m", "openai", api_key="k")
            assert judge.judge("Hello world.", "mentions a deadline") is False

    def test_judge_raises_on_invalid_response(self):
        mock_response = MagicMock(choices=[MagicMock(message=MagicMock(content="JUNK"))])
        with patch("any_llm.completion", return_value=mock_response):
            judge = AnyLLMJudge("m", "openai", api_key="k")
            with pytest.raises(ValueError, match="Could not parse verdict"):
                judge.judge("doc", "criterion")

    def test_judge_accepts_json_verdict(self):
        mock_response = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"result": "PASS"}'))]
        )
        with patch("any_llm.completion", return_value=mock_response):
            judge = AnyLLMJudge("m", "openai", api_key="k")
            assert judge.judge("doc", "criterion") is True

    def test_judge_sends_system_prompt(self):
        from pytest_llm_rubric.preflight import JUDGE_SYSTEM_PROMPT

        mock_response = MagicMock(choices=[MagicMock(message=MagicMock(content="PASS"))])
        with patch("any_llm.completion", return_value=mock_response) as mock_comp:
            judge = AnyLLMJudge("m", "openai", api_key="k")
            judge.judge("doc", "criterion")

        messages = mock_comp.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == JUDGE_SYSTEM_PROMPT
        assert "DOCUMENT:\ndoc" in messages[1]["content"]
        assert "CRITERION:\ncriterion" in messages[1]["content"]


# ---------------------------------------------------------------------------
# Fixture tests (using pytester for isolation)
# ---------------------------------------------------------------------------

pytest_plugins = ["pytester"]

# Shared conftest snippet for pytester tests that need a fake judge_llm.
# Defined once here so that JudgeLLM Protocol changes only require one update.


def _preflight_conftest(
    *,
    passed: bool,
    correct: int,
    total: int,
    stopped_early: bool,
    details: list[dict[str, object]] | None = None,
) -> str:
    """Generate conftest that patches preflight and wires ``_preflight_or_skip``."""
    details_repr = repr(details or [])
    return f"""
import pytest
from unittest.mock import patch
from pytest_llm_rubric.plugin import AnyLLMJudge, _preflight_or_skip
from pytest_llm_rubric import register_judge
from pytest_llm_rubric.preflight import PreflightResult

@pytest.fixture(scope="session")
def judge_llm(request):
    judge = AnyLLMJudge("fake", "groq")
    register_judge(request.config, judge, model="groq:fake")
    fake_result = PreflightResult(
        passed={passed},
        correct={correct},
        total={total},
        stopped_early={stopped_early},
        details={details_repr},
    )
    with patch("pytest_llm_rubric.plugin.preflight", return_value=fake_result):
        return _preflight_or_skip(judge, config=request.config)
"""


_FAKE_JUDGE_CONFTEST = """
import pytest

class FakeLLM:
    def complete(self, messages, max_output_tokens=256, response_format=None):
        return "fake"

@pytest.fixture(scope="session")
def judge_llm():
    return FakeLLM()
"""


class TestJudgeLLMFixture:
    def test_fails_when_model_not_set(self, pytester, monkeypatch):
        """MODELS must be set — no silent default."""
        monkeypatch.delenv("PYTEST_LLM_RUBRIC_MODELS", raising=False)
        monkeypatch.setenv("PYTHONUTF8", "1")
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(["*PYTEST_LLM_RUBRIC_MODELS*not set*"])

    def test_falls_back_to_ini_when_env_unset(self, pytester, monkeypatch):
        """When env var is unset but llm_rubric_models is in ini, use the ini list."""
        monkeypatch.delenv("PYTEST_LLM_RUBRIC_MODELS", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("PYTHONUTF8", "1")
        pytester.makeini("""
            [pytest]
            llm_rubric_models =
                anthropic:claude-haiku-4-5
        """)
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        # Should try the ini list (not error about MODELS not set).
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(["*anthropic*ANTHROPIC_API_KEY*"])
        # The failure should mention the model, not "PYTEST_LLM_RUBRIC_MODELS is not set".
        result.stdout.no_fnmatch_line("*PYTEST_LLM_RUBRIC_MODELS*not set*")

    def test_fails_on_invalid_model_format(self, pytester, monkeypatch):
        """Invalid model string should produce a clear error, not a traceback."""
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODELS", "qwen3.5:9b")
        monkeypatch.setenv("PYTHONUTF8", "1")
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(["*Unknown provider*"])

    def test_override_fixture(self, pytester):
        pytester.makeconftest(_FAKE_JUDGE_CONFTEST)
        pytester.makepyfile("""
            def test_uses_fake(judge_llm):
                result = judge_llm.complete([{"role": "user", "content": "hi"}])
                assert result == "fake"
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(passed=1)

    def test_explicit_ollama_fails_when_unavailable(self, pytester, monkeypatch):
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODELS", "ollama:qwen3.5:9b")
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:19999")
        monkeypatch.setenv("PYTHONUTF8", "1")
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(errors=1)

    def test_explicit_openai_fails_without_key(self, pytester, monkeypatch):
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODELS", "openai:gpt-4o")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("PYTHONUTF8", "1")
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(errors=1)

    def test_explicit_anthropic_fails_without_key(self, pytester, monkeypatch):
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODELS", "anthropic:claude-haiku-4-5")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("PYTHONUTF8", "1")
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(errors=1)

    def test_auto_fails_with_reasons(self, pytester, monkeypatch):
        """Auto mode should list per-provider reasons when all backends fail."""
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODELS", "auto")
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:19999")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("PYTHONUTF8", "1")
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(
            [
                "*ollama:*",
                "*anthropic:*ANTHROPIC_API_KEY*",
                "*openai:*OPENAI_API_KEY*",
            ]
        )

    def test_multi_model_warns_on_cloud_fallback(self, pytester, monkeypatch):
        """Multi-model list falling through to cloud should emit a warning."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv(
            "PYTEST_LLM_RUBRIC_MODELS",
            "openai:gpt-fake,anthropic:claude-haiku-4-5",
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT", "1")
        monkeypatch.setenv("PYTHONUTF8", "1")
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v", "-W", "all")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*cloud provider*anthropic*third-party API*"])
        result.stdout.no_fnmatch_line("*LLM Rubric*")

    def test_ini_single_cloud_model_warns(self, pytester, monkeypatch):
        """A single cloud model from ini should still emit the cloud warning."""
        monkeypatch.delenv("PYTEST_LLM_RUBRIC_MODELS", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT", "1")
        monkeypatch.setenv("PYTHONUTF8", "1")
        pytester.makeini("""
            [pytest]
            llm_rubric_models =
                anthropic:claude-haiku-4-5
        """)
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v", "-W", "all")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*cloud provider*anthropic*third-party API*"])

    def test_preflight_timing_in_output(self, pytester, monkeypatch):
        """Preflight pass message should include elapsed time in terminal summary."""
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODELS", "groq:llama-3.3-70b")
        monkeypatch.setenv("PYTHONUTF8", "1")
        monkeypatch.delenv("PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT", raising=False)
        pytester.makeconftest(
            _preflight_conftest(
                passed=True,
                correct=12,
                total=12,
                stopped_early=False,
            )
        )
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(["*LLM Rubric*", "*preflight passed*12*in*s*"])

    def test_preflight_failure_timing_in_output(self, pytester, monkeypatch):
        """Preflight failure skip message should include elapsed time."""
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODELS", "groq:llama-3.3-70b")
        monkeypatch.setenv("PYTHONUTF8", "1")
        monkeypatch.delenv("PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT", raising=False)
        pytester.makeconftest(
            _preflight_conftest(
                passed=False,
                correct=4,
                total=12,
                stopped_early=True,
                details=[
                    {"criterion": "test", "expected": "PASS", "actual": "FAIL", "correct": False}
                ],
            )
        )
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v", "-rs")
        result.assert_outcomes(skipped=1)
        result.stdout.fnmatch_lines(["*failed preflight*in*s*"])

    def test_preflight_failure_includes_action(self, pytester, monkeypatch):
        """Preflight failure message should tell the user what to do next."""
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODELS", "groq:llama-3.3-70b")
        monkeypatch.setenv("PYTHONUTF8", "1")
        monkeypatch.delenv("PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT", raising=False)
        pytester.makeconftest(
            _preflight_conftest(
                passed=False,
                correct=4,
                total=12,
                stopped_early=True,
                details=[
                    {"criterion": "test", "expected": "PASS", "actual": "FAIL", "correct": False}
                ],
            )
        )
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v", "-rs")
        result.assert_outcomes(skipped=1)
        result.stdout.fnmatch_lines(["*PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT*"])

    def test_preflight_failure_shows_terminal_summary(self, pytester, monkeypatch):
        """Preflight failure should show model and status in the LLM Rubric summary."""
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODELS", "groq:llama-3.3-70b")
        monkeypatch.setenv("PYTHONUTF8", "1")
        monkeypatch.delenv("PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT", raising=False)
        pytester.makeconftest("""
import pytest
from unittest.mock import patch
from pytest_llm_rubric.plugin import AnyLLMJudge, _preflight_or_skip
from pytest_llm_rubric import register_judge
from pytest_llm_rubric.preflight import PreflightResult

@pytest.fixture(scope="session")
def judge_llm(request):
    judge = AnyLLMJudge("llama-3.3-70b", "groq")
    register_judge(request.config, judge, model="groq:llama-3.3-70b")
    fake_result = PreflightResult(
        passed=False,
        correct=4,
        total=12,
        stopped_early=True,
        details=[{"criterion": "test", "expected": "PASS", "actual": "FAIL", "correct": False}],
    )
    with patch("pytest_llm_rubric.plugin.preflight", return_value=fake_result):
        return _preflight_or_skip(judge, config=request.config)
""")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(skipped=1)
        result.stdout.fnmatch_lines(
            ["*LLM Rubric*", "*Model: groq:llama-3.3-70b*Preflight: FAILED*"]
        )

    def test_passthrough_provider_creates_judge(self, pytester, monkeypatch):
        """groq:model creates AnyLLMJudge via passthrough."""
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODELS", "groq:llama-3.3-70b-versatile")
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT", "1")
        monkeypatch.setenv("PYTHONUTF8", "1")
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(passed=1)

    def test_llm_rubric_marker_auto_applied(self, pytester):
        pytester.makeconftest(_FAKE_JUDGE_CONFTEST)
        pytester.makepyfile("""
            def test_with_judge(judge_llm):
                assert judge_llm is not None

            def test_without_judge():
                assert True
        """)
        result = pytester.runpytest_subprocess("-v", "-m", "llm_rubric")
        result.assert_outcomes(passed=1)

    def test_exclude_llm_rubric_marker(self, pytester):
        pytester.makeconftest(_FAKE_JUDGE_CONFTEST)
        pytester.makepyfile("""
            def test_with_judge(judge_llm):
                assert judge_llm is not None

            def test_without_judge():
                assert True
        """)
        result = pytester.runpytest_subprocess("-v", "-m", "not llm_rubric")
        result.assert_outcomes(passed=1, deselected=1)


class TestRubricSummary:
    """Tests for the LLM Rubric terminal summary."""

    def _judge_conftest(self, *, verdicts: list[str]) -> str:
        """Generate conftest where judge() returns canned PASS/FAIL verdicts."""
        return f"""
import pytest
from unittest.mock import patch
from pytest_llm_rubric.plugin import AnyLLMJudge, _preflight_or_skip
from pytest_llm_rubric import register_judge
from pytest_llm_rubric.preflight import PreflightResult

@pytest.fixture(scope="session")
def judge_llm(request):
    judge = AnyLLMJudge("fake-model", "groq")
    register_judge(request.config, judge, model="groq:fake-model")
    fake_result = PreflightResult(
        passed=True, correct=12, total=12, stopped_early=False, details=[],
    )
    with patch("pytest_llm_rubric.plugin.preflight", return_value=fake_result):
        judge = _preflight_or_skip(judge, config=request.config)
    verdicts = {verdicts!r}
    call_count = [0]
    def fake_complete(messages, max_output_tokens=256, response_format=None):
        idx = call_count[0]
        call_count[0] += 1
        return verdicts[idx] if idx < len(verdicts) else "PASS"
    judge.complete = fake_complete
    return judge
"""

    def test_summary_shows_model_and_counts(self, pytester, monkeypatch):
        """Summary should show model, preflight, and pass/fail counts."""
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODELS", "groq:fake-model")
        monkeypatch.setenv("PYTHONUTF8", "1")
        monkeypatch.delenv("PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT", raising=False)
        pytester.makeconftest(self._judge_conftest(verdicts=["PASS", "PASS"]))
        pytester.makepyfile("""
            def test_one(judge_llm):
                assert judge_llm.judge("doc", "criterion A")

            def test_two(judge_llm):
                assert judge_llm.judge("doc", "criterion B")
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(
            [
                "*LLM Rubric*",
                "*Model: groq:fake-model*Preflight:*",
                "*2 passed, 0 failed*",
            ]
        )

    def test_summary_shows_failures(self, pytester, monkeypatch):
        """Summary should list failed judgments with criterion and node ID."""
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODELS", "groq:fake-model")
        monkeypatch.setenv("PYTHONUTF8", "1")
        monkeypatch.delenv("PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT", raising=False)
        pytester.makeconftest(self._judge_conftest(verdicts=["PASS", "FAIL"]))
        pytester.makepyfile("""
            def test_pass(judge_llm):
                assert judge_llm.judge("doc", "good criterion")

            def test_fail(judge_llm):
                result = judge_llm.judge("doc", "bad criterion")
                assert result, "Judgment failed"
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(passed=1, failed=1)
        result.stdout.fnmatch_lines(
            [
                "*1 passed, 1 failed*",
                "*FAIL*test_fail*",
                '*criterion: "bad criterion"*',
            ]
        )

    def test_no_summary_without_rubric_tests(self, pytester, monkeypatch):
        """No summary section when no rubric tests ran."""
        monkeypatch.setenv("PYTHONUTF8", "1")
        monkeypatch.delenv("PYTEST_LLM_RUBRIC_MODELS", raising=False)
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_plain():
                assert True
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(passed=1)
        result.stdout.no_fnmatch_line("*LLM Rubric*")

    def test_multiple_judgments_in_one_test(self, pytester, monkeypatch):
        """Multiple judge() calls in a single test should all be recorded."""
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODELS", "groq:fake-model")
        monkeypatch.setenv("PYTHONUTF8", "1")
        monkeypatch.delenv("PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT", raising=False)
        pytester.makeconftest(self._judge_conftest(verdicts=["PASS", "FAIL"]))
        pytester.makepyfile("""
            def test_multi(judge_llm):
                r1 = judge_llm.judge("doc", "first criterion")
                r2 = judge_llm.judge("doc", "second criterion")
                assert r1 and r2, f"r1={r1}, r2={r2}"
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(failed=1)
        result.stdout.fnmatch_lines(
            [
                "*1 passed, 1 failed*",
                "*FAIL*test_multi*",
                '*criterion: "second criterion"*',
            ]
        )

    def test_record_appears_in_summary(self, pytester, monkeypatch):
        """Manually recorded judgments via record() should appear in the summary."""
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODELS", "groq:fake-model")
        monkeypatch.setenv("PYTHONUTF8", "1")
        monkeypatch.delenv("PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT", raising=False)
        pytester.makeconftest(self._judge_conftest(verdicts=["PASS"]))
        pytester.makepyfile("""
            def test_with_record(judge_llm):
                assert judge_llm.judge("doc", "auto criterion")
                judge_llm.record(criterion="manual criterion", passed=True)
                judge_llm.record(criterion="manual fail", passed=False)
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(
            [
                "*2 passed, 1 failed*",
                "*FAIL*test_with_record*",
                '*criterion: "manual fail"*',
            ]
        )


# ---------------------------------------------------------------------------
# register_judge tests
# ---------------------------------------------------------------------------


class TestRegisterJudge:
    def test_registers_model_and_judgments(self, pytester):
        """register_judge should set model and judgments stash keys."""
        pytester.makeconftest("""
import pytest
from pytest_llm_rubric import register_judge
from pytest_llm_rubric.plugin import _model_stash_key, _judgments_stash_key

class FakeJudge:
    def __init__(self):
        self._judgments = []
    def complete(self, messages, max_output_tokens=256, response_format=None):
        return "PASS"
    def judge(self, document, criterion):
        return True
    def record(self, criterion, *, passed):
        pass

@pytest.fixture(scope="session")
def judge_llm(request):
    judge = FakeJudge()
    register_judge(request.config, judge, model="custom:my-model")
    return judge

@pytest.fixture(scope="session")
def _stash_check(request):
    yield
    assert request.config.stash[_model_stash_key] == "custom:my-model"
    assert request.config.stash[_judgments_stash_key] is not None
""")
        pytester.makepyfile("""
            def test_registered(judge_llm, _stash_check):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(passed=1)

    def test_uses_explicit_judgments_list(self, pytester):
        """register_judge with explicit judgments= should use that list."""
        pytester.makeconftest("""
import pytest
from pytest_llm_rubric import register_judge, JudgmentRecord
from pytest_llm_rubric.plugin import _judgments_stash_key

class FakeJudge:
    def complete(self, messages, max_output_tokens=256, response_format=None):
        return "PASS"
    def judge(self, document, criterion):
        return True
    def record(self, criterion, *, passed):
        pass

@pytest.fixture(scope="session")
def judge_llm(request):
    judge = FakeJudge()
    my_list = [JudgmentRecord(node_id="test", criterion="c", passed=True)]
    register_judge(request.config, judge, model="custom:m", judgments=my_list)
    return judge

@pytest.fixture(scope="session")
def _stash_check(request):
    yield
    jl = request.config.stash[_judgments_stash_key]
    assert len(jl) == 1
    assert jl[0].criterion == "c"
""")
        pytester.makepyfile("""
            def test_explicit(judge_llm, _stash_check):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(passed=1)

    def test_custom_backend_shows_summary(self, pytester, monkeypatch):
        """Custom backend using register_judge should appear in terminal summary."""
        monkeypatch.setenv("PYTHONUTF8", "1")
        pytester.makeconftest("""
import pytest
from pytest_llm_rubric import register_judge, JudgmentRecord

class FakeJudge:
    def __init__(self):
        self._judgments = []
    def complete(self, messages, max_output_tokens=256, response_format=None):
        return "PASS"
    def judge(self, document, criterion):
        self._judgments.append(
            JudgmentRecord(node_id="", criterion=criterion, passed=True)
        )
        return True
    def record(self, criterion, *, passed):
        self._judgments.append(
            JudgmentRecord(node_id="", criterion=criterion, passed=passed)
        )

@pytest.fixture(scope="session")
def judge_llm(request):
    judge = FakeJudge()
    register_judge(request.config, judge, model="custom:my-model")
    return judge
""")
        pytester.makepyfile("""
            def test_one(judge_llm):
                assert judge_llm.judge("doc", "criterion A")
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(passed=1)
        result.stdout.fnmatch_lines(
            ["*LLM Rubric*", "*Model: custom:my-model*", "*1 passed, 0 failed*"]
        )


# ---------------------------------------------------------------------------
# xdist E2E test
# ---------------------------------------------------------------------------


class TestXdist:
    def test_summary_aggregates_across_workers(self, pytester, monkeypatch):
        """Judgments from multiple xdist workers should be aggregated in summary."""
        monkeypatch.setenv("PYTHONUTF8", "1")
        pytester.makeconftest("""
import pytest
from unittest.mock import patch
from pytest_llm_rubric.plugin import AnyLLMJudge, _preflight_or_skip, get_shared_tmp
from pytest_llm_rubric import register_judge
from pytest_llm_rubric.preflight import PreflightResult

@pytest.fixture(scope="session")
def judge_llm(request, tmp_path_factory):
    judge = AnyLLMJudge("fake-model", "groq")
    register_judge(request.config, judge, model="groq:fake-model")
    shared_tmp = get_shared_tmp(request.config, tmp_path_factory)
    fake_result = PreflightResult(
        passed=True, correct=12, total=12, stopped_early=False, details=[],
    )
    with patch("pytest_llm_rubric.plugin.preflight", return_value=fake_result):
        judge = _preflight_or_skip(judge, config=request.config, shared_tmp=shared_tmp)
    def fake_complete(messages, max_output_tokens=256, response_format=None):
        return "PASS"
    judge.complete = fake_complete
    return judge
""")
        pytester.makepyfile("""
            def test_a(judge_llm):
                assert judge_llm.judge("doc", "criterion A")

            def test_b(judge_llm):
                assert judge_llm.judge("doc", "criterion B")
        """)
        result = pytester.runpytest_subprocess("-v", "-n2")
        result.assert_outcomes(passed=2)
        result.stdout.fnmatch_lines(
            ["*LLM Rubric*", "*Model: groq:fake-model*", "*2 passed, 0 failed*"]
        )


# ---------------------------------------------------------------------------
# Integration: actual LLM call (requires a running backend)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration:
    def test_ollama_complete(self):
        result = _make_judge("ollama", "")
        if isinstance(result, str):
            pytest.skip(f"Ollama is not running: {result}")
        judge = result
        response = judge.complete(
            [
                {"role": "user", "content": "Reply with exactly: hello"},
            ]
        )
        assert len(response) > 0

    def test_anthropic_complete(self):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        result = _make_judge("anthropic", "claude-haiku-4-5")
        assert isinstance(result, AnyLLMJudge)
        response = result.complete(
            [
                {"role": "user", "content": "Reply with exactly: hello"},
            ]
        )
        assert "hello" in response.lower()

    def test_openai_complete(self):
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        result = _make_judge("openai", "gpt-5.4-nano")
        assert isinstance(result, AnyLLMJudge)
        response = result.complete(
            [
                {"role": "user", "content": "Reply with exactly: hello"},
            ]
        )
        assert "hello" in response.lower()
