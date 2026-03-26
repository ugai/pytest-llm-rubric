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
# _resolve_auto_models tests
# ---------------------------------------------------------------------------


class TestResolveAutoModels:
    def test_env_var_takes_priority(self, pytester, monkeypatch):
        """PYTEST_LLM_RUBRIC_AUTO_MODELS env var wins over ini and defaults."""
        monkeypatch.setenv(
            "PYTEST_LLM_RUBRIC_AUTO_MODELS",
            "anthropic:claude-haiku-4-5, openai:gpt-4o",
        )
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODEL", "auto")
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
        # Should only try the two models from env var, not the defaults.
        result.stdout.fnmatch_lines(
            [
                "*anthropic:claude-haiku-4-5: ANTHROPIC_API_KEY*",
                "*openai:gpt-4o: OPENAI_API_KEY*",
            ]
        )
        # Failure reasons should NOT include an ollama entry (not in the custom list).
        result.stdout.no_fnmatch_line("*ollama:*")

    def test_ini_option_over_defaults(self, pytester, monkeypatch):
        """ini option wins over defaults.AUTO_MODELS."""
        monkeypatch.delenv("PYTEST_LLM_RUBRIC_AUTO_MODELS", raising=False)
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODEL", "auto")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("PYTHONUTF8", "1")
        pytester.makeini("""
            [pytest]
            llm_rubric_auto_models =
                anthropic:claude-haiku-4-5
        """)
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(["*anthropic*ANTHROPIC_API_KEY*"])


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
        """MODEL must be set — no silent default."""
        monkeypatch.delenv("PYTEST_LLM_RUBRIC_MODEL", raising=False)
        monkeypatch.setenv("PYTHONUTF8", "1")
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(errors=1)
        result.stdout.fnmatch_lines(["*PYTEST_LLM_RUBRIC_MODEL*not set*"])

    def test_fails_on_invalid_model_format(self, pytester, monkeypatch):
        """Invalid model string should produce a clear error, not a traceback."""
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODEL", "qwen3.5:9b")
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
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODEL", "ollama:qwen3.5:9b")
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
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODEL", "openai:gpt-4o")
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
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODEL", "anthropic:claude-haiku-4-5")
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
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODEL", "auto")
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

    def test_auto_warns_on_cloud_fallback(self, pytester, monkeypatch):
        """When auto falls through to a cloud provider, a warning should be emitted."""
        monkeypatch.setenv(
            "PYTEST_LLM_RUBRIC_AUTO_MODELS",
            "anthropic:claude-haiku-4-5",
        )
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODEL", "auto")
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

    def test_preflight_failure_includes_action(self, pytester, monkeypatch):
        """Preflight failure message should tell the user what to do next."""
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODEL", "groq:llama-3.3-70b")
        monkeypatch.setenv("PYTHONUTF8", "1")
        monkeypatch.delenv("PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT", raising=False)
        # Provide a conftest that patches preflight to always fail,
        # but lets the real _preflight_or_skip format the message.
        pytester.makeconftest("""
import pytest
from unittest.mock import patch
from pytest_llm_rubric.plugin import AnyLLMJudge, _preflight_or_skip
from pytest_llm_rubric.preflight import PreflightResult

@pytest.fixture(scope="session")
def judge_llm():
    judge = AnyLLMJudge("fake", "groq")
    fake_result = PreflightResult(
        passed=False,
        correct=4,
        total=12,
        stopped_early=True,
        details=[
            {"criterion": "test", "expected": "PASS", "actual": "FAIL", "correct": False},
        ],
    )
    with patch("pytest_llm_rubric.plugin.preflight", return_value=fake_result):
        return _preflight_or_skip(judge)
        """)
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v", "-rs")
        result.assert_outcomes(skipped=1)
        result.stdout.fnmatch_lines(["*PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT*"])

    def test_passthrough_provider_creates_judge(self, pytester, monkeypatch):
        """groq:model creates AnyLLMJudge via passthrough."""
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_MODEL", "groq:llama-3.3-70b-versatile")
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
