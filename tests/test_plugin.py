"""Tests for pytest-llm-rubric plugin."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from pytest_llm_rubric.plugin import (
    OpenAICompatibleJudge,
    _discover_anthropic,
    _discover_ollama,
    _discover_openai,
)

# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


class TestDiscoverOllama:
    def test_returns_judge_when_running(self):
        judge = _discover_ollama()
        if judge is None:
            pytest.skip("Ollama is not running")
        assert isinstance(judge, OpenAICompatibleJudge)

    def test_returns_none_when_unreachable(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:19999")
        assert _discover_ollama() is None

    def test_returns_none_when_model_not_found(self, monkeypatch):
        """Requesting a non-existent model should return None, not silently substitute."""
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_OLLAMA_MODEL", "nonexistent-model-xyz")
        judge = _discover_ollama()
        if judge is not None:
            # Ollama is not running or has the exact model — skip
            pytest.skip("Ollama not running or model unexpectedly exists")
        assert judge is None


class TestDiscoverAnthropic:
    def test_returns_none_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert _discover_anthropic() is None

    def test_returns_judge_with_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        judge = _discover_anthropic()
        assert isinstance(judge, OpenAICompatibleJudge)


class TestDiscoverOpenAI:
    def test_returns_none_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _discover_openai() is None

    def test_returns_judge_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        judge = _discover_openai()
        assert isinstance(judge, OpenAICompatibleJudge)


# ---------------------------------------------------------------------------
# Fixture tests (using pytester for isolation)
# ---------------------------------------------------------------------------

pytest_plugins = ["pytester"]

# Shared conftest snippet for pytester tests that need a fake judge_llm.
# Defined once here so that JudgeLLM Protocol changes only require one update.
_FAKE_JUDGE_CONFTEST = """
import pytest

class FakeLLM:
    def complete(self, messages, max_output_tokens=256):
        return "fake"

@pytest.fixture(scope="session")
def judge_llm():
    return FakeLLM()
"""


class TestJudgeLLMFixture:
    def test_skip_when_no_backend(self, pytester, monkeypatch):
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_BACKEND", "")
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:19999")
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(skipped=1)

    def test_default_backend_ignores_api_keys(self, pytester, monkeypatch):
        """Default (empty) backend must use Ollama only, even when paid API keys are present."""
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_BACKEND", "")
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:19999")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-should-not-be-used")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-should-not-be-used")
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(skipped=1)

    def test_override_fixture(self, pytester):
        pytester.makeconftest(_FAKE_JUDGE_CONFTEST)
        pytester.makepyfile("""
            def test_uses_fake(judge_llm):
                result = judge_llm.complete([{"role": "user", "content": "hi"}])
                assert result == "fake"
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(passed=1)

    def test_unknown_backend_skips(self, pytester, monkeypatch):
        monkeypatch.setenv("PYTEST_LLM_RUBRIC_BACKEND", "bogus")
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_judge(judge_llm):
                assert judge_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(skipped=1)

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
# max_completion_tokens vs max_tokens
# ---------------------------------------------------------------------------


class TestMaxTokensParam:
    """Verify that use_legacy_max_tokens controls which parameter is sent."""

    def _make_judge(
        self, *, use_legacy_max_tokens: bool
    ) -> tuple[OpenAICompatibleJudge, MagicMock]:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="PASS"))]
        )
        judge = OpenAICompatibleJudge(
            mock_client, "test-model", use_legacy_max_tokens=use_legacy_max_tokens
        )
        return judge, mock_client

    def test_default_uses_max_completion_tokens(self):
        judge, mock_client = self._make_judge(use_legacy_max_tokens=False)
        judge.complete([{"role": "user", "content": "hi"}], max_output_tokens=100)
        kwargs = mock_client.chat.completions.create.call_args
        assert "max_completion_tokens" in kwargs.kwargs
        assert "max_tokens" not in kwargs.kwargs
        assert kwargs.kwargs["max_completion_tokens"] == 100

    def test_legacy_uses_max_tokens(self):
        judge, mock_client = self._make_judge(use_legacy_max_tokens=True)
        judge.complete([{"role": "user", "content": "hi"}], max_output_tokens=100)
        kwargs = mock_client.chat.completions.create.call_args
        assert "max_tokens" in kwargs.kwargs
        assert "max_completion_tokens" not in kwargs.kwargs
        assert kwargs.kwargs["max_tokens"] == 100


# ---------------------------------------------------------------------------
# Integration: actual LLM call (requires a running backend)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration:
    def test_ollama_complete(self):
        judge = _discover_ollama()
        if judge is None:
            pytest.skip("Ollama is not running")
        response = judge.complete(
            [
                {"role": "user", "content": "Reply with exactly: hello"},
            ]
        )
        assert len(response) > 0

    def test_anthropic_complete(self):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        judge = _discover_anthropic()
        assert judge is not None
        response = judge.complete(
            [
                {"role": "user", "content": "Reply with exactly: hello"},
            ]
        )
        assert "hello" in response.lower()

    def test_openai_complete(self):
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        judge = _discover_openai()
        assert judge is not None
        response = judge.complete(
            [
                {"role": "user", "content": "Reply with exactly: hello"},
            ]
        )
        assert "hello" in response.lower()
