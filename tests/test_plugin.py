"""Tests for pytest-rubric-grader plugin."""

from __future__ import annotations

import os

import pytest

from pytest_rubric_grader.plugin import (
    OpenAICompatibleGrader,
    _discover_anthropic,
    _discover_ollama,
    _discover_openai,
)

# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


class TestDiscoverOllama:
    def test_returns_grader_when_running(self):
        grader = _discover_ollama()
        if grader is None:
            pytest.skip("Ollama is not running")
        assert isinstance(grader, OpenAICompatibleGrader)

    def test_returns_none_when_unreachable(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:19999")
        assert _discover_ollama() is None


class TestDiscoverAnthropic:
    def test_returns_none_without_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert _discover_anthropic() is None

    def test_returns_grader_with_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        grader = _discover_anthropic()
        assert isinstance(grader, OpenAICompatibleGrader)


class TestDiscoverOpenAI:
    def test_returns_none_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _discover_openai() is None

    def test_returns_grader_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        grader = _discover_openai()
        assert isinstance(grader, OpenAICompatibleGrader)


# ---------------------------------------------------------------------------
# Fixture tests (using pytester for isolation)
# ---------------------------------------------------------------------------

pytest_plugins = ["pytester"]


class TestGraderLLMFixture:
    def test_skip_when_no_backend(self, pytester, monkeypatch):
        monkeypatch.setenv("PYTEST_RUBRIC_GRADER_BACKEND", "")
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:19999")
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_grader(grader_llm):
                assert grader_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(skipped=1)

    def test_override_fixture(self, pytester):
        pytester.makeconftest("""
            import pytest

            class FakeLLM:
                def complete(self, messages):
                    return "fake response"

            @pytest.fixture
            def grader_llm():
                return FakeLLM()
        """)
        pytester.makepyfile("""
            def test_uses_fake(grader_llm):
                result = grader_llm.complete([{"role": "user", "content": "hi"}])
                assert result == "fake response"
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=1)

    def test_unknown_backend_skips(self, pytester, monkeypatch):
        monkeypatch.setenv("PYTEST_RUBRIC_GRADER_BACKEND", "bogus")
        pytester.makeconftest("")
        pytester.makepyfile("""
            def test_uses_grader(grader_llm):
                assert grader_llm is not None
        """)
        result = pytester.runpytest_subprocess("-v")
        result.assert_outcomes(skipped=1)


# ---------------------------------------------------------------------------
# Integration: actual LLM call (requires a running backend)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestIntegration:
    def test_ollama_complete(self):
        grader = _discover_ollama()
        if grader is None:
            pytest.skip("Ollama is not running")
        response = grader.complete(
            [
                {"role": "user", "content": "Reply with exactly: hello"},
            ]
        )
        assert len(response) > 0

    def test_anthropic_complete(self):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")
        grader = _discover_anthropic()
        assert grader is not None
        response = grader.complete(
            [
                {"role": "user", "content": "Reply with exactly: hello"},
            ]
        )
        assert "hello" in response.lower()

    def test_openai_complete(self):
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")
        grader = _discover_openai()
        assert grader is not None
        response = grader.complete(
            [
                {"role": "user", "content": "Reply with exactly: hello"},
            ]
        )
        assert "hello" in response.lower()
