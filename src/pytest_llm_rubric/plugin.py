"""pytest plugin entry point for pytest-llm-rubric."""

from __future__ import annotations

import os
import warnings
from typing import Any, Protocol, cast

import pytest

from pytest_llm_rubric.defaults import (
    ANTHROPIC_MODEL,
    OLLAMA_MODEL,
    OPENAI_MODEL,
)
from pytest_llm_rubric.preflight import preflight
from pytest_llm_rubric.utils import parse_ollama_host

ENV_BACKEND = "PYTEST_LLM_RUBRIC_BACKEND"
ENV_MODEL = "PYTEST_LLM_RUBRIC_MODEL"
ENV_SKIP_PREFLIGHT = "PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT"
ENV_OLLAMA_MODEL = "PYTEST_LLM_RUBRIC_OLLAMA_MODEL"
ENV_ANTHROPIC_MODEL = "PYTEST_LLM_RUBRIC_ANTHROPIC_MODEL"
ENV_ANTHROPIC_BASE_URL = "PYTEST_LLM_RUBRIC_ANTHROPIC_BASE_URL"
ENV_OPENAI_MODEL = "PYTEST_LLM_RUBRIC_OPENAI_MODEL"


def _resolve_model(env_var: str, default: str) -> str:
    """Resolve model name: provider-specific env var > shared env var > default."""
    return os.environ.get(env_var) or os.environ.get(ENV_MODEL) or default


class JudgeLLM(Protocol):
    """Protocol for LLM backends. Override the judge_llm fixture to provide your own."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_output_tokens: int = 256,
        response_format: type | None = None,
    ) -> str: ...


class AnyLLMJudge:
    """Judge backed by any-llm-sdk. Supports Ollama, OpenAI, Anthropic, and more."""

    def __init__(
        self,
        model: str,
        provider: str,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._provider = provider
        self._api_base = api_base
        self._api_key = api_key

    _MAX_EMPTY_RETRIES = 2

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_output_tokens: int = 256,
        response_format: type | None = None,
    ) -> str:
        from any_llm import completion
        from any_llm.types.completion import ChatCompletion

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "provider": self._provider,
            "max_tokens": max_output_tokens,
            "stream": False,
        }
        if self._api_base is not None:
            kwargs["api_base"] = self._api_base
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if response_format is not None:
            kwargs["response_format"] = response_format

        for attempt in range(1 + self._MAX_EMPTY_RETRIES):
            response = cast(ChatCompletion, completion(**kwargs))
            content = response.choices[0].message.content or ""
            if content:
                return content
            if attempt < self._MAX_EMPTY_RETRIES:
                warnings.warn(
                    f"Empty response from {self._model} "
                    f"(attempt {attempt + 1}/{1 + self._MAX_EMPTY_RETRIES}), retrying.",
                    stacklevel=2,
                )
        return ""


def _discover_ollama() -> AnyLLMJudge | None:
    """Try to connect to a local Ollama instance."""
    try:
        import ollama as _ollama  # noqa: F401
    except ImportError:
        warnings.warn(
            "ollama package is not installed. Install with: pip install 'any-llm-sdk[ollama]'",
            stacklevel=2,
        )
        return None

    import httpx

    base_url = parse_ollama_host(os.environ.get("OLLAMA_HOST"))
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        if not models:
            return None
        available = {m["name"] for m in models}
        requested = _resolve_model(ENV_OLLAMA_MODEL, OLLAMA_MODEL or "")
        if requested and requested in available:
            model_name = requested
        elif requested and requested not in available:
            warnings.warn(
                f"Requested Ollama model {requested!r} not found. "
                f"Available: {', '.join(sorted(available))}.",
                stacklevel=2,
            )
            return None
        else:
            model_name = models[0]["name"]
        return AnyLLMJudge(model_name, "ollama", api_base=base_url)
    except Exception:
        return None


def _discover_anthropic() -> AnyLLMJudge | None:
    """Use Anthropic API if ANTHROPIC_API_KEY is set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    model = _resolve_model(ENV_ANTHROPIC_MODEL, ANTHROPIC_MODEL)
    api_base = os.environ.get(ENV_ANTHROPIC_BASE_URL)
    return AnyLLMJudge(model, "anthropic", api_key=api_key, api_base=api_base)


def _discover_openai() -> AnyLLMJudge | None:
    """Use OpenAI API if OPENAI_API_KEY is set."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    model = _resolve_model(ENV_OPENAI_MODEL, OPENAI_MODEL)
    return AnyLLMJudge(model, "openai", api_key=api_key)


def _default_judge_llm() -> JudgeLLM:
    """Auto-discover an available LLM backend.

    Controlled by PYTEST_LLM_RUBRIC_BACKEND:
      (unset)    - try Ollama only, skip if unavailable
      auto       - try Ollama → Anthropic → OpenAI; fail if none found
      ollama     - Ollama only; fail if unavailable
      anthropic  - Anthropic API only; fail if unavailable
      openai     - OpenAI API only; fail if unavailable

    Explicit backends fail (pytest.fail) instead of skip so that CI
    misconfigurations surface as errors rather than silent skips.
    """
    backend = os.environ.get(ENV_BACKEND, "").lower()

    if backend == "openai":
        if (judge := _discover_openai()) is not None:
            return judge
        pytest.fail("PYTEST_LLM_RUBRIC_BACKEND=openai but OPENAI_API_KEY is not set.")

    elif backend == "anthropic":
        if (judge := _discover_anthropic()) is not None:
            return judge
        pytest.fail("PYTEST_LLM_RUBRIC_BACKEND=anthropic but ANTHROPIC_API_KEY is not set.")

    elif backend == "ollama" or backend == "":
        if (judge := _discover_ollama()) is not None:
            return judge
        if backend == "ollama":
            pytest.fail("PYTEST_LLM_RUBRIC_BACKEND=ollama but Ollama is not running.")
        pytest.skip("No LLM backend available. Run Ollama or set PYTEST_LLM_RUBRIC_BACKEND.")

    elif backend == "auto":
        for discover in (_discover_ollama, _discover_anthropic, _discover_openai):
            if (judge := discover()) is not None:
                return judge
        pytest.fail("PYTEST_LLM_RUBRIC_BACKEND=auto but no backend found.")

    else:
        pytest.fail(f"Unknown PYTEST_LLM_RUBRIC_BACKEND: {backend!r}")


def _preflight_or_skip(judge: JudgeLLM) -> JudgeLLM:
    """Run preflight check and skip if the backend is unreliable."""
    if os.environ.get(ENV_SKIP_PREFLIGHT, "").lower() in ("1", "true", "yes"):
        return judge
    result = preflight(judge)
    if not result.passed:
        failures = [d for d in result.details if not d["correct"]]
        tested = len(result.details)
        suffix = f" (stopped early after {tested}/{result.total})" if result.stopped_early else ""
        msg = (
            f"LLM backend failed preflight ({result.correct}/{result.total}){suffix}.\n"
            + "\n".join(
                f"  {f['criterion']}: expected {f['expected']}, got {f['actual']}" for f in failures
            )
        )
        pytest.skip(msg)
    return judge


@pytest.fixture(scope="session")
def judge_llm() -> JudgeLLM:
    """Provide an LLM backend for rubric judging.

    Override this fixture in your conftest.py to use a custom backend.
    Note: if overriding, use scope="session" to match the default scope.
    The backend is verified once per session via preflight golden tests.
    """
    judge = _default_judge_llm()
    return _preflight_or_skip(judge)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "llm_rubric: tests that use the judge_llm fixture (auto-applied)",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-apply the llm_rubric marker to tests that use judge_llm."""
    marker = pytest.mark.llm_rubric
    for item in items:
        if "judge_llm" in getattr(item, "fixturenames", ()):
            item.add_marker(marker)
