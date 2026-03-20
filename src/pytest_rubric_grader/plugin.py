"""pytest plugin entry point for rubric-grader."""

from __future__ import annotations

import os
import warnings
from typing import Any, Protocol, cast

import pytest
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from pytest_rubric_grader.calibration import calibrate
from pytest_rubric_grader.defaults import (
    ANTHROPIC_BASE_URL,
    ANTHROPIC_MODEL,
    OLLAMA_MODEL,
    OPENAI_MODEL,
)

ENV_BACKEND = "PYTEST_RUBRIC_GRADER_BACKEND"
ENV_MODEL = "PYTEST_RUBRIC_GRADER_MODEL"
ENV_SKIP_CALIBRATION = "PYTEST_RUBRIC_GRADER_SKIP_CALIBRATION"


def _resolve_model(env_var: str, default: str) -> str:
    """Resolve model name: provider-specific env var > shared env var > default."""
    return os.environ.get(env_var) or os.environ.get(ENV_MODEL) or default


class GraderLLM(Protocol):
    """Protocol for LLM backends. Override the grader_llm fixture to provide your own."""

    def complete(self, messages: list[dict[str, Any]], max_tokens: int = 256) -> str: ...


class OpenAICompatibleGrader:
    """Grader backed by any OpenAI-compatible API (OpenAI, Anthropic, Ollama, Groq, etc.)."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def complete(self, messages: list[dict[str, Any]], max_tokens: int = 256) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


def _discover_ollama() -> OpenAICompatibleGrader | None:
    """Try to connect to a local Ollama instance."""
    import httpx

    base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        resp = httpx.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        if not models:
            return None
        available = {m["name"] for m in models}
        requested = _resolve_model("PYTEST_RUBRIC_GRADER_OLLAMA_MODEL", OLLAMA_MODEL or "")
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
        client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama", timeout=120.0)
        return OpenAICompatibleGrader(client, model_name)
    except Exception:
        return None


def _discover_anthropic() -> OpenAICompatibleGrader | None:
    """Use Anthropic API via OpenAI-compatible endpoint if ANTHROPIC_API_KEY is set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    model = _resolve_model("PYTEST_RUBRIC_GRADER_ANTHROPIC_MODEL", ANTHROPIC_MODEL)
    base_url = os.environ.get("PYTEST_RUBRIC_GRADER_ANTHROPIC_BASE_URL", ANTHROPIC_BASE_URL)
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=30.0)
    return OpenAICompatibleGrader(client, model)


def _discover_openai() -> OpenAICompatibleGrader | None:
    """Use OpenAI API if OPENAI_API_KEY is set."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    model = _resolve_model("PYTEST_RUBRIC_GRADER_OPENAI_MODEL", OPENAI_MODEL)
    client = OpenAI(api_key=api_key, timeout=30.0)
    return OpenAICompatibleGrader(client, model)


def _default_grader_llm() -> GraderLLM:
    """Auto-discover an available LLM backend.

    Controlled by PYTEST_RUBRIC_GRADER_BACKEND:
      (unset)    - try Ollama only, skip if unavailable
      auto       - try Ollama, then Anthropic, then OpenAI
      ollama     - Ollama only
      anthropic  - Anthropic API only
      openai     - OpenAI API only
    """
    backend = os.environ.get(ENV_BACKEND, "").lower()

    if backend == "openai":
        grader = _discover_openai()
        if grader is not None:
            return grader
        pytest.skip("PYTEST_RUBRIC_GRADER_BACKEND=openai but OPENAI_API_KEY is not set.")

    elif backend == "anthropic":
        grader = _discover_anthropic()
        if grader is not None:
            return grader
        pytest.skip("PYTEST_RUBRIC_GRADER_BACKEND=anthropic but ANTHROPIC_API_KEY is not set.")

    elif backend == "ollama" or backend == "":
        grader = _discover_ollama()
        if grader is not None:
            return grader
        if backend == "ollama":
            pytest.skip("PYTEST_RUBRIC_GRADER_BACKEND=ollama but Ollama is not running.")
        pytest.skip("No LLM backend available. Run Ollama or set PYTEST_RUBRIC_GRADER_BACKEND.")

    elif backend == "auto":
        grader = _discover_ollama()
        if grader is not None:
            return grader
        grader = _discover_anthropic()
        if grader is not None:
            return grader
        grader = _discover_openai()
        if grader is not None:
            return grader
        pytest.skip("PYTEST_RUBRIC_GRADER_BACKEND=auto but no backend found.")

    else:
        pytest.skip(f"Unknown PYTEST_RUBRIC_GRADER_BACKEND: {backend!r}")


def _calibrate_or_skip(grader: GraderLLM) -> GraderLLM:
    """Run calibration and skip if the backend is unreliable."""
    if os.environ.get(ENV_SKIP_CALIBRATION, "").lower() in ("1", "true", "yes"):
        return grader
    result = calibrate(grader)
    if not result.passed:
        failures = [d for d in result.details if not d["correct"]]
        msg = f"LLM backend failed calibration ({result.correct}/{result.total}).\n" + "\n".join(
            f"  {f['criterion']}: expected {f['expected']}, got {f['actual']}" for f in failures
        )
        pytest.skip(msg)
    return grader


@pytest.fixture(scope="session")
def grader_llm() -> GraderLLM:
    """Provide an LLM backend for rubric grading.

    Override this fixture in your conftest.py to use a custom backend.
    Note: if overriding, use scope="session" to match the default scope.
    The backend is calibrated once per session against golden tests.
    """
    grader = _default_grader_llm()
    return _calibrate_or_skip(grader)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "rubric_grading: tests that use the grader_llm fixture (auto-applied)",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-apply the rubric_grading marker to tests that use grader_llm."""
    marker = pytest.mark.rubric_grading
    for item in items:
        if "grader_llm" in getattr(item, "fixturenames", ()):
            item.add_marker(marker)
