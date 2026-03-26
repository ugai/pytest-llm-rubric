"""pytest plugin entry point for pytest-llm-rubric."""

from __future__ import annotations

import functools
import os
import time
import warnings
from typing import Any, Protocol, cast

import pytest

from pytest_llm_rubric.defaults import AUTO_MODELS
from pytest_llm_rubric.preflight import JUDGE_SYSTEM_PROMPT, parse_verdict, preflight
from pytest_llm_rubric.utils import get_ollama_models, parse_ollama_host

ENV_MODEL = "PYTEST_LLM_RUBRIC_MODEL"
ENV_AUTO_MODELS = "PYTEST_LLM_RUBRIC_AUTO_MODELS"
ENV_SKIP_PREFLIGHT = "PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT"

_preflight_stash_key: pytest.StashKey[str] = pytest.StashKey()


@functools.cache
def _get_known_providers() -> frozenset[str]:
    """Return the set of recognised provider names (cached after first call)."""
    providers: set[str] = {"ollama", "anthropic", "openai"}
    try:
        from any_llm import LLMProvider

        providers |= {p.value.lower() for p in LLMProvider}
    except Exception:  # pragma: no cover
        pass
    return frozenset(providers)


def _parse_model(value: str) -> tuple[str, str]:
    """Parse a ``provider:model`` string into ``(provider, model)``.

    The first colon is used as the separator only when the prefix matches a
    known provider name.  This avoids mis-parsing bare Ollama tags like
    ``qwen3.5:9b`` (where ``qwen3.5`` is not a provider).

    Raises ``ValueError`` when the provider cannot be determined.
    """
    if ":" not in value:
        raise ValueError(
            f"Invalid model format {value!r}. "
            "Expected 'provider:model' (e.g. 'anthropic:claude-haiku-4-5')."
        )

    prefix, rest = value.split(":", 1)
    prefix_lower = prefix.lower()

    known = _get_known_providers()
    if prefix_lower in known:
        return prefix_lower, rest

    raise ValueError(
        f"Unknown provider {prefix!r} in model string {value!r}. "
        f"Known providers: {', '.join(sorted(known))}."
    )


class JudgeLLM(Protocol):
    """Protocol for LLM backends. Override the judge_llm fixture to provide your own."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        max_output_tokens: int = 256,
        response_format: type | None = None,
    ) -> str: ...

    def judge(self, document: str, criterion: str) -> bool: ...


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

    def judge(self, document: str, criterion: str) -> bool:
        """Evaluate whether a document meets a criterion.

        Returns True (PASS) or False (FAIL).
        Raises ValueError if the LLM response cannot be parsed as a verdict.
        """
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"DOCUMENT:\n{document}\n\nCRITERION:\n{criterion}",
            },
        ]
        raw = self.complete(messages).strip()
        verdict = parse_verdict(raw)
        if verdict.startswith("INVALID"):
            raise ValueError(f"Could not parse verdict from LLM response: {raw!r}")
        return verdict == "PASS"


def _make_judge(provider: str, model: str) -> AnyLLMJudge | str:
    """Try to construct an ``AnyLLMJudge`` for the given provider and model.

    Returns the judge on success, or a human-readable reason string on failure.
    """
    _API_KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}

    if provider == "ollama":
        try:
            import ollama as _ollama  # noqa: F401
        except ImportError:
            return "ollama package is not installed."

        base_url = parse_ollama_host(os.environ.get("OLLAMA_HOST"))
        try:
            models = get_ollama_models(base_url)
            if not models:
                return "Ollama is running but has no models installed."
            available = {m["name"] for m in models}
            if model and model in available:
                return AnyLLMJudge(model, "ollama", api_base=base_url)
            elif model and model not in available:
                return (
                    f"Ollama model {model!r} not found. Available: {', '.join(sorted(available))}."
                )
            else:
                return AnyLLMJudge(models[0]["name"], "ollama", api_base=base_url)
        except Exception:
            return f"Could not connect to Ollama at {base_url}."

    elif key_env := _API_KEY_ENV.get(provider):
        api_key = os.environ.get(key_env)
        if not api_key:
            return f"{key_env} is not set."
        return AnyLLMJudge(model, provider, api_key=api_key)

    else:
        return AnyLLMJudge(model, provider)


def _resolve_auto_models(config: pytest.Config) -> list[str]:
    """Resolve the auto-discovery model list.

    Priority: env var > ini option > defaults.AUTO_MODELS.
    """
    env = os.environ.get(ENV_AUTO_MODELS, "").strip()
    if env:
        return [e.strip() for e in env.split(",") if e.strip()]

    ini: list[str] = config.getini("llm_rubric_auto_models")
    if ini:
        return ini

    return AUTO_MODELS


def _default_judge_llm(config: pytest.Config) -> JudgeLLM:
    """Build an LLM judge from ``PYTEST_LLM_RUBRIC_MODEL``.

    The env var must be one of:
      ``provider:model``  — e.g. ``anthropic:claude-haiku-4-5``
      ``auto``            — try each model in the auto-discovery list

    Raises ``pytest.fail`` when the requested backend cannot be reached.
    """
    raw = os.environ.get(ENV_MODEL, "").strip()

    if not raw:
        pytest.fail(
            "PYTEST_LLM_RUBRIC_MODEL is not set. "
            "Set it to 'provider:model' (e.g. 'anthropic:claude-haiku-4-5') "
            "or 'auto' to try defaults."
        )

    if raw.lower() == "auto":
        auto_models = _resolve_auto_models(config)
        reasons: list[str] = []
        for entry in auto_models:
            provider, model = _parse_model(entry)
            result = _make_judge(provider, model)
            if isinstance(result, AnyLLMJudge):
                if provider != "ollama":
                    warnings.warn(
                        f"PYTEST_LLM_RUBRIC_MODEL=auto: using cloud provider "
                        f"'{provider}' ({model}). Test documents will be sent to "
                        f"a third-party API. Set the model explicitly to suppress "
                        f"this warning.",
                        stacklevel=2,
                    )
                return result
            reasons.append(f"  {entry}: {result}")
        pytest.fail("PYTEST_LLM_RUBRIC_MODEL=auto but no backend found.\n" + "\n".join(reasons))

    try:
        provider, model = _parse_model(raw)
    except ValueError as exc:
        pytest.fail(str(exc))
    result = _make_judge(provider, model)
    if isinstance(result, AnyLLMJudge):
        return result
    pytest.fail(f"{raw}: {result}")


def _preflight_or_skip(judge: JudgeLLM, config: pytest.Config | None = None) -> JudgeLLM:
    """Run preflight check and skip if the backend is unreliable."""
    if os.environ.get(ENV_SKIP_PREFLIGHT, "").lower() in ("1", "true", "yes"):
        return judge
    t0 = time.monotonic()
    result = preflight(judge)
    elapsed = time.monotonic() - t0
    if not result.passed:
        failures = [d for d in result.details if not d["correct"]]
        tested = len(result.details)
        suffix = f" (stopped early after {tested}/{result.total})" if result.stopped_early else ""
        msg = (
            f"LLM backend failed preflight "
            f"({result.correct}/{result.total}){suffix} in {elapsed:.1f}s.\n"
            + "\n".join(
                f"  {f['criterion']}: expected {f['expected']}, got {f['actual']}" for f in failures
            )
            + "\nTry a larger model, or set PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT=1 to bypass."
        )
        pytest.skip(msg)
    summary = f"preflight passed ({result.correct}/{result.total}) in {elapsed:.1f}s"
    if config is not None:
        config.stash[_preflight_stash_key] = summary
    return judge


@pytest.fixture(scope="session")
def judge_llm(request: pytest.FixtureRequest) -> JudgeLLM:
    """Provide an LLM backend for rubric judging.

    Override this fixture in your conftest.py to use a custom backend.
    Note: if overriding, use scope="session" to match the default scope.
    The backend is verified once per session via preflight golden tests.
    """
    judge = _default_judge_llm(request.config)
    return _preflight_or_skip(judge, config=request.config)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addini(
        "llm_rubric_auto_models",
        type="linelist",
        default=[],
        help="List of provider:model strings to try when PYTEST_LLM_RUBRIC_MODEL=auto.",
    )


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


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, config: pytest.Config
) -> None:
    summary = config.stash.get(_preflight_stash_key, None)
    if summary is not None:
        terminalreporter.section("LLM Rubric")
        terminalreporter.line(summary)
