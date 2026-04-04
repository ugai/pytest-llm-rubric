"""pytest plugin entry point for pytest-llm-rubric."""

from __future__ import annotations

import functools
import json
import os
import time
import warnings
from collections.abc import Generator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import pytest
from filelock import FileLock

from pytest_llm_rubric.defaults import AUTO_MODELS
from pytest_llm_rubric.preflight import JUDGE_SYSTEM_PROMPT, parse_verdict, preflight
from pytest_llm_rubric.utils import get_ollama_models, parse_ollama_host

ENV_MODELS = "PYTEST_LLM_RUBRIC_MODELS"
ENV_SKIP_PREFLIGHT = "PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT"


@dataclass
class JudgmentRecord:
    """One judge() call result, recorded for the terminal summary."""

    node_id: str
    criterion: str
    passed: bool


_preflight_stash_key: pytest.StashKey[str] = pytest.StashKey()
_model_stash_key: pytest.StashKey[str] = pytest.StashKey()
_judgments_stash_key: pytest.StashKey[list[JudgmentRecord]] = pytest.StashKey()
_shared_tmp_stash_key: pytest.StashKey[Path] = pytest.StashKey()

# Set by pytest_runtest_call hook so judge() can tag results with the test node ID.
# Per-process global — safe under pytest-xdist (each worker is a separate process).
_current_node_id: str = ""


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

    def record(self, criterion: str, *, passed: bool) -> None: ...


# ---------------------------------------------------------------------------
# Public helper: register_judge
# ---------------------------------------------------------------------------


def register_judge(
    config: pytest.Config,
    judge: JudgeLLM,
    *,
    model: str,
    judgments: list[JudgmentRecord] | None = None,
) -> None:
    """Register a judge backend for terminal summary support.

    Call this from a custom ``judge_llm`` fixture override so that the
    LLM Rubric terminal summary can report the model name and judgment results.

    Example::

        from pytest_llm_rubric import register_judge

        @pytest.fixture(scope="session")
        def judge_llm(request):
            judge = MyCustomBackend()
            register_judge(request.config, judge, model="custom:my-model")
            return judge

    Parameters
    ----------
    config:
        The pytest ``Config`` object (available via ``request.config``).
    judge:
        The judge instance.  If it has a ``_judgments`` attribute (like
        ``AnyLLMJudge``), that list is used automatically.
    model:
        A ``provider:model`` string shown in the terminal summary header.
    judgments:
        Optional explicit judgments list.  When *None*, ``judge._judgments``
        is used if present; otherwise a new empty list is created and stored.
    """
    config.stash[_model_stash_key] = model
    if judgments is None:
        judgments = getattr(judge, "_judgments", None)
        if judgments is None:
            judgments = []
    config.stash[_judgments_stash_key] = judgments


# ---------------------------------------------------------------------------
# Shared tmp directory helpers (xdist inter-process communication)
# ---------------------------------------------------------------------------


def get_shared_tmp(
    config: pytest.Config,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Return the shared tmp directory (common to all xdist workers)."""
    if _shared_tmp_stash_key in config.stash:
        return config.stash[_shared_tmp_stash_key]
    base = tmp_path_factory.getbasetemp()
    shared = base.parent if os.environ.get("PYTEST_XDIST_WORKER") else base
    config.stash[_shared_tmp_stash_key] = shared
    return shared


# ---------------------------------------------------------------------------
# AnyLLMJudge implementation
# ---------------------------------------------------------------------------


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
        self._judgments: list[JudgmentRecord] = []

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
        passed = verdict == "PASS"
        self._judgments.append(
            JudgmentRecord(node_id=_current_node_id, criterion=criterion, passed=passed)
        )
        return passed

    def record(self, criterion: str, *, passed: bool) -> None:
        """Manually record a judgment for the terminal summary.

        Use this for verdicts obtained via ``complete()`` with custom prompts,
        so they appear in the LLM Rubric summary alongside ``judge()`` results.
        """
        self._judgments.append(
            JudgmentRecord(node_id=_current_node_id, criterion=criterion, passed=passed)
        )


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


def _resolve_models(config: pytest.Config) -> tuple[str, list[str]]:
    """Resolve the model list from env var, ini option, or ``auto``.

    Returns ``(source, models)`` where *source* is a label for error messages
    and *models* is the list of ``provider:model`` strings to try in order.

    Raises ``pytest.fail`` when no configuration is found.
    """
    raw = os.environ.get(ENV_MODELS, "").strip()

    if not raw:
        # Fall back to ini option when the env var is unset.
        ini: list[str] = config.getini("llm_rubric_models")
        if ini:
            return "llm_rubric_models (ini)", ini
        pytest.fail(
            "PYTEST_LLM_RUBRIC_MODELS is not set. "
            "Set it to 'provider:model' (e.g. 'anthropic:claude-haiku-4-5'), "
            "a comma-separated list, 'auto' to try defaults, or configure "
            "llm_rubric_models in your pyproject.toml [tool.pytest.ini_options]."
        )

    if raw.lower() == "auto":
        return "auto (defaults)", AUTO_MODELS

    entries = [e.strip() for e in raw.split(",") if e.strip()]
    return "PYTEST_LLM_RUBRIC_MODELS", entries


def _default_judge_llm(config: pytest.Config) -> JudgeLLM:
    """Build an LLM judge from ``PYTEST_LLM_RUBRIC_MODELS``.

    The env var accepts:
      ``provider:model``                  — use that backend directly
      ``provider:m1,provider:m2,...``      — try each in order
      ``auto``                            — try the default model list

    Raises ``pytest.fail`` when no usable backend is found.
    """
    source, models = _resolve_models(config)

    if len(models) == 1:
        # Single model — fail immediately if unavailable.
        try:
            provider, model = _parse_model(models[0])
        except ValueError as exc:
            pytest.fail(str(exc))
        result = _make_judge(provider, model)
        if isinstance(result, AnyLLMJudge):
            return result
        pytest.fail(f"{models[0]}: {result}")

    # Multiple models — try each in order, like the old "auto" behaviour.
    reasons: list[str] = []
    for entry in models:
        try:
            provider, model = _parse_model(entry)
        except ValueError as exc:
            reasons.append(f"  {entry}: {exc}")
            continue
        result = _make_judge(provider, model)
        if isinstance(result, AnyLLMJudge):
            if provider != "ollama":
                warnings.warn(
                    f"PYTEST_LLM_RUBRIC_MODELS: using cloud provider "
                    f"'{provider}' ({model}). Test documents will be sent to "
                    f"a third-party API. Set a single model explicitly to "
                    f"suppress this warning.",
                    stacklevel=2,
                )
            return result
        reasons.append(f"  {entry}: {result}")
    pytest.fail(f"{source}: no backend found.\n" + "\n".join(reasons))


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------


def _run_preflight_check(judge: JudgeLLM) -> dict[str, Any]:
    """Run preflight and return a result dict (serialisable to JSON)."""
    t0 = time.monotonic()
    result = preflight(judge)
    elapsed = time.monotonic() - t0
    if not result.passed:
        failures = [d for d in result.details if not d["correct"]]
        tested = len(result.details)
        suffix = f" (stopped early after {tested}/{result.total})" if result.stopped_early else ""
        summary = f"FAILED ({result.correct}/{result.total}){suffix} in {elapsed:.1f}s"
        skip_msg = (
            f"LLM backend failed preflight "
            f"({result.correct}/{result.total}){suffix} in {elapsed:.1f}s.\n"
            + "\n".join(
                f"  {f['criterion']}: expected {f['expected']}, got {f['actual']}" for f in failures
            )
            + "\nTry a larger model, or set "
            "PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT=1 to bypass."
        )
        return {"passed": False, "summary": summary, "skip_msg": skip_msg}
    summary = f"preflight passed ({result.correct}/{result.total}) in {elapsed:.1f}s"
    return {"passed": True, "summary": summary, "skip_msg": None}


def _preflight_or_skip(
    judge: JudgeLLM,
    config: pytest.Config,
    *,
    shared_tmp: Path | None = None,
) -> JudgeLLM:
    """Run preflight check and skip if the backend is unreliable.

    When *shared_tmp* is provided, a ``FileLock`` ensures preflight runs only
    once across pytest-xdist workers.
    """
    if os.environ.get(ENV_SKIP_PREFLIGHT, "").lower() in ("1", "true", "yes"):
        return judge

    if shared_tmp is not None:
        preflight_file = shared_tmp / "llm_rubric_preflight.json"
        with FileLock(str(preflight_file) + ".lock"):
            if preflight_file.exists():
                data = json.loads(preflight_file.read_text(encoding="utf-8"))
            else:
                data = _run_preflight_check(judge)
                preflight_file.write_text(json.dumps(data), encoding="utf-8")
    else:
        data = _run_preflight_check(judge)

    config.stash[_preflight_stash_key] = data["summary"]
    if not data["passed"]:
        pytest.skip(data["skip_msg"])
    return judge


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def judge_llm(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> JudgeLLM:
    """Provide an LLM backend for rubric judging.

    Override this fixture in your conftest.py to use a custom backend.
    Note: if overriding, use scope="session" to match the default scope.
    The backend is verified once per session via preflight golden tests.
    """
    judge = _default_judge_llm(request.config)
    if isinstance(judge, AnyLLMJudge):
        register_judge(request.config, judge, model=f"{judge._provider}:{judge._model}")
    shared_tmp = get_shared_tmp(request.config, tmp_path_factory)
    judge = _preflight_or_skip(judge, request.config, shared_tmp=shared_tmp)
    return judge


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addini(
        "llm_rubric_models",
        type="linelist",
        default=[],
        help="List of provider:model strings to try in order. "
        "Used when PYTEST_LLM_RUBRIC_MODELS is unset.",
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


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Generator[None, None, None]:
    global _current_node_id  # noqa: PLW0603
    _current_node_id = item.nodeid
    try:
        return (yield)
    finally:
        _current_node_id = ""


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Write judgment data to the shared tmp dir for xdist aggregation."""
    if not os.environ.get("PYTEST_XDIST_WORKER"):
        return

    config = session.config
    shared_tmp = config.stash.get(_shared_tmp_stash_key, None)
    if shared_tmp is None:
        return

    model = config.stash.get(_model_stash_key, None)
    preflight_summary = config.stash.get(_preflight_stash_key, None)
    judgments = config.stash.get(_judgments_stash_key, None)

    if model is None and preflight_summary is None and not judgments:
        return

    worker_id = os.environ["PYTEST_XDIST_WORKER"]
    result_file = shared_tmp / f"llm_rubric_{worker_id}.json"
    data = {
        "model": model,
        "preflight": preflight_summary,
        "judgments": [asdict(j) for j in (judgments or [])],
    }
    result_file.write_text(json.dumps(data), encoding="utf-8")


def _aggregate_worker_results(
    config: pytest.Config,
) -> tuple[str | None, str | None, list[JudgmentRecord]]:
    """Read judgment data from xdist worker JSON files."""
    try:
        # Private API — no public way to get basetemp from a hook.
        # Stable since pytest 5.4; if it breaks, only xdist summary is affected.
        base = config._tmp_path_factory.getbasetemp()  # type: ignore[attr-defined]
    except AttributeError:
        return None, None, []

    # xdist worker IDs are "gw0", "gw1", … — written by pytest_sessionfinish.
    files = sorted(base.glob("llm_rubric_gw*.json"))
    if not files:
        return None, None, []

    model: str | None = None
    preflight_summary: str | None = None
    all_judgments: list[JudgmentRecord] = []

    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("model") and model is None:
            model = data["model"]
        if data.get("preflight") and preflight_summary is None:
            preflight_summary = data["preflight"]
        for j in data.get("judgments", []):
            all_judgments.append(JudgmentRecord(**j))

    return model, preflight_summary, all_judgments


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter, config: pytest.Config
) -> None:
    preflight_val = config.stash.get(_preflight_stash_key, None)
    model = config.stash.get(_model_stash_key, None)
    judgments: list[JudgmentRecord] | None = config.stash.get(_judgments_stash_key, None)

    # xdist controller: no in-memory data — aggregate from worker files.
    if preflight_val is None and model is None and not judgments:
        model, preflight_val, agg = _aggregate_worker_results(config)
        judgments = agg or None

    if preflight_val is None and not judgments:
        return

    terminalreporter.section("LLM Rubric")

    # Header: model + preflight
    header_parts: list[str] = []
    if model is not None:
        header_parts.append(f"Model: {model}")
    if preflight_val is not None:
        header_parts.append(f"Preflight: {preflight_val}")
    if header_parts:
        terminalreporter.line("  ".join(header_parts))

    if not judgments:
        return

    passed = sum(1 for j in judgments if j.passed)
    failed = len(judgments) - passed
    terminalreporter.line(f"{passed} passed, {failed} failed")

    if failed:
        terminalreporter.line("")
        for j in judgments:
            if not j.passed:
                terminalreporter.line(f"  FAIL {j.node_id}")
                terminalreporter.line(f'       criterion: "{j.criterion}"')
