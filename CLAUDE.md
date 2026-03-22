# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --extra ollama                   # Install dependencies (editable mode, with Ollama)
uv run pre-commit install                # Install pre-commit hooks (ruff + ty)
uv run pytest -m "not integration"       # Run tests (no LLM calls)
uv run pytest -m integration             # Run integration tests (requires Ollama / API keys)
uv run pytest tests/test_plugin.py::TestDiscoverOllama -v  # Run a single test class
uv run ruff check src/ tests/            # Lint
uv run ruff check --fix src/ tests/      # Lint with auto-fix
uv run ruff format src/ tests/           # Format
uv run ty check src/                     # Type check (required — py.typed marker is present)
uv run python -m pytest_llm_rubric.find_local_model  # Find best local model
```

## Architecture

This is a pytest plugin (`pytest11` entry point) that provides `judge_llm`, a session-scoped fixture for rubric-based LLM-as-judge testing.

**Core pipeline: discover → preflight → judge**

- **`plugin.py`** — Entry point. Defines the `JudgeLLM` Protocol and `AnyLLMJudge` implementation. The `judge_llm` fixture auto-discovers a backend, runs preflight, and returns it. Users override the fixture in their `conftest.py` for custom backends.

- **`preflight.py`** — Golden test suite (12 pairs: 6 short-form + 6 haystack) that validates whether an LLM can reliably do binary PASS/FAIL semantic judgments. Session runs preflight once; if the LLM fails, all rubric tests skip.

- **`defaults.py`** — Single file for default model names and endpoints per provider. Intended to be human-editable.

- **`find_local_model.py`** — CLI tool that runs preflight against all local models (currently Ollama) and recommends the smallest passing one.

**Backend discovery order** is controlled by `PYTEST_LLM_RUBRIC_BACKEND`:
- Empty (default): Ollama only — safe, no API costs
- `auto`: Ollama → Anthropic → OpenAI
- Explicit: `ollama` / `anthropic` / `openai`

Anthropic is accessed via its OpenAI-compatible endpoint (`api.anthropic.com/v1`), so all three providers use `OpenAICompatibleJudge`.

**Model resolution** for each provider: `PYTEST_LLM_RUBRIC_<PROVIDER>_MODEL` > `PYTEST_LLM_RUBRIC_MODEL` > default in `defaults.py`. This prevents model name collisions when `auto` mode falls through multiple providers.

## Key Design Decisions

- The `judge_llm` fixture is `scope="session"` — preflight runs once per test session.
- `PYTEST_LLM_RUBRIC_BACKEND` defaults to empty (Ollama only) to prevent accidental API costs. Cloud APIs require explicit opt-in.
- `max_tokens=512` for preflight calls (accommodates thinking models), `256` default for general use.
- Preflight golden tests include "haystack" pairs (rule buried in long doc vs. similar doc without the rule) to screen out models that can only do trivial matching.
