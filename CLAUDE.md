# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                  # Install dependencies (editable mode)
uv run pre-commit install                # Install pre-commit hooks (ruff + ty)
uv run pytest -m "not integration"       # Run tests (no LLM calls)
uv run pytest -m integration             # Run integration tests (requires Ollama / API keys)
uv run pytest tests/test_plugin.py::TestDiscoverOllama -v  # Run a single test class
uv run ruff check src/ tests/            # Lint
uv run ruff check --fix src/ tests/      # Lint with auto-fix
uv run ruff format src/ tests/           # Format
uv run ty check src/                     # Type check (required — py.typed marker is present)
uv run python -m pytest_rubric_grader.find_model  # Find best Ollama model
```

## Architecture

This is a pytest plugin (`pytest11` entry point) that provides `grader_llm`, a session-scoped fixture for rubric-based LLM-as-judge testing.

**Core pipeline: discover → calibrate → judge**

- **`plugin.py`** — Entry point. Defines the `GraderLLM` Protocol and `OpenAICompatibleGrader` implementation. The `grader_llm` fixture auto-discovers a backend, calibrates it, and returns it. All providers use the OpenAI SDK (`openai` is the only LLM dependency). Users override the fixture in their `conftest.py` for custom backends.

- **`calibration.py`** — Golden test suite (12 pairs: 6 short-form + 6 haystack) that validates whether an LLM can reliably do binary PASS/FAIL semantic judgments. Session runs calibration once; if the LLM fails, all rubric tests skip.

- **`defaults.py`** — Single file for default model names and endpoints per provider. Intended to be human-editable.

- **`find_model.py`** — CLI tool that runs calibration against all Ollama models and recommends the smallest passing one.

**Backend discovery order** is controlled by `PYTEST_RUBRIC_GRADER_BACKEND`:
- Empty (default): Ollama only — safe, no API costs
- `auto`: Ollama → Anthropic → OpenAI
- Explicit: `ollama` / `anthropic` / `openai`

Anthropic is accessed via its OpenAI-compatible endpoint (`api.anthropic.com/v1`), so all three providers use `OpenAICompatibleGrader`.

**Model resolution** for each provider: `PYTEST_RUBRIC_GRADER_<PROVIDER>_MODEL` > `PYTEST_RUBRIC_GRADER_MODEL` > default in `defaults.py`. This prevents model name collisions when `auto` mode falls through multiple providers.

## Key Design Decisions

- The `grader_llm` fixture is `scope="session"` — calibration runs once per test session.
- `PYTEST_RUBRIC_GRADER_BACKEND` defaults to empty (Ollama only) to prevent accidental API costs. Cloud APIs require explicit opt-in.
- `max_tokens=16` for calibration calls, `256` default for general use.
- Calibration golden tests include "haystack" pairs (rule buried in long doc vs. similar doc without the rule) to screen out models that can only do trivial matching.
