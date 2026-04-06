# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                  # Install dependencies (editable mode)
uv run pre-commit install                # Install pre-commit hooks (ruff + ty)
uv run pytest -m "not integration"       # Run tests (no LLM calls)
uv run pytest -m integration             # Run integration tests (requires Ollama / API keys)
uv run pytest tests/test_plugin.py::TestMakeJudge -v  # Run a single test class
uv run ruff check src/ tests/            # Lint
uv run ruff check --fix src/ tests/      # Lint with auto-fix
uv run ruff format src/ tests/           # Format
uv run ty check src/                     # Type check (required — py.typed marker is present)
uv run python -m pytest_llm_rubric.find_local_model  # Find best local model
uv run git-cliff --output CHANGELOG.md   # Generate changelog (then review/edit before commit)
```

## Development Workflow

- Always run `uv run pre-commit install` after `uv sync`.
  Pre-commit hooks automatically run ruff check / ruff format / ty check / pytest.
- CI (GitHub Actions) runs the same checks, so PRs are blocked even without local hooks.

## Release Workflow

1. Bump `version` in `pyproject.toml`
2. Run `uv run git-cliff --output CHANGELOG.md` to generate the changelog
3. Review and polish the generated CHANGELOG.md (optionally with an LLM)
4. Commit and tag:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "release: vX.Y.Z"
   git tag vX.Y.Z
   git push && git push --tags
   ```
5. CI detects the tag and publishes to PyPI

## Architecture

This is a pytest plugin (`pytest11` entry point) that provides `judge_llm`, a session-scoped fixture for rubric-based LLM-as-judge testing.

**Core pipeline: discover → preflight → judge**

- **`plugin.py`** — Entry point. Defines the `JudgeLLM` Protocol and `AnyLLMJudge` implementation. The `judge_llm` fixture auto-discovers a backend, runs preflight, and returns it. Users override the fixture in their `conftest.py` for custom backends.

- **`preflight.py`** — Golden test suite (12 pairs: 6 short-form + 6 haystack) that validates whether an LLM can reliably do binary PASS/FAIL semantic judgments. Session runs preflight once; if the LLM fails, all rubric tests skip.

- **`defaults.py`** — `AUTO_MODELS` list of `provider:model` strings tried in order when `PYTEST_LLM_RUBRIC_MODELS=auto`. Intended to be human-editable.

- **`find_local_model.py`** — CLI tool that runs preflight against all local models (currently Ollama) and recommends the smallest passing one.

**Model selection** is controlled by a single env var `PYTEST_LLM_RUBRIC_MODELS`:
- `provider:model` (e.g. `anthropic:claude-haiku-4-5`, `ollama:qwen3.5:9b`) — direct
- Comma-separated list (e.g. `ollama:gpt-oss:20b,anthropic:claude-haiku-4-5`) — try each in order
- `auto` — tries each entry in `defaults.AUTO_MODELS` in order
- Unset — error (explicit configuration required)

The `provider:model` syntax follows the any-llm-sdk convention (colon separator). The prefix before the first colon must match a known provider (built-in: `ollama`, `anthropic`, `openai`; extended via any-llm's `LLMProvider` enum for `groq`, `mistral`, etc.).

## Key Design Decisions

- The `judge_llm` fixture is `scope="session"` — preflight runs once per test session.
- `PYTEST_LLM_RUBRIC_MODELS` must be explicitly set — no silent defaults to prevent accidental API costs.
- `max_tokens=512` for preflight calls (accommodates thinking models), `256` default for general use.
- Preflight golden tests include "haystack" pairs (rule buried in long doc vs. similar doc without the rule) to screen out models that can only do trivial matching.
