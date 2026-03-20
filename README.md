# pytest-rubric-grader

A pytest plugin for rubric-based LLM-as-judge testing with auto-discovery and calibration.

## Features

- **Auto-discovery** - detects available LLM backends (Ollama, Anthropic, OpenAI)
- **Calibration** - validates the LLM can reliably judge rubric criteria before running tests
- **Fixture-based** - provides a `grader_llm` fixture, overridable in `conftest.py`
- **Model finder** - finds the smallest Ollama model that passes calibration

## Installation

```bash
pip install pytest-rubric-grader
# or
uv add --dev pytest-rubric-grader
```

## Quick Start

```python
def test_document_has_rule(grader_llm):
    document = open("SKILL.md").read()
    result = grader_llm.complete([
        {"role": "system", "content": "Does this document express the criterion? Reply PASS or FAIL."},
        {"role": "user", "content": f"DOCUMENT:\n{document}\n\nCRITERION:\nBugs are prioritized over enhancements."},
    ])
    assert "PASS" in result.upper()
```

The plugin automatically discovers an LLM backend and calibrates it. No configuration needed if Ollama is running locally.

## Configuration

Control the backend via environment variables:

| Variable | Values | Default |
|---|---|---|
| `PYTEST_RUBRIC_GRADER_BACKEND` | `ollama`, `anthropic`, `openai`, `auto`, (empty) | (empty) = Ollama only |
| `PYTEST_RUBRIC_GRADER_MODEL` | Any model name | Provider-specific default |
| `PYTEST_RUBRIC_GRADER_OLLAMA_MODEL` | Ollama model name | `granite4:3b` |
| `PYTEST_RUBRIC_GRADER_ANTHROPIC_MODEL` | Anthropic model name | `claude-haiku-4-5` |
| `PYTEST_RUBRIC_GRADER_OPENAI_MODEL` | OpenAI model name | `gpt-5.4-nano` |
| `PYTEST_RUBRIC_GRADER_SKIP_CALIBRATION` | `1`, `true`, `yes` | (disabled) |

Model resolution order: provider-specific env var > `PYTEST_RUBRIC_GRADER_MODEL` > default in `defaults.py`.

**Backend behavior:**

- **(empty)** - try Ollama only, skip if unavailable (safe default, no API costs)
- **`auto`** - try Ollama, then Anthropic, then OpenAI
- **`ollama`** / **`anthropic`** / **`openai`** - use only the specified backend

## Custom Backend

Override the `grader_llm` fixture in your `conftest.py`:

```python
import pytest

class MyBackend:
    def complete(self, messages, max_tokens=256):
        # Your LLM call here
        return "PASS"

@pytest.fixture(scope="session")
def grader_llm():
    return MyBackend()
```

## Find Best Ollama Model

```bash
uv run python -m pytest_rubric_grader.find_model
```

Runs calibration against all local Ollama models and recommends the smallest one that passes.

## Development

```bash
git clone https://github.com/ugai/pytest-rubric-grader.git
cd pytest-rubric-grader
uv sync
uv run pre-commit install   # ruff + ty on every commit
uv run pytest -m "not integration"
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run ty check src/
```

## License

MIT
