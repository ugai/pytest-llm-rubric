# pytest-rubric-grader

Pytest plugin for semantic PASS/FAIL checks against text or documents.

## Use When

- Wording varies but meaning must be preserved
- Exact string assertions are too brittle
- Tests need binary semantic judgments: PASS or FAIL

Not a general essay grader or multi-dimensional scoring system.

## Quick Start

### Prerequisites

```bash
pip install pytest-rubric-grader   # or: uv add --dev pytest-rubric-grader
ollama serve                       # start Ollama (if not already running)
ollama pull granite4:3b            # any chat model works
```

### Minimal Test

```python
def test_mentions_deadline(grader_llm):
    text = "The report is due by March 31st."
    response = grader_llm.complete([
        {"role": "system", "content": "Does this text express the criterion? Reply PASS or FAIL."},
        {"role": "user", "content": f"TEXT:\n{text}\n\nCRITERION:\nThe delivery deadline is mentioned."},
    ])
    assert "PASS" in response.upper()
```

## Execution Flow

1. **Discover** — find an available LLM backend (local Ollama by default)
2. **Calibrate** — run 12 golden tests to verify reliable PASS/FAIL judgment
3. **Provide** — expose the `grader_llm` session fixture on success
4. **Skip** — skip dependent tests on backend absence or calibration failure (not fail)

By default, only local Ollama is tried. Paid cloud APIs require explicit opt-in.

## Example: Policy Document Checks

Verify that each policy document semantically expresses required rules.

```python
import pytest
from pathlib import Path

DOCS_DIR = Path("policies")
REQUIRED_RULES = [
    "Personal data must be encrypted at rest",
    "Access logs are retained for at least 90 days",
    "Third-party integrations require security review",
]

@pytest.mark.parametrize("doc", [p.name for p in DOCS_DIR.glob("*.md")])
@pytest.mark.parametrize("rule", REQUIRED_RULES)
def test_policy_expresses_rule(grader_llm, doc, rule):
    text = (DOCS_DIR / doc).read_text()
    response = grader_llm.complete([
        {"role": "system", "content": "Does this document express the criterion? Reply PASS or FAIL."},
        {"role": "user", "content": f"DOCUMENT:\n{text}\n\nCRITERION:\n{rule}"},
    ])
    assert "PASS" in response.upper(), f"{doc} is missing rule: {rule}"
```

## Configuration

| Variable | Values | Default |
|---|---|---|
| `PYTEST_RUBRIC_GRADER_BACKEND` | `ollama`, `anthropic`, `openai`, `auto`, (empty) | (empty) = Ollama only |
| `PYTEST_RUBRIC_GRADER_MODEL` | Any model name | Provider-specific default |
| `PYTEST_RUBRIC_GRADER_OLLAMA_MODEL` | Ollama model name | `granite4:3b` |
| `PYTEST_RUBRIC_GRADER_ANTHROPIC_MODEL` | Anthropic model name | `claude-haiku-4-5` |
| `PYTEST_RUBRIC_GRADER_OPENAI_MODEL` | OpenAI model name | `gpt-5.4-nano` |
| `PYTEST_RUBRIC_GRADER_ANTHROPIC_BASE_URL` | Anthropic endpoint URL | `https://api.anthropic.com/v1` |
| `PYTEST_RUBRIC_GRADER_SKIP_CALIBRATION` | `1`, `true`, `yes` | (disabled) |

Model resolution: provider-specific env var > `PYTEST_RUBRIC_GRADER_MODEL` > default in `defaults.py`.

### Backend Behavior

- **(empty)** — Ollama only. Safe default, no API costs.
- **`auto`** — Ollama → Anthropic → OpenAI (first available)
- **`ollama`** / **`anthropic`** / **`openai`** — use only the specified backend

### CI

Set `PYTEST_RUBRIC_GRADER_BACKEND` and the matching provider credentials in your CI secrets.

```yaml
env:
  PYTEST_RUBRIC_GRADER_BACKEND: openai   # or: anthropic
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Markers

Tests that use the `grader_llm` fixture automatically receive the `rubric_grading` marker.

```bash
pytest -m "not rubric_grading"   # run everything except LLM-graded tests
pytest -m rubric_grading         # run only LLM-graded tests
```

## Custom Backend

Override the fixture for a custom LLM client or internal gateway.

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

## Custom System Prompt

Pass a custom system prompt to `calibrate()` for calibration with your own instructions.

```python
from pytest_rubric_grader.calibration import calibrate, JUDGE_SYSTEM_PROMPT

result = calibrate(llm, system_prompt="Your custom prompt here.")
```

The default `JUDGE_SYSTEM_PROMPT` is used when `system_prompt` is omitted.

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
