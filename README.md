# pytest-llm-rubric

[![CI](https://github.com/ugai/pytest-llm-rubric/actions/workflows/ci.yml/badge.svg)](https://github.com/ugai/pytest-llm-rubric/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pytest-llm-rubric)](https://pypi.org/project/pytest-llm-rubric/)

> **Experimental** — this plugin is in early development. APIs may change without notice.

Minimal pytest plugin for LLM-as-a-Judge — simple semantic PASS/FAIL checks against text or documents.

## Why pytest?

Your CI already runs pytest. Semantic text checks shouldn't need a separate framework. Just another test file.

## Use When

- Wording varies but meaning must be preserved
- Exact string assertions are too brittle
- Tests need binary semantic judgments: PASS or FAIL

e.g.

- Agent skill regression — instruction docs still contain required rules after edits
- Prompt regression — LLM output quality hasn't degraded after prompt changes
- Doc generation CI — auto-generated docs include all required sections
- Translation fidelity — specific meanings are preserved across languages

Not a general essay grader or multi-dimensional scoring system.

## Quick Start

### Prerequisites

<!--pytest.mark.skip-->
```bash
pip install pytest-llm-rubric  # or: uv add --dev pytest-llm-rubric
ollama serve                   # start Ollama (if not already running)
ollama pull granite4:3b        # any chat model works
```

### Minimal Test

```python
def test_mentions_deadline(judge_llm):
    # In practice, text is usually much longer —
    # policy docs, generated reports, LLM outputs, etc.
    text = "The report is due by March 31st."
    criterion = "The delivery deadline is mentioned."
    response = judge_llm.complete([
        {"role": "system", "content": "Does this text express the criterion? Reply PASS or FAIL."},
        {"role": "user", "content": f"TEXT:\n{text}\n\nCRITERION:\n{criterion}"},
    ])
    assert "PASS" in response.upper()
```

## Execution Flow

1. **Discover** — find an available LLM backend (local Ollama by default)
2. **Calibrate** — run 12 golden tests to verify reliable PASS/FAIL judgment (skippable)
3. **Provide** — expose the `judge_llm` session fixture on success
4. **Skip** — skip dependent tests on backend absence or calibration failure (not fail)

By default, only local Ollama is tried. Paid cloud APIs require explicit opt-in.

## Example: Policy Document Checks

Verify that each policy document semantically expresses required rules.

```python
import pytest
from pathlib import Path
from pytest_llm_rubric import JudgeLLM

DOCS_DIR = Path("policies")
POLICY_DOCS = sorted(DOCS_DIR.rglob("*.md"))
REQUIRED_RULES = [
    "Personal data must be encrypted at rest",
    "Access logs are retained for at least 90 days",
    "Third-party integrations require security review",
]

@pytest.mark.parametrize("doc", POLICY_DOCS)
@pytest.mark.parametrize("rule", REQUIRED_RULES)
def test_policy_expresses_rule(judge_llm: JudgeLLM, doc, rule):
    text = doc.read_text()
    response = judge_llm.complete([
        {"role": "system", "content": "Does this document express the criterion? Reply PASS or FAIL."},
        {"role": "user", "content": f"DOCUMENT:\n{text}\n\nCRITERION:\n{rule}"},
    ])
    assert "PASS" in response.upper(), f"{doc} is missing rule: {rule}"
```

## Configuration

| Variable | Default |
|---|---|
| `PYTEST_LLM_RUBRIC_BACKEND` | (empty) = Ollama only. `ollama`, `anthropic`, `openai`, `auto` |
| `PYTEST_LLM_RUBRIC_MODEL` | Provider-specific default |
| `PYTEST_LLM_RUBRIC_<PROVIDER>_MODEL` | Overrides `MODEL` per provider |
| `PYTEST_LLM_RUBRIC_SKIP_CALIBRATION` | (disabled) |

Model resolution: `<PROVIDER>_MODEL` > `MODEL` > default in [`defaults.py`](src/pytest_llm_rubric/defaults.py).

### Backend Behavior

- **(empty)** — Ollama only. Safe default, no API costs.
- **`auto`** — Ollama → Anthropic → OpenAI (first available)
- **`ollama`** / **`anthropic`** / **`openai`** — use only the specified backend

If no backend is available or calibration fails, dependent tests are skipped (not failed).

### CI

Set `PYTEST_LLM_RUBRIC_BACKEND` and the matching provider credentials in your CI secrets.

<!--pytest.mark.skip-->
```yaml
env:
  PYTEST_LLM_RUBRIC_BACKEND: openai  # or: anthropic
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Markers

Tests that use the `judge_llm` fixture automatically receive the `llm_rubric` marker.

<!--pytest.mark.skip-->
```bash
pytest -m "not llm_rubric"  # run everything except LLM-judged tests
pytest -m llm_rubric        # run only LLM-judged tests
```

## Custom Backend

Override the fixture for a custom LLM client or internal gateway.

```python
import pytest

class MyBackend:
    def complete(self, messages, max_tokens=256):
        # Call your internal LLM gateway
        resp = requests.post("https://internal-llm.corp/v1/chat", json={"messages": messages})
        return resp.json()["content"]

@pytest.fixture(scope="session")
def judge_llm():
    return MyBackend()
```

## Custom System Prompt

Tweak the calibration system prompt if your model needs specific instructions to pass calibration.

<!--pytest.mark.skip-->
```python
from pytest_llm_rubric.calibration import calibrate, JUDGE_SYSTEM_PROMPT

result = calibrate(llm, system_prompt="Your custom prompt here.")
```

The default `JUDGE_SYSTEM_PROMPT` is used when `system_prompt` is omitted.

## Find Best Local Model

<!--pytest.mark.skip-->
```bash
uv run python -m pytest_llm_rubric.find_local_model
```

Runs calibration against all local Ollama models and recommends the smallest one that passes.

## Development

<!--pytest.mark.skip-->
```bash
git clone https://github.com/ugai/pytest-llm-rubric.git
cd pytest-llm-rubric
uv sync
uv run pre-commit install           # ruff + ty on every commit
uv run pytest -m "not integration"  # no LLM calls, runs offline
uv run ruff check src/ tests/
uv run ruff format src/ tests/
uv run ty check src/
```

## References

This plugin's design — decomposing evaluation into multiple binary PASS/FAIL criteria instead of multi-level scoring — aligns with Anthropic's recommended practices:

- **[Define success criteria and build evaluations](https://docs.anthropic.com/en/docs/test-and-evaluate/develop-tests)** — LLM-based grading section recommends binary classification (`"correct"` / `"incorrect"`) with clear rubrics over qualitative scales.
- **[Skill authoring best practices](https://docs.anthropic.com/en/docs/agents-and-tools/agent-skills/best-practices)** — Evaluation-driven development section structures `expected_behavior` as an array of individually verifiable statements, not a single aggregate score.

## License

MIT
