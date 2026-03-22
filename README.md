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
pip install pytest-llm-rubric          # or: uv add --dev pytest-llm-rubric
ollama serve                           # start Ollama (if not already running)
ollama pull gpt-oss:20b               # any chat model works
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

1. **Discover** — auto-detect available backends based on installed extras and env vars
2. **Preflight** — verify the discovered backend can reliably judge PASS/FAIL before exposing it as `judge_llm` (skippable)
3. **Provide, skip, or fail** — expose the `judge_llm` session fixture on success. If the default (empty) backend is unavailable or preflight fails, dependent tests are skipped. If an explicit backend is unavailable, tests fail

Paid cloud APIs never run unless explicitly configured.

## Example: Policy Document Checks

Verify that each policy document semantically expresses required rules.

```python
import pytest
from pathlib import Path
from pytest_llm_rubric import JudgeLLM

POLICY_DOCS = sorted(Path("docs/policies").rglob("*.md"))
REQUIRED_RULES = [
    "Personal data must be encrypted at rest",
    "Access logs are retained for at least 90 days",
    "Third-party integrations require security review",
]

# @pytest.mark.flaky(reruns=2)  # requires `pytest-rerunfailures` (recommended)
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

All configuration is through environment variables.

### Backend selection

Cloud backends require their corresponding extra:

| `PYTEST_LLM_RUBRIC_BACKEND` | Extra | API key |
|---|---|---|
| (empty) / `ollama` | — (included) | — |
| `anthropic` | `[anthropic]` | `ANTHROPIC_API_KEY` |
| `openai` | `[openai]` | `OPENAI_API_KEY` |
| `auto` | any of the above | — |

`auto` tries Ollama → Anthropic → OpenAI, using the first available.
If the default (empty) backend is unavailable or preflight fails, dependent tests are skipped. If an explicit backend (`ollama`, `anthropic`, `openai`, `auto`) is set but unavailable, tests **fail** to surface CI misconfigurations.

CI example:

<!--pytest.mark.skip-->
```yaml
env:
  PYTEST_LLM_RUBRIC_BACKEND: openai  # or: anthropic
  PYTEST_LLM_RUBRIC_OPENAI_MODEL: gpt-5.4-mini
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

### Model selection

Override the default model per provider with `PYTEST_LLM_RUBRIC_<PROVIDER>_MODEL` (e.g. `PYTEST_LLM_RUBRIC_OLLAMA_MODEL=gpt-oss:20b`). Defaults are in [`defaults.py`](src/pytest_llm_rubric/defaults.py).

### Skipping preflight

Set `PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT=1` to bypass the built-in golden tests.

## Markers

Tests that use the `judge_llm` fixture automatically receive the `llm_rubric` marker.

<!--pytest.mark.skip-->
```bash
pytest -m "not llm_rubric"  # run everything except LLM-judged tests
pytest -m llm_rubric        # run only LLM-judged tests
```

## Flaky test mitigation

LLM-based tests are inherently non-deterministic — the same input may produce different judgments across runs. This is a feature, not a bug: deterministic settings (`temperature=0`) would undermine the fuzzy semantic matching that makes this approach valuable.

Preflight screens out models that are too unreliable, but borderline cases may still produce occasional flaky results. Rather than fighting non-determinism, use pytest's existing ecosystem:

<!--pytest.mark.skip-->
```bash
pip install pytest-rerunfailures
pytest --reruns 2 -m llm_rubric  # rerun failed LLM tests up to 2 times
```

See the [pytest documentation on flaky tests](https://docs.pytest.org/en/stable/explanation/flaky.html) for more strategies.

## Customization

### Custom backend

Override the `judge_llm` fixture for a custom LLM client or internal gateway.

```python
import pytest

class MyBackend:
    def complete(self, messages, max_output_tokens=256, response_format=None):
        # Call your internal LLM gateway
        resp = requests.post("https://internal-llm.corp/v1/chat", json={"messages": messages})
        return resp.json()["content"]

@pytest.fixture(scope="session")
def judge_llm():
    return MyBackend()
```

### Custom system prompt

Tweak the preflight system prompt if your model needs specific instructions to pass preflight.

<!--pytest.mark.skip-->
```python
from pytest_llm_rubric.preflight import preflight, JUDGE_SYSTEM_PROMPT

result = preflight(llm, system_prompt="Your custom prompt here.")
```

The default `JUDGE_SYSTEM_PROMPT` is used when `system_prompt` is omitted.

## Find Best Local Model

<!--pytest.mark.skip-->
```bash
uv run python -m pytest_llm_rubric.find_local_model
```

Runs preflight against all local Ollama models and recommends the smallest one that passes.

Not sure which models to pull? These tools help you find models that fit your hardware:

- [canirun.ai](https://www.canirun.ai/) — browser-based hardware detection, shows which models and quantization levels your machine can handle
- [llmfit](https://github.com/AlexsJones/llmfit) — CLI tool that scores models by fit, speed, and quality for your specific GPU/RAM

## Development

<!--pytest.mark.skip-->
```bash
git clone https://github.com/ugai/pytest-llm-rubric.git
cd pytest-llm-rubric
uv sync --extra ollama
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
