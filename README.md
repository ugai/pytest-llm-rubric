# pytest-llm-rubric

[![CI](https://github.com/ugai/pytest-llm-rubric/actions/workflows/ci.yml/badge.svg)](https://github.com/ugai/pytest-llm-rubric/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pytest-llm-rubric)](https://pypi.org/project/pytest-llm-rubric/)

> **Experimental** — this plugin is in early development. APIs may change without notice.

Pytest plugin for LLM-as-a-Judge semantic PASS/FAIL checks.  
Just a thin layer between pytest and your LLM stack.

## Use Cases

Catch semantic regressions in:

- Agent skills: instruction docs still contain rules after edits
- Prompts: LLM output quality hasn't degraded after changes
- Generated docs: auto-generated content includes all required sections
- Translations: specific meanings are preserved across languages

Not a general essay grader or multi-dimensional scoring system.

## Quick Start

Install and configure with a local Ollama model:

<!--pytest.mark.skip-->
```bash
pip install pytest-llm-rubric
ollama pull gpt-oss:20b
export PYTEST_LLM_RUBRIC_MODELS="ollama:gpt-oss:20b"
```

See [Model selection](#model-selection) for other backends.

```python
# test code
def test_semantic_check(judge_llm):
    text = "The quick brown fox jumps over the lazy dog."
    assert judge_llm.judge(text, "Two animals appear in the text.")

    results = [
        judge_llm.judge(text, "A fox leaps over a dog."),
        judge_llm.judge(text, "The dog is beneath the fox."),
    ]
    assert sum(results) / len(results) >= 0.5
```

```bash
# output
$ pytest test_example.py -v
================================= LLM Rubric ==================================
Model: ollama:gpt-oss:20b  Preflight: preflight passed (12/12) in 231.8s
3 passed, 0 failed
```

## How It Works

1. **Discover** - resolve the LLM backend from `PYTEST_LLM_RUBRIC_MODELS`
2. **Preflight** - run a sanity-check to verify the backend can reliably judge PASS/FAIL ([skippable](#skipping-preflight))
3. **Provide** - pass the `judge_llm` fixture to your tests
    - If the backend is unavailable, tests **fail**
    - If preflight fails, tests are **skipped**

## Example: Policy Document Checks

Verify that each policy document expresses required rules.

```python
import pytest
from pathlib import Path
from pytest_llm_rubric import JudgeLLM

POLICY_DOC = Path("docs/policies/data-security.md")
REQUIRED_RULES = [
    "Personal data must be encrypted at rest",
    "Access logs are retained for at least 90 days",
    "Third-party integrations require security review",
]

@pytest.mark.flaky(reruns=2)  # requires `pytest-rerunfailures`
@pytest.mark.parametrize("rule", REQUIRED_RULES)
def test_data_security_policy(judge_llm: JudgeLLM, rule):
    assert judge_llm.judge(POLICY_DOC.read_text(), rule)
```

## Configuration

### Model selection

Set `PYTEST_LLM_RUBRIC_MODELS` to one or more `provider:model` values:

| Value | Description |
| --- | --- |
| `ollama:gpt-oss:20b` | Ollama |
| `anthropic:claude-haiku-4-5` | Requires `ANTHROPIC_API_KEY` [*](#additional-sdk) |
| `openai:gpt-5.4-nano` | Requires `OPENAI_API_KEY` [*](#additional-sdk) |
| `groq:llama-3.3-70b` | Requires `GROQ_API_KEY` [*](#additional-sdk) |
| `ollama:gpt-oss:20b,anthropic:claude-haiku-4-5` | Comma-separated: use first available |
| `auto` | Try the [default model list](src/pytest_llm_rubric/defaults.py) |
| (unset) | Error, unless `llm_rubric_models` is configured in ini |

#### Additional SDK

Cloud providers need their SDK via [any-llm-sdk](https://github.com/mozilla-ai/any-llm): `pip install any-llm-sdk[anthropic]` (or `[openai]`, `[groq]`). Ollama is included by default.

<!--pytest.mark.skip-->
```yaml
# GitHub Actions workflow
env:
  PYTEST_LLM_RUBRIC_MODELS: anthropic:claude-haiku-4-5
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

### Fallback list

Model resolution order: env var `PYTEST_LLM_RUBRIC_MODELS` > ini option `llm_rubric_models`.

> [!IMPORTANT]
> The default list includes cloud providers (Anthropic, OpenAI). If their API keys are set, `auto` may incur API costs. To avoid this, list only providers you intend to use.

<!--pytest.mark.skip-->
```toml
# pyproject.toml
[tool.pytest.ini_options]
llm_rubric_models = [
    "ollama:qwen3.5:9b",
    "anthropic:claude-haiku-4-5",
]
```

### Markers

Tests that use the `judge_llm` fixture automatically receive the `llm_rubric` marker, so you can run or skip them selectively:

<!--pytest.mark.skip-->
```bash
pytest -m llm_rubric        # run only LLM-judged tests
pytest -m "not llm_rubric"  # skip LLM-judged tests
```

### Skipping preflight

Set `PYTEST_LLM_RUBRIC_SKIP_PREFLIGHT=1` to bypass the built-in golden tests.

### Find best local model

A tiny CLI utility that runs preflight against all local Ollama models and recommends the smallest one that passes.

<!--pytest.mark.skip-->
```bash
$ uv run python -m pytest_llm_rubric.find_local_model --base-url http://localhost:11434 gemma4:e2b gemma4:e4b gemma4:26b
Found 3 model(s) in Ollama. Running preflight...

  gemma4:e2b                     ( 6.7GB) ... FAIL (0/12 stopped at 1/12)
  gemma4:e4b                     ( 8.9GB) ... FAIL (0/12 stopped at 1/12)
  gemma4:26b                     (16.8GB) ... PASS (12/12)
Recommended: gemma4:26b (smallest passing model)
```

These tools can also help you find models that fit your hardware:

- [canirun.ai](https://www.canirun.ai/) - browser-based, shows which models fit your hardware
- [llmfit](https://github.com/AlexsJones/llmfit) - CLI tool that scores models by fit, speed, and quality

## Advanced Usage

### `complete()`

`complete()` gives you full control over the LLM interaction: you provide the messages and get back the raw response. Use it when `judge()` is too opinionated.

```python
from pytest_llm_rubric import parse_verdict

def test_custom_prompt(judge_llm):
    response = judge_llm.complete([
        {"role": "system", "content": "You are a compliance auditor. Reply PASS or FAIL."},
        {"role": "user", "content": f"DOCUMENT:\n{POLICY_DOC.read_text()}\n\nRULE:\nPersonal data must be encrypted at rest"},
    ])
    passed = parse_verdict(response) == "PASS"
    judge_llm.record(criterion="encryption at rest", passed=passed)
    assert passed
```

### Custom backend

Override the `judge_llm` fixture for a custom LLM client or internal gateway.

<!--pytest.mark.skip-->
```python
import pytest
import requests
from pytest_llm_rubric import AnyLLMJudge, register_judge

class MyBackend(AnyLLMJudge):
    def complete(self, messages, max_output_tokens=256, response_format=None):
        resp = requests.post("https://internal-llm.corp/v1/chat", json={"messages": messages})
        return resp.json()["content"]

# Override the fixture directly
@pytest.fixture(scope="session")
def judge_llm(request):
    judge = MyBackend("my-model", "internal")
    register_judge(request.config, judge, model="internal:my-model")
    return judge
```

Extend `AnyLLMJudge` and override `complete()`. Call `register_judge()` in your fixture so the terminal summary picks up the results.

### AI coding assistant CLIs as backends

AI coding assistant CLIs like [Claude Code](https://claude.com/product/claude-code/) or [GitHub Copilot](https://github.com/features/copilot/cli/) can also be used as backends without an API key:

<!--pytest.mark.skip-->
```python
import subprocess
from pytest_llm_rubric import AnyLLMJudge

class ClaudeCLIBackend(AnyLLMJudge):
    def complete(self, messages, max_output_tokens=256, response_format=None):
        prompt = messages[-1]["content"]
        result = subprocess.run(
            ["claude", "-p", prompt],  # or ["copilot", "-p", prompt]
            capture_output=True, timeout=300,
        )
        return result.stdout.decode("utf-8")
```

### Parallel execution (pytest-xdist)

Works with [pytest-xdist](https://pypi.org/project/pytest-xdist/). Preflight runs once across workers. Not extensively tested yet, please report issues.

<!--pytest.mark.skip-->
```bash
pip install pytest-xdist
pytest -n auto -m llm_rubric
```

### Flaky tests

LLM-based tests are inherently non-deterministic. Preflight screens out unreliable models, but borderline cases may still flake. Use [pytest-rerunfailures](https://pypi.org/project/pytest-rerunfailures/) to retry:

<!--pytest.mark.skip-->
```bash
pip install pytest-rerunfailures
pytest --reruns 2 -m llm_rubric  # rerun failed LLM tests up to 2 times
```

Deterministic settings (`temperature=0`) would undermine the fuzzy semantic matching that makes this approach valuable. See the [pytest documentation on flaky tests](https://docs.pytest.org/en/stable/explanation/flaky.html) for more strategies.

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

This plugin's design — binary PASS/FAIL criteria, not multi-level scoring — aligns with Anthropic's recommended practices:

- [Define success criteria and build evaluations](https://docs.anthropic.com/en/docs/test-and-evaluate/develop-tests) — binary classification with clear rubrics over qualitative scales
- [Skill authoring best practices](https://docs.anthropic.com/en/docs/agents-and-tools/agent-skills/best-practices) — `expected_behavior` as individually verifiable statements, not a single aggregate score

## License

MIT
