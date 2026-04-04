"""Default model list for ``auto`` discovery.

Each entry is a ``provider:model`` string tried in order.
The first backend that is reachable wins.

To customise, set ``PYTEST_LLM_RUBRIC_MODELS`` (env var) or
``llm_rubric_models`` (pyproject.toml) instead of editing this file.
"""

AUTO_MODELS: list[str] = [
    "ollama:gpt-oss:20b",
    "anthropic:claude-haiku-4-5",
    "openai:gpt-5.4-nano",
]
