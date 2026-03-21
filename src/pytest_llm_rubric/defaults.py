"""Default models and endpoints for each provider.

Edit these values to change which model is used when PYTEST_LLM_RUBRIC_MODEL
is not set. Each provider falls back to its default listed here.
"""

# For multilingual rubrics, qwen3.5:2b may work better.
OLLAMA_MODEL = "nemotron-3-nano:4b"
ANTHROPIC_MODEL = "claude-haiku-4-5"
OPENAI_MODEL = "gpt-5.4-nano"
