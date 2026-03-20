"""Default models and endpoints for each provider.

Edit these values to change which model is used when PYTEST_LLM_RUBRIC_MODEL
is not set. Each provider falls back to its default listed here.
"""

OLLAMA_MODEL = "granite4:3b"
ANTHROPIC_MODEL = "claude-haiku-4-5"
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
OPENAI_MODEL = "gpt-5.4-nano"
