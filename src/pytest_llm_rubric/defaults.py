"""Default models and endpoints for each provider.

Edit these values to change which model is used when PYTEST_LLM_RUBRIC_MODEL
is not set. Each provider falls back to its default listed here.
"""

# Chosen for stability (5/5 calibration passes) and multilingual strength.
# Alternatives: nemotron-3-nano:4b (strong IFEval but intermittent empty responses).
OLLAMA_MODEL = "qwen3.5:2b"
ANTHROPIC_MODEL = "claude-haiku-4-5"
OPENAI_MODEL = "gpt-5.4-nano"
