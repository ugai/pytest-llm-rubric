"""Default models for each built-in provider.

These are used when PYTEST_LLM_RUBRIC_MODEL is not set.
Override via the environment variable rather than editing this file.
"""

# Chosen for stability (5/5 preflight passes) and multilingual strength.
# Alternatives: gpt-oss:20b (20/20 stable, needs more RAM),
#   nemotron-3-nano:4b (strong IFEval but intermittent empty responses).
OLLAMA_MODEL = "qwen3.5:2b"
ANTHROPIC_MODEL = "claude-haiku-4-5"
OPENAI_MODEL = "gpt-5.4-nano"
