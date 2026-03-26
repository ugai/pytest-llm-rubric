"""Default models for each built-in provider.

These are used when PYTEST_LLM_RUBRIC_MODEL is not set.
Override via the environment variable rather than editing this file.
"""

# Chosen for speed and stability (MoE, 20/20 preflight stable).
# Alternative: qwen3.5:9b (smaller VRAM footprint, also stable).
OLLAMA_MODEL = "gpt-oss:20b"
ANTHROPIC_MODEL = "claude-haiku-4-5"
OPENAI_MODEL = "gpt-5.4-nano"
