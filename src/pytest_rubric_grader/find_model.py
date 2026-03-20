"""Find the best (smallest passing) Ollama model for rubric grading.

Usage:
    uv run python -m pytest_rubric_grader.find_model
    uv run python -m pytest_rubric_grader.find_model --base-url http://host:11434
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx
from openai import OpenAI

from pytest_rubric_grader.calibration import calibrate
from pytest_rubric_grader.plugin import OpenAICompatibleGrader


def _get_ollama_models(base_url: str) -> list[dict]:
    resp = httpx.get(f"{base_url}/api/tags", timeout=5)
    resp.raise_for_status()
    return resp.json().get("models", [])


def _size_label(size_bytes: int) -> str:
    gb = size_bytes / (1024**3)
    if gb >= 1:
        return f"{gb:.1f}GB"
    mb = size_bytes / (1024**2)
    return f"{mb:.0f}MB"


def find_best_model(base_url: str = "http://localhost:11434") -> None:
    try:
        models = _get_ollama_models(base_url)
    except Exception as e:
        print(f"Could not connect to Ollama at {base_url}: {e}")
        sys.exit(1)

    if not models:
        print("No models found in Ollama.")
        sys.exit(1)

    # Sort by size ascending (smallest first)
    models.sort(key=lambda m: m.get("size", 0))

    # Filter out non-generative models
    skip_patterns = ("embed", "vision", "clip", "whisper")
    models = [m for m in models if not any(p in m["name"].lower() for p in skip_patterns)]

    print(f"Found {len(models)} model(s) in Ollama. Running calibration...\n")

    results = []
    recommended = None

    for model in models:
        name = model["name"]
        size = _size_label(model.get("size", 0))
        print(f"  {name:<30} ({size:>6}) ... ", end="", flush=True)

        client = OpenAI(base_url=f"{base_url}/v1", api_key="ollama")
        grader = OpenAICompatibleGrader(client, name)

        try:
            result = calibrate(grader)
            status = f"{'PASS' if result.passed else 'FAIL'} ({result.correct}/{result.total})"
            print(status)
            results.append({"name": name, "size": size, "result": result})
            if result.passed and recommended is None:
                recommended = name
        except Exception as e:
            print(f"ERROR: {e}")

    print()
    if recommended:
        print(f"Recommended: {recommended} (smallest passing model)")
        print("\nSet in defaults.py or environment:")
        print(f"  PYTEST_RUBRIC_GRADER_MODEL={recommended}")
    else:
        print("No model passed calibration.")
        print("Consider pulling a larger model: ollama pull granite4:3b")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the best Ollama model for rubric grading")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama base URL (default: $OLLAMA_HOST or http://localhost:11434)",
    )
    args = parser.parse_args()
    find_best_model(args.base_url)
