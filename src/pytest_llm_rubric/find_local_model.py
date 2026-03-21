"""Find the best (smallest passing) local model for rubric grading.

Usage:
    uv run python -m pytest_llm_rubric.find_local_model
    uv run python -m pytest_llm_rubric.find_local_model qwen3.5:27b
    uv run python -m pytest_llm_rubric.find_local_model granite4:3b lfm2:24b --verbose
    uv run python -m pytest_llm_rubric.find_local_model --base-url http://host:11434
"""

from __future__ import annotations

import argparse
import os
import sys

import httpx

from pytest_llm_rubric.calibration import calibrate
from pytest_llm_rubric.plugin import AnyLLMJudge
from pytest_llm_rubric.utils import OLLAMA_DEFAULT_HOST, OLLAMA_DEFAULT_PORT, parse_ollama_host


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


def find_best_local_model(
    base_url: str | None = None,
    *,
    verbose: bool = False,
    model_names: list[str] | None = None,
) -> None:
    if base_url is None:
        base_url = parse_ollama_host(os.environ.get("OLLAMA_HOST"))
    try:
        all_models = _get_ollama_models(base_url)
    except Exception as e:
        print(f"Could not connect to Ollama at {base_url}: {e}")
        sys.exit(1)

    if not all_models:
        print("No models found in Ollama.")
        sys.exit(1)

    available = {m["name"]: m for m in all_models}

    if model_names:
        # Use only the specified models, in the order given
        models = []
        for name in model_names:
            if name in available:
                models.append(available[name])
            else:
                avail = ", ".join(sorted(available))
                print(f"Model {name!r} not found in Ollama. Available: {avail}")
                sys.exit(1)
    else:
        # Sort by size ascending (smallest first)
        models = sorted(all_models, key=lambda m: m.get("size", 0))

        # Filter out non-generative models
        skip_patterns = ("embed", "vision", "clip", "whisper")
        all_count = len(models)
        models = [m for m in models if not any(p in m["name"].lower() for p in skip_patterns)]
        skipped = all_count - len(models)
        if skipped:
            print(f"Skipped {skipped} non-generative model(s) (embedding/vision/etc.).")

    print(f"Found {len(models)} model(s) in Ollama. Running calibration...\n")

    results = []
    recommended = None

    for model in models:
        name = model["name"]
        size = _size_label(model.get("size", 0))
        print(f"  {name:<30} ({size:>6}) ... ", end="", flush=True)

        judge = AnyLLMJudge(name, "ollama", api_base=base_url)

        try:
            result = calibrate(judge)
            label = "PASS" if result.passed else "FAIL"
            tested = len(result.details)
            early = f" stopped at {tested}/{result.total}" if result.stopped_early else ""
            status = f"{label} ({result.correct}/{result.total}{early})"
            print(status)
            results.append({"name": name, "size": size, "result": result})
            if result.passed and recommended is None:
                recommended = name
            if verbose:
                for d in result.details:
                    mark = "OK" if d["correct"] else "NG"
                    raw = d.get("raw_response", "")
                    print(f"    [{mark}] expected={d['expected']:<4}  raw={raw!r}")
        except Exception as e:
            print(f"ERROR: {e}")

    print()
    if recommended:
        print(f"Recommended: {recommended} (smallest passing model)")
        print("\nSet in defaults.py or environment:")
        print(f"  PYTEST_LLM_RUBRIC_MODEL={recommended}")
    else:
        print("No model passed calibration.")
        print("Consider pulling a larger model: ollama pull granite4:3b")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the best local model for rubric grading")
    parser.add_argument(
        "models",
        nargs="*",
        help="Specific model name(s) to test (default: all available models)",
    )
    parser.add_argument(
        "--base-url",
        default=parse_ollama_host(os.environ.get("OLLAMA_HOST")),
        help=(
            f"Ollama base URL (default: $OLLAMA_HOST or "
            f"http://{OLLAMA_DEFAULT_HOST}:{OLLAMA_DEFAULT_PORT})"
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show raw LLM responses for each calibration test",
    )
    args = parser.parse_args()
    find_best_local_model(
        args.base_url,
        verbose=args.verbose,
        model_names=args.models or None,
    )
