"""Test structured output for calibration judgments via any-llm.

Compares free-text vs structured output (Pydantic response_format) across
Ollama models to see if structured output improves reliability.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Literal, cast

from pydantic import BaseModel

from pytest_llm_rubric.calibration import JUDGE_SYSTEM_PROMPT
from pytest_llm_rubric.golden_tests import GOLDEN_TESTS
from pytest_llm_rubric.utils import parse_ollama_host


class Verdict(BaseModel):
    result: Literal["PASS", "FAIL"]


def run_single_test(
    model: str,
    base_url: str,
    test: dict[str, Any],
    *,
    use_structured: bool,
) -> dict[str, Any]:
    """Run one golden test and return the result."""
    from any_llm import completion
    from any_llm.types.completion import ChatCompletion, ParsedChatCompletion

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"DOCUMENT:\n{test['document']}\n\nCRITERION:\n{test['criterion']}",
        },
    ]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "provider": "ollama",
        "api_base": base_url,
        "max_tokens": 16,
        "reasoning_effort": "none",
        "stream": False,
    }
    if use_structured:
        kwargs["response_format"] = Verdict

    try:
        response = completion(**kwargs)
        if use_structured:
            parsed = cast(ParsedChatCompletion[Verdict], response)
            verdict_obj = parsed.choices[0].message.parsed
            if verdict_obj is not None:
                raw = parsed.choices[0].message.content or ""
                return {
                    "verdict": verdict_obj.result,
                    "raw": raw,
                    "error": None,
                }
            raw = parsed.choices[0].message.content or ""
            return {"verdict": f"PARSE_FAIL", "raw": raw, "error": None}
        else:
            resp = cast(ChatCompletion, response)
            raw = resp.choices[0].message.content or ""
            # Simple extraction matching calibration.py logic
            import re

            m = re.match(r"\W*(PASS|FAIL)\b", raw.upper())
            verdict = m.group(1) if m else f"INVALID: {raw[:50]}"
            return {"verdict": verdict, "raw": raw, "error": None}
    except Exception as e:
        return {"verdict": "ERROR", "raw": "", "error": str(e)}


def run_model(model: str, base_url: str, *, use_structured: bool) -> list[dict[str, Any]]:
    """Run all golden tests for a model."""
    results = []
    for i, test in enumerate(GOLDEN_TESTS):
        r = run_single_test(model, base_url, test, use_structured=use_structured)
        r["index"] = i
        r["expected"] = test["expected"]
        r["correct"] = r["verdict"] == test["expected"]
        r["criterion"] = test["criterion"][:60]
        results.append(r)
    return results


def print_results(model: str, mode: str, results: list[dict[str, Any]]) -> None:
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    print(f"  {mode}: {correct}/{total}", end="")
    failures = [r for r in results if not r["correct"]]
    if failures:
        for f in failures:
            err = f", error={f['error']}" if f['error'] else ""
            print(f"\n    test {f['index']}: expected={f['expected']}, got={f['verdict']}, raw={f['raw']!r}{err}", end="")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test structured output for calibration")
    parser.add_argument("models", nargs="+", help="Model name(s) to test")
    parser.add_argument("-n", "--runs", type=int, default=1, help="Number of runs per model")
    parser.add_argument(
        "--base-url",
        default=parse_ollama_host(os.environ.get("OLLAMA_HOST")),
    )
    args = parser.parse_args()

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({args.runs} run(s))")
        print(f"{'='*60}")

        for run in range(1, args.runs + 1):
            if args.runs > 1:
                print(f"\n  --- Run {run} ---")

            freetext_results = run_model(model_name, args.base_url, use_structured=False)
            print_results(model_name, "free-text ", freetext_results)

            structured_results = run_model(model_name, args.base_url, use_structured=True)
            print_results(model_name, "structured", structured_results)


if __name__ == "__main__":
    main()
