"""Stability test: run preflight N times per model and report flaky tests."""

from __future__ import annotations

import argparse
import os
import sys

from pytest_llm_rubric.preflight import preflight
from pytest_llm_rubric.plugin import AnyLLMJudge
from pytest_llm_rubric.utils import parse_ollama_host


def main() -> None:
    parser = argparse.ArgumentParser(description="Stability test for preflight")
    parser.add_argument("models", nargs="+", help="Model name(s) to test")
    parser.add_argument("-n", "--runs", type=int, default=5, help="Number of runs per model")
    parser.add_argument(
        "--base-url",
        default=parse_ollama_host(os.environ.get("OLLAMA_HOST")),
    )
    args = parser.parse_args()

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"Model: {model_name} ({args.runs} runs)")
        print(f"{'='*60}")

        # Track per-test failure counts
        test_failures: dict[int, list[dict]] = {}

        for run in range(1, args.runs + 1):
            judge = AnyLLMJudge(model_name, "ollama", api_base=args.base_url)
            result = preflight(judge)
            status = "PASS" if result.passed else "FAIL"
            print(f"  Run {run}: {status} ({result.correct}/{result.total})", end="")

            if not result.passed:
                for i, d in enumerate(result.details):
                    if not d["correct"]:
                        if i not in test_failures:
                            test_failures[i] = []
                        test_failures[i].append({
                            "run": run,
                            "expected": d["expected"],
                            "actual": d["actual"],
                            "raw": d.get("raw_response", ""),
                            "criterion": d["criterion"],
                        })
                        print(f"  [test {i}: expected={d['expected']}, raw={d.get('raw_response', '')!r}]", end="")
            print()

        if test_failures:
            print(f"\n  Flaky tests for {model_name}:")
            for idx, failures in sorted(test_failures.items()):
                f0 = failures[0]
                print(f"    Test {idx} ({f0['criterion'][:60]}...)")
                print(f"      Expected: {f0['expected']}, failed {len(failures)}/{args.runs} times")
                for f in failures:
                    print(f"      Run {f['run']}: raw={f['raw']!r}")
        else:
            print(f"\n  {model_name}: stable ({args.runs}/{args.runs} passed)")


if __name__ == "__main__":
    main()
