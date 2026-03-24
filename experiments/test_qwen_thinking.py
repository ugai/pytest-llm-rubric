"""Experiment: qwen3.5 thinking mode vs nothink vs higher max_tokens."""

import sys

from openai import OpenAI

# -- Config --
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://192.168.3.2:11434"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.5:9b"

SYSTEM = (
    'You are a rubric grader. You will be given a DOCUMENT and a CRITERION.\n'
    'Determine whether the document expresses the criterion.\n'
    'Respond with a single word: "PASS" or "FAIL".\n'
    'Your response must be exactly one word. Do not explain.'
)

# 2 easy test cases: one PASS, one FAIL
CASES = [
    {
        "doc": "Agents must prioritize bug issues over enhancement issues.",
        "criterion": "Bug issues are prioritized over enhancement issues.",
        "expected": "PASS",
    },
    {
        "doc": "The system supports dark mode and light mode themes.",
        "criterion": "Bug issues are prioritized over enhancement issues.",
        "expected": "FAIL",
    },
]

client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="ollama", timeout=120.0)


def run_test(label: str, *, max_tokens: int, nothink: bool) -> None:
    print(f"\n=== {label} (max_tokens={max_tokens}, nothink={nothink}) ===")
    system = SYSTEM
    if nothink:
        system = "/nothink\n" + SYSTEM

    for case in CASES:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"DOCUMENT:\n{case['doc']}\n\nCRITERION:\n{case['criterion']}"},
        ]
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            # Check if there's reasoning_content (thinking)
            msg = resp.choices[0].message
            thinking = getattr(msg, "reasoning_content", None) or ""
            print(f"  expected={case['expected']:<4}  content={content!r}")
            if thinking:
                print(f"    thinking={thinking[:120]!r}...")
        except Exception as e:
            print(f"  expected={case['expected']:<4}  ERROR: {e}")


# Test 1: Default (thinking enabled, max_tokens=512) — matches preflight
run_test("default (thinking, 512 tokens)", max_tokens=512, nothink=False)

# Test 2: Nothink + 512 tokens
run_test("nothink + 512 tokens", max_tokens=512, nothink=True)
