"""Calibration: verify an LLM backend can reliably judge rubric criteria.

Runs a small set of golden test pairs (known pass/fail) against the backend.
If the backend fails to match expected verdicts, it is considered unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pytest_rubric_grader.plugin import GraderLLM

JUDGE_SYSTEM_PROMPT = """\
You are a rubric grader. You will be given a DOCUMENT and a CRITERION.
Determine whether the document expresses the criterion.
Respond with a single word: "PASS" or "FAIL".
Your response must be exactly one word. Do not explain."""

# ---------------------------------------------------------------------------
# Synthetic haystacks — resembles real SKILL.md structure
# ---------------------------------------------------------------------------

_HAYSTACK_WITH_PRIORITY = """\
# Task Runner

Run a complete task flow inside an isolated git worktree.

## Task Selection

**Eligibility** — ALL must be true:

1. Has `ready` label
2. State is `open`
3. Not assigned

**Priority**: `bug` > `enhancement`, then `p0` > `p1` > `p2` > `p3`
(no label = p2), then lowest issue number.

## Worktree Setup

Create an isolated worktree from the main branch:

```bash
git fetch origin main
git worktree add .worktrees/task-<N> -b task/<N> origin/main
```

## Implementation

Follow the project's coding standards and conventions.
Update documentation to reflect changes.

## Verification

All project checks must pass before pushing.

## Pull Request

Write the PR body to a temp file to avoid shell-escaping issues.
PR body should include: overview, changes, and testing status.
Do not merge. Notify the approver that the PR is ready.
"""

_HAYSTACK_WITHOUT_PRIORITY = """\
# Task Runner

Run a complete task flow inside an isolated git worktree.

## Task Selection

**Eligibility** — ALL must be true:

1. Has `ready` label
2. State is `open`
3. Not assigned

Pick the first eligible task by issue number.

## Worktree Setup

Create an isolated worktree from the main branch:

```bash
git fetch origin main
git worktree add .worktrees/task-<N> -b task/<N> origin/main
```

## Implementation

Follow the project's coding standards and conventions.
Update documentation to reflect changes.

## Verification

All project checks must pass before pushing.

## Pull Request

Write the PR body to a temp file to avoid shell-escaping issues.
PR body should include: overview, changes, and testing status.
Do not merge. Notify the approver that the PR is ready.
"""

_HAYSTACK_WITH_LABEL_PROTOCOL = """\
# Issue Scout

Scan the codebase for potential improvements and open issues.

## Scanning

Look for code smells, missing tests, outdated dependencies,
and documentation gaps. Produce a ranked list of candidates.

## Issue Creation

When creating new issues, the scout MUST:

1. Add the `agent:proposed` label to every issue it creates
2. Never add `agent:ready` — that label is reserved for human maintainers

Include a clear title, description, and reproduction steps where applicable.

## Reporting

Summarize findings in a comment on the tracking issue.
Include severity assessment and recommended priority.
"""

_HAYSTACK_WITHOUT_LABEL_PROTOCOL = """\
# Issue Scout

Scan the codebase for potential improvements and open issues.

## Scanning

Look for code smells, missing tests, outdated dependencies,
and documentation gaps. Produce a ranked list of candidates.

## Issue Creation

Create issues with clear titles and descriptions.
Include reproduction steps where applicable.
Tag issues with appropriate component labels.

## Reporting

Summarize findings in a comment on the tracking issue.
Include severity assessment and recommended priority.
"""

_HAYSTACK_WITH_PICKUP = """\
# Code Fixer

Pick up and resolve reported issues.

## Issue Selection

Choose the highest-priority unassigned issue from the ready queue.

## Claiming

Before writing any code, the fixer must:
- Assign itself to the issue
- Post a comment announcing that work has started

This prevents race conditions with other agents.

## Implementation

Read the issue description carefully. Reproduce the problem first,
then implement the fix. Write tests to cover the fixed behavior.

## Review

Run the full test suite and linter before submitting.
"""

_HAYSTACK_WITHOUT_PICKUP = """\
# Code Fixer

Pick up and resolve reported issues.

## Issue Selection

Choose the highest-priority unassigned issue from the ready queue.

## Implementation

Read the issue description carefully. Reproduce the problem first,
then implement the fix. Write tests to cover the fixed behavior.

## Review

Run the full test suite and linter before submitting.
"""

# ---------------------------------------------------------------------------
# Golden tests
# ---------------------------------------------------------------------------

GOLDEN_TESTS: list[dict] = [
    # === Short-form: basic semantic matching ===
    # --- Expected PASS ---
    {
        "document": "Agents must prioritize bug issues over enhancement issues.",
        "criterion": "Bug issues are prioritized over enhancement issues.",
        "expected": "PASS",
    },
    {
        "document": (
            "Before writing any code, the agent self-assigns the issue "
            "and posts a comment announcing work has started."
        ),
        "criterion": "The agent must self-assign and comment before coding.",
        "expected": "PASS",
    },
    {
        "document": (
            "Issues labelled agent:proposed were created by automation. "
            "They are not approved for implementation until a maintainer "
            "adds the agent:ready label."
        ),
        "criterion": "Agent-created issues receive the agent:proposed label.",
        "expected": "PASS",
    },
    # --- Expected FAIL ---
    {
        "document": "The system supports dark mode and light mode themes.",
        "criterion": "Bug issues are prioritized over enhancement issues.",
        "expected": "FAIL",
    },
    {
        "document": "All API endpoints require authentication via OAuth 2.0.",
        "criterion": "The agent must self-assign and comment before coding.",
        "expected": "FAIL",
    },
    {
        "document": "The deployment pipeline runs nightly at 02:00 UTC.",
        "criterion": "Agent-created issues receive the agent:proposed label.",
        "expected": "FAIL",
    },
    # === Haystack: rule buried in a long document ===
    # --- Expected PASS ---
    {
        "document": _HAYSTACK_WITH_PRIORITY,
        "criterion": "Bug issues are prioritized over enhancement issues.",
        "expected": "PASS",
    },
    {
        "document": _HAYSTACK_WITH_LABEL_PROTOCOL,
        "criterion": "Agent-created issues receive the agent:proposed label.",
        "expected": "PASS",
    },
    {
        "document": _HAYSTACK_WITH_PICKUP,
        "criterion": "The agent must self-assign and comment before coding.",
        "expected": "PASS",
    },
    # --- Expected FAIL: similar document but rule is absent ---
    {
        "document": _HAYSTACK_WITHOUT_PRIORITY,
        "criterion": "Bug issues are prioritized over enhancement issues.",
        "expected": "FAIL",
    },
    {
        "document": _HAYSTACK_WITHOUT_LABEL_PROTOCOL,
        "criterion": "Agent-created issues receive the agent:proposed label.",
        "expected": "FAIL",
    },
    {
        "document": _HAYSTACK_WITHOUT_PICKUP,
        "criterion": "The agent must self-assign and comment before coding.",
        "expected": "FAIL",
    },
]


@dataclass
class CalibrationResult:
    passed: bool
    total: int
    correct: int
    details: list[dict]


def calibrate(llm: GraderLLM, system_prompt: str | None = None) -> CalibrationResult:
    """Run golden tests against the LLM and return results.

    Parameters:
        llm: Any object implementing the GraderLLM protocol.
        system_prompt: Custom system prompt. Defaults to JUDGE_SYSTEM_PROMPT.
    """
    prompt = system_prompt if system_prompt is not None else JUDGE_SYSTEM_PROMPT
    details = []
    correct = 0

    for test in GOLDEN_TESTS:
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (f"DOCUMENT:\n{test['document']}\n\nCRITERION:\n{test['criterion']}"),
            },
        ]
        try:
            response = llm.complete(messages, max_tokens=16).strip().upper()
            if response == "PASS":
                verdict = "PASS"
            elif response == "FAIL":
                verdict = "FAIL"
            else:
                verdict = f"INVALID: {response[:50]}"
        except Exception as e:
            verdict = f"ERROR: {e}"

        is_correct = verdict == test["expected"]
        if is_correct:
            correct += 1

        details.append(
            {
                "criterion": test["criterion"],
                "expected": test["expected"],
                "actual": verdict,
                "correct": is_correct,
            }
        )

    return CalibrationResult(
        passed=correct == len(GOLDEN_TESTS),
        total=len(GOLDEN_TESTS),
        correct=correct,
        details=details,
    )
