# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0](https://github.com/ugai/pytest-llm-rubric/releases/tag/v0.4.0) — 2026-04-07

### Bug Fixes

- Surface any-llm ImportError instead of silent preflight failure (#52)(48c19d8)
- Rewrite 0.0.0.0 to localhost in parse_ollama_host (#48) (#51)(c2d59c0)
- Correct `__all__` alphabetical sort order(4c28654)
- Show LLM Rubric terminal summary on preflight failure(ba8ba93)

### Documentation

- Add CLI backend aside (Claude Code / Copilot)(ae38d91)
- Position complete() as co-equal API alongside judge() (#41)(08f7065)

### Features

- Add llm_rubric_skip_preflight ini option (#60)(ac15f4e)
- Add register_judge helper and pytest-xdist compatibility (#42)(9dd1e6e)

### Refactor

- Unify PYTEST_LLM_RUBRIC_MODEL + AUTO_MODELS into PYTEST_LLM_RUBRIC_MODELS (#49) (**BREAKING**)(a542a69)
- Use TypedDict for golden test data and preflight details(a30ac63)

## [0.3.0](https://github.com/ugai/pytest-llm-rubric/releases/tag/v0.3.0) — 2026-03-28

### Bug Fixes

- Fall back to auto when llm_rubric_auto_models is set in ini (#36)(e52c2a9)

### Features

- Show rubric judgment summary in terminal report (#39)(2109b03)
- Show preflight duration in test output (#38)(61ba325)
- Unify PROVIDER+MODEL env vars into provider:model syntax (#32) (**BREAKING**)(47f23d1)

## [0.2.0](https://github.com/ugai/pytest-llm-rubric/releases/tag/v0.2.0) — 2026-03-26

### Bug Fixes

- Mock httpx in model-not-found test to avoid CI failure(918114d)
- Surface discovery failure reasons in skip/fail messages (#21)(1814a8b)
- Address review feedback on import guard(5eaf09b)
- Include ollama in base dependencies and add import guard(bcd69fb)
- Support thinking models (Qwen3.5 etc.) in preflight(43eba7a)
- Update remaining calibrate→preflight references(bf7cac1)
- Update experiments/stability_test.py for calibrate→preflight rename(720746d)
- Fail instead of skip when explicit backend is unavailable(7b9d675)
- Retry on empty LLM response to handle intermittent Ollama failures (#10)(4745891)
- Use max_completion_tokens instead of deprecated max_tokens(29fccd5)
- Normalize OLLAMA_HOST to handle all formats accepted by Ollama(2281dba)

### Documentation

- Fix model inconsistencies and clarify skip/fail behavior (#30)(4a8c9cc)
- Use uv sync --extra ollama in dev setup instructions(1a1643a)
- Mention canirun.ai and llmfit for model selection guidance(9c650a7)
- Overhaul README for clarity and accuracy (#15)(4c1bebe)
- Add CI and PyPI badges to README(6eb81a8)

### Features

- Add judge() convenience method and export parse_verdict (#29)(7ce3329)
- Simplify env vars and add any-llm provider passthrough (#19)(86a2aa0)
- Use structured output for calibration judgments (#11) (**BREAKING**)(c71d198)

### Performance

- Stop calibration early on first failure(6d62a7d)

### Refactor

- Consolidate env var constants and fix skip/fail docs(9e00a44)
- Rename calibrate to preflight (**BREAKING**)(47a40da)
- Migrate LLM backend to any-llm (#8)(510a379)
- Rename max_tokens param to max_output_tokens and improve type safety(dec5ec5)
- Extract OLLAMA_DEFAULT_HOST and OLLAMA_DEFAULT_PORT constants(d4188f2)

## [0.1.0](https://github.com/ugai/pytest-llm-rubric/releases/tag/v0.1.0) — 2026-03-20

### Bug Fixes

- Formatting issues in README commands(561464b)

### CI

- Merge publish workflow into release workflow(f6c55f2)
- Add README codeblocks syntax check with pytest-codeblocks(91727ef)
- Add CI and Release workflows(b29c1e6)
- Add pytest unit tests to pre-commit hooks(c5571d3)

### Documentation

- Fix wording and link defaults to source(fa9adb3)
- Sharpen README positioning and rename find_model to find_local_model(05a09fe)

### Features

- Initial implementation of pytest-rubric-grader(c46e72f)

### Miscellaneous

- Prepare v0.1.0 release(f2e3952)

### Refactor

- Improve calibration parsing, consolidate test fakes, extract golden tests(0c7318b)
- Rename package to pytest-llm-rubric and align public API(e57a506)
- Rewrite README for reader-order, add marker and system prompt features(b0a37dd)
