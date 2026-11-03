# CLAUDE.md — working on madeleine

madeleine gives AI agents involuntary, cue-triggered memory. Read, in order:
`docs/adr/0001-core-model.md`, `ARCHITECTURE.md`, `src/madeleine/core/types.py`.

## Commands

```bash
uv sync                      # install (dev group included)
uv run pytest                # offline tests (models-marked tests are skipped)
uv run pytest -m models      # tests that download real models
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy                  # strict, on src/madeleine
```

All four must pass before every commit.

## Rules

1. **The contract is `core/types.py`.** Build only against it. Never change it unilaterally:
   if something is missing or wrong, stop and report to the lead (a change needs an ADR).
2. **Ownership.** Only edit files your workstream owns (table in ARCHITECTURE.md) plus your
   own `tests/<area>/`. Shared files (`pyproject.toml`, `tests/conftest.py`, `cli.py`) are
   the lead's: if you need a dependency or CLI hook, say so in your final report.
3. **Offline tests.** Unit tests use `HashEmbedder` / fakes. Anything that needs a real model
   or network is `@pytest.mark.models` and skipped by default.
4. **No side effects at import.** No network, no model loading, no heavy imports
   (torch, open_clip, transformers, sentence_transformers, fastapi in core paths) at module
   import time. Load lazily on first use and raise `ChannelNotAvailableError` with the
   pip extra to install when a dependency is missing.
5. **Privacy by default.** Raw audio/image bytes are never persisted unless
   `store_raw=True`. Never log raw media or full memory contents at INFO or above.
6. **Types.** `mypy --strict` clean. Timestamps are timezone-aware everywhere.
7. **Commits.** Conventional commits (`feat(engine): ...`, `test(channels): ...`), small and
   focused. No model identifiers in commit messages.
8. **stdout is sacred** in the stdio MCP server: log to stderr only.
9. **End every workstream with a summary**: what was built, test results, open issues,
   any contract change requests.

## Style

Python 3.11+, `from __future__ import annotations`, dataclasses over dicts, small pure
functions, `logging.getLogger(__name__)`, docstrings on public API. Line length 100.
