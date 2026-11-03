# Plan: madeleine v0.1.0

Goal: an open-source Python library + MCP server that gives AI agents **involuntary,
cue-triggered memory** (see [ADR 0001](docs/adr/0001-core-model.md)).

## Phase 0 — contract (lead, sequential)

- [x] `pyproject.toml` (uv, hatchling, extras), LICENSE (Apache-2.0)
- [x] `src/madeleine/core/types.py`: every protocol and dataclass, documented
- [x] ADR 0001 (core model), ADR 0002 (dependencies, MCP SDK 2.x)
- [x] ARCHITECTURE.md (layout, ownership, seams), CLAUDE.md (rules for agents)
- [x] Contract tests (`tests/test_types.py`), `config.py`

## Phase 1 — parallel workstreams (one git worktree each)

| # | Workstream | Deliverables | Done when |
|---|---|---|---|
| 1 | core-engine | `core/fusion.py`, `engine.py`, `store/sqlite.py`, `store/memory.py` | property tests for fusion monotonicity/bounds, refractory, decay, queue atomicity; sqlite-vec + fallback |
| 2 | channels | text/image/audio/time/geo/biometric/smell, `HashEmbedder`, geohash, registry | contract tests for every embedder (dim, determinism, sim bounds); models tests marked |
| 3 | mcp-server | 5 tools, resource, notifications, stdio + streamable HTTP | in-process MCP client integration tests |
| 4 | ingestion-integrations | FastAPI app, webhook adapters, Claude Code hook, `examples/` | TestClient tests; hook stdin/stdout test |
| 5 | evals | synthetic generator, metrics, `run_bench()` | deterministic report with recall@k, trigger precision, false-trigger rate, p50/p95 |

Workstreams 3–5 build against the protocols; until 1–2 merge they use local fakes in their
own test directories. The lead merges each branch after its tests pass and wires real
implementations.

## Integration (lead)

- `cli.py`, `serve.py`, `demo.py`; swap fakes for real engine in integration tests
- `tests/test_end_to_end.py`: the definition-of-done demo scenario

## Phase 2 — sequential

6. **reviewer** (fresh subagent): REVIEW.md against ARCHITECTURE.md, security/privacy
   checklist, API consistency. Every item fixed or explicitly deferred with reason.
7. **docs-release**: README, docs/concepts.md, CONTRIBUTING, SECURITY, CHANGELOG, CI
   (lint, typecheck, tests on 3.11–3.13), PyPI trusted publishing workflow, `server.json`.
   Nothing is published.

## Definition of done (v0.1.0)

- `uvx madeleine-mcp demo` runs offline end-to-end: 3 memories with different fingerprints,
  a cue (same geohash + similar hour) → an involuntary recollection is queued → the MCP
  tool returns it.
- CI green; README quickstart works copy-pasted in a clean environment.
- Benchmark results reported honestly, including failure modes.

## Out of scope (v0)

Real olfactory sensors (stub + docs only), cloud hosting, UI, multi-user auth, learned
fusion, re-embedding tooling.

## Branching note

This session may push only to `fmv/dazzling-bell-jmzwuo`; that branch plays the role of
`main` for integration. Phase 1 workstreams branch from it (`ws/<name>`) in
`.claude/worktrees/<name>` and are merged back into it by the lead.
