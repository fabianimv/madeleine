# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-11-05

### Added
- Read and edit memories ([ADR 0003](docs/adr/0003-reading-and-editing-memories.md)):
  - MCP tools `get_memory` and `update_memory`;
  - HTTP `GET` and `PATCH /memories/{id}`;
  - CLI `madeleine show` and `madeleine edit`;
  - contract: `MemoryEngine.get_memory`, `MemoryEngine.update_memory`, `MemoryStore.update`.
  Only annotations are editable (content, tags, salience, metadata, media_ref); the sensory
  context of the moment of encoding is not. Editing derived text re-embeds it.
- `madeleine bench --text-backend sentence-transformers` runs the benchmark with semantic text
  embeddings; results for both backends are in the README and docs/benchmarks.md.

### Changed
- The default semantic text model (all-MiniLM-L6-v2) is calibrated: text similarity is cosine
  rescaled from [0.35, 0.80] to [0, 1]. Plain cosine let unrelated sentences score 0.2-0.4 and
  ride along on place/time matches. Benchmark seed 0: precision 0.46 -> 0.70, hit rate
  0.54 -> 0.67, false triggers still 0.

## [0.1.1] - 2026-11-04

### Changed
- MCP tool descriptions now say when to use each tool versus the others, list their side
  effects and return values, and every tool declares complete annotations (read-only,
  destructive, idempotent, open-world). Behaviour is unchanged.

## [0.1.0] - 2026-11-03

First public version.

### Added
- **Core model** ([ADR 0001](docs/adr/0001-core-model.md)): memories with sensory
  fingerprints, multi-channel cues, normative recall fusion (base · coverage · salience ·
  decay with floor), pull and push modes, a symmetric refractory period (habituation), and
  the rule that context alone never triggers.
- **Channels**:
  - `text`: sentence-transformers via the `[text]` extra, or the offline lexical hash
    embedder.
  - `time`: cyclical hour, weekday and season, computed in local time.
  - `geo`: geohash and great-circle distance.
  - `biometric`.
  - `image`: open_clip via the `[image]` extra.
  - `audio`: LAION CLAP via the `[audio]` extra.
  - `smell`: a stub.
  - Models load lazily and are never imported at import time.
- **Storage**:
  - SQLite with a `sqlite-vec` KNN index, and an exact numpy fallback.
  - A persistent recollection queue shared across processes, with atomic drain and atomic
    trigger claims.
  - Schema versioning.
- **MCP server** (official SDK 2.x):
  - Tools: `remember`, `recall`, `sense`, `pending_recollections`, `forget`.
  - Resource: `recollections://pending`, with `resources/updated` notifications, both
    in-process and across processes.
  - Transports: stdio and streamable HTTP.
- **HTTP ingestion** (FastAPI):
  - Routes: `/cues`, plus adapters for generic webhooks, Home Assistant and iOS Shortcuts;
    `/memories`, `/recall`, `/recollections/pending` (peek) and
    `/recollections/drain`.
  - Protection: Host/Origin allow-list, an optional bearer token, and a body size limit.
- **Claude Code `UserPromptSubmit` hook** (`madeleine hook claude-code`): injects pending
  involuntary recollections as delimited, untrusted context, and never blocks a prompt.
- **CLI**: `serve`, `remember`, `sense`, `recall`, `pending`, `forget`, `demo` (offline,
  end to end), `bench`.
- **Synthetic benchmark** (`madeleine bench`): recall@k, trigger precision, hit rate,
  false-trigger rate, latency p50/p95, a habituation check and a threshold sweep. See
  [docs/benchmarks.md](docs/benchmarks.md).
- **Examples**: Claude Code, Home Assistant, iOS Shortcuts, curl.
- **CI**: lint, strict typecheck, tests on Python 3.11–3.13, and an offline demo run from the
  built wheel. Release workflow for PyPI Trusted Publishing and the MCP Registry
  (`server.json`).

### Security
- An independent pre-release review ([REVIEW.md](REVIEW.md)) led to these changes:
  - Local file access is off by default and enforced in the media adapters.
  - DNS-rebinding protection is added on the REST routes.
  - Draining recollections is now a POST.
  - Owner-only file permissions.
  - `secure_delete` on `forget`.
  - Slimmer queue payloads (no embeddings).
  - The hook no longer loses recollections, and wraps them as untrusted data.

[0.2.0]: https://github.com/fabianimv/madeleine/releases/tag/v0.2.0
[0.1.1]: https://github.com/fabianimv/madeleine/releases/tag/v0.1.1
[0.1.0]: https://github.com/fabianimv/madeleine/releases/tag/v0.1.0
