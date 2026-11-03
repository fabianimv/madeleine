# Contributing to madeleine

Thanks for helping! Bug reports, new channels (senses), integrations and benchmark
improvements are all welcome.

## Setup

```bash
git clone https://github.com/FabianIMV/madeleine && cd madeleine
uv sync                       # Python 3.11+; installs the dev group
uv run madeleine demo         # sanity check, fully offline
```

## Before every commit

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy                   # strict
uv run pytest                 # offline; real-model tests are skipped
```

`uv run pytest -m models` runs the tests that download real models. They need the extras:
`uv sync --all-extras`.

## Ground rules

* **The contract** is `src/madeleine/core/types.py`. Changing an entity, a protocol or the
  fusion formula needs an ADR in `docs/adr/` (copy the format of ADR 0001).
* **Tests must pass offline.** Use `HashEmbedder` / `hash_registry()` or the fakes in
  `madeleine.testing`. Mark anything that needs a model or the network
  `@pytest.mark.models`.
* **No side effects at import time:** no network, no model loading, no heavy imports. Load
  lazily and raise `ChannelNotAvailableError` naming the pip extra.
* **Privacy by default:** never persist raw media unless `store_raw=True`, never log memory
  contents at INFO or above, never read local files unless `allow_local_files` is on.
* **Parse external input with `madeleine.core.serde`**, so the MCP server, the HTTP API, the
  CLI and webhooks behave identically.
* The stdio MCP server must never print to stdout. Log to stderr.
* Use conventional commits (`feat(channels): ...`, `fix(engine): ...`) and keep PRs small.

## Adding a channel

See [docs/channels.md](docs/channels.md). A channel needs an adapter, an embedder, a
similarity function and a `ChannelSpec`. Add it to the parametrised contract test in
`tests/channels/`, document it, and run `madeleine bench` to see its effect.

## Reporting security issues

Please do not open public issues for vulnerabilities; see [SECURITY.md](SECURITY.md).
