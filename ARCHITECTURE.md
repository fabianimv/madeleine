# Architecture (v0)

The contract is `src/madeleine/core/types.py`; the reasoning is in
[ADR 0001](docs/adr/0001-core-model.md) and [ADR 0002](docs/adr/0002-dependencies-and-sdk.md).
This document fixes the **module layout, ownership, and the seams between components** so
that workstreams can be built in parallel.

```mermaid
flowchart LR
    subgraph Sources
      HA[Home Assistant] -->|webhook| ING
      IOS[iOS Shortcuts] -->|webhook| ING
      CLI[madeleine CLI] --> ENG
      AG[Agent via MCP] --> MCP
    end
    ING[ingest: FastAPI POST /cues] --> ENG
    MCP[mcp_server: tools + resource] --> ENG
    subgraph ENG[engine: Madeleine]
      direction TB
      CH[channels: adapter → embedder → similarity] --> FU[fusion: base·coverage·salience·decay]
      FU --> RF{push? threshold + refractory}
    end
    ENG <--> ST[(SQLite + sqlite-vec: memories, vectors, triggers, queue)]
    RF -->|enqueue| ST
    ST -->|drain| HOOK[Claude Code UserPromptSubmit hook]
    ST -->|drain / notify| MCP
```

## Module layout and ownership

| Path | Owner (Phase 1 workstream) | Purpose |
|---|---|---|
| `core/types.py` | lead | Contract. Changes need an ADR. |
| `config.py` | lead | `load_config(env, **overrides)`, `resolve_db_path()` |
| `core/geohash.py` | lead | Pure geohash `encode/decode/bounds/is_valid`, `haversine_km` (use it; don't re-implement) |
| `core/serde.py` | lead | `parse_timestamp`, `context_from_fields`, `signal_from_json(s)`: the ONE way MCP/HTTP/CLI/webhooks parse inputs |
| `testing.py` | lead | `FakeClock`, `FakeEngine` (crude in-memory `MemoryEngine` for testing consumers), `FakeQueue` |
| `cli.py`, `demo.py`, `serve.py` | lead (integration) | Typer CLI and composition |
| `core/fusion.py` | core-engine | Pure functions: `fuse()`, `decay_factor()`, `salience_factor()`, `coverage()` |
| `engine.py` | core-engine | `Madeleine` (implements `MemoryEngine`), `create_engine()` |
| `store/sqlite.py` | core-engine | `SQLiteMemoryStore`, `SQLiteRecollectionQueue` (same file) |
| `store/memory.py` | core-engine | `InMemoryStore`, `InMemoryRecollectionQueue` (tests, ephemeral) |
| `channels/registry.py` | channels | `DefaultChannelRegistry`, `default_registry(config)` |
| `channels/{text,image,audio,time,geo,biometric,smell}.py` | channels | One module per sense |
| `channels/hashing.py` | channels | `HashEmbedder` (offline fake) |
| `channels/similarity.py` | channels | `cosine01`, helpers |
| `mcp_server/server.py` | mcp-server | `create_server(engine) -> MCPServer`, `run_stdio(engine)` |
| `mcp_server/http.py` | mcp-server | `mcp_asgi_app(engine)` + lifespan for streamable HTTP |
| `ingest/app.py` | ingestion | `create_app(engine, *, mounts=None) -> FastAPI` |
| `ingest/webhook.py`, `ingest/adapters/*.py` | ingestion | `CueAdapter`s: generic, Home Assistant, iOS Shortcuts |
| `integrations/claude_code.py` | ingestion | `UserPromptSubmit` hook entry point |
| `evals/*` | evals | Synthetic generator, metrics, runner, `run_bench()` |
| `examples/` | ingestion (+ lead) | Payloads, Home Assistant YAML, Shortcuts recipe, hook settings |

Tests mirror this: `tests/engine/`, `tests/channels/`, `tests/mcp_server/`, `tests/ingest/`,
`tests/evals/`. Each workstream owns its test directory only. Shared fixtures that more
than one workstream needs are added by the lead in `tests/conftest.py`.

## Seams (the functions other components call)

```python
# engine.py (core-engine)
def create_engine(config: MadeleineConfig, *, clock: Clock | None = None,
                  registry: ChannelRegistry | None = None,
                  store: MemoryStore | None = None,
                  queue: RecollectionQueue | None = None) -> Madeleine: ...
# registry None -> madeleine.channels.default_registry(config) (lazy import)
# store/queue None -> SQLite at resolve_db_path(config.db_path)

# channels/registry.py (channels)
def default_registry(config: MadeleineConfig) -> DefaultChannelRegistry: ...
#  registers config.channels; optional channels whose extra is missing are skipped with a
#  logged warning (an Embedder may also raise ChannelNotAvailableError lazily on first use).
def hash_registry(channels: Iterable[str] = (TEXT, TIME, GEO, BIOMETRIC, IMAGE, AUDIO),
                  dim: int = 256) -> DefaultChannelRegistry: ...
#  every ML channel backed by HashEmbedder; time/geo/biometric use their real deterministic
#  encoders. Used by tests, demo and evals. Never downloads anything.

# mcp_server/server.py (mcp-server)
def create_server(engine: MemoryEngine, *, poll_interval: float = 1.0) -> MCPServer: ...
def run_stdio(engine: MemoryEngine) -> None: ...
# mcp_server/http.py
def mcp_asgi_app(engine: MemoryEngine, *, host: str = "127.0.0.1", allow_file_uris: bool = False,
                 **kwargs) -> tuple[ASGIApp, Callable[[], AsyncContextManager[None]]]
#  (app, lifespan) to mount at /mcp inside the FastAPI app.

# ingest/app.py (ingestion)
def create_app(engine: MemoryEngine, *, mounts: Mapping[str, ASGIApp] | None = None,
               lifespans: Sequence[Callable[[], AsyncContextManager[None]]] = (),
               adapters: Sequence[CueAdapter] | None = None,
               api_token: str | None = None,       # None -> $MADELEINE_API_TOKEN, "" -> off
               max_body_bytes: int | None = None,  # None -> $MADELEINE_MAX_BODY_BYTES or 10 MiB
               clock: Clock | None = None) -> FastAPI: ...

# integrations/claude_code.py (ingestion)
def main(argv: Sequence[str] | None = None, *, stdin=None, stdout=None,
         engine_factory: Callable[[MadeleineConfig], MemoryEngine] | None = None) -> int
#  flags: --url --token-file --token (deprecated) --db --sense-prompt --limit --timeout; always exits 0

# evals (evals)
def run_bench(*, memories: int = 500, cues: int = 300, seed: int = 0,
              engine_factory: Callable[[RecallConfig], MemoryEngine] | None = None,
              recall_config: RecallConfig | None = None,
              thresholds: Sequence[float] | None = None,
              habituation: bool = True) -> BenchReport  # .to_markdown(), .to_dict()
# evals/cli.py: main(argv) with --memories --cues --seed --json --output PATH --sweep
```

## Channels (v0 defaults)

| Channel | Embedder | dim | Similarity → [0,1] | default weight | solo_trigger | Extra |
|---|---|---|---|---|---|---|
| `text` | sentence-transformers MiniLM-L6-v2 (or HashEmbedder) | 384 (hash: 256) | `max(0, cos)`; hash backend: cosine rescaled [0.10, 0.45] → [0, 1] | 1.0 | yes | `[text]` |
| `image` | open_clip ViT-B-32 | 512 | rescaled cosine (CLIP cosines cluster high) | 1.0 | yes | `[image]` |
| `audio` | LAION CLAP (htsat-unfused) | 512 | rescaled cosine | 1.0 | yes | `[audio]` |
| `time` | cyclical: hour-of-day, weekday, day-of-year as (sin, cos) | 6 | mean of per-cycle `(1+cos Δ)/2`, hour-weighted | 0.5 | **no** | – |
| `geo` | geohash → cell centre → unit-sphere xyz | 3 | `exp(-d_km / 1.0)` | 1.0 | yes | – |
| `biometric` | fixed-range normalisation with presence mask | 2×features | similarity over features present in both | 0.5 | **no** | – |
| `smell` | stub: adapter raises `ChannelNotAvailableError` | – | – | – | – | future |

ImageBind is **not** a dependency (CC-BY-NC); users may plug it in themselves as an
`Embedder` (documented in `docs/channels.md`).

## Storage (SQLite)

One file (default `~/.madeleine/madeleine.db`, WAL mode). Tables (core-engine may refine):

* `memories(id PK, content, media_ref, created_at, ts, geohash, salience, tags_json, metadata_json, observations_json)`
* `raw_blobs(memory_id, channel, data)` — only when `store_raw`
* `channels(name PK, dim, model_id)`
* `vectors(memory, channel, embedding)` — source of truth; `vec_ch<channel id>` — `vec0` KNN index rebuilt from it (named by integer id, never by channel name)
* `triggers(memory_id, at)`
* `queue(seq INTEGER PK AUTOINCREMENT, rec_id, memory_id, payload_json, enqueued_at)` —
  `payload_json` (schema 2) holds only the memory id + score explanation; the memory is
  rehydrated from `memories` on peek/drain, and rows of forgotten memories are dropped

`drain` deletes in a single transaction (`BEGIN IMMEDIATE`). Every connection sets
`PRAGMA secure_delete = ON`; files are created owner-only (`0600`, directory `0700`).
`meta.vec_dirty` (vectors written by a process without sqlite-vec) makes `nearest` use exact
brute force until the index is rebuilt. If `sqlite_vec` cannot be
loaded (some system Pythons disable extension loading) the store falls back to numpy
brute-force over vectors kept in a plain table, logging a warning once.

## MCP surface (mcp-server)

| Tool | Input | Output |
|---|---|---|
| `remember` | `content`, `geohash?`, `lat?`/`lon?`, `at?` (ISO 8601), `tags?`, `salience?`, `signals?` (`[{channel, value, mime_type?, uri?}]`), `media_ref?` | `Memory.to_dict()` |
| `recall` | `query?` (text), `geohash?`, `lat?`/`lon?`, `at?`, `channels?`, `tags?`, `k=5` | `[ScoredMemory.to_dict()]` |
| `sense` | `text?`, `geohash?`, `lat?`/`lon?`, `at?` (default now), `biometric?`, `signals?`, `source?` | `[Recollection.to_dict()]` that fired |
| `pending_recollections` | `limit?`, `drain=true` | `[Recollection.to_dict()]` |
| `forget` | `memory_id` | `{deleted: bool}` |

Resource `recollections://pending` (JSON list, non-draining peek). When the in-process queue
listener fires, or the poller sees `pending_count` grow (other process), the server sends
`notifications/resources/updated` for that URI to subscribed sessions.
Image/audio over MCP and HTTP: `signals[].data_b64` (base64) with `mime_type`, or a `uri` (file paths
only if the server was started with `--allow-file-uris`; off by default).

## HTTP surface (ingestion)

* `POST /cues` — body: `{signals: [{channel, value?, data_b64?, mime_type?, uri?}], text?, at?, geohash?, lat?, lon?, biometric?, tags?, salience?, source?}` → `{cue_id, recollections: [...]}`; an empty cue is a 422
* `POST /cues/{adapter}` — adapter-specific payload (`generic`, `home_assistant`, `ios_shortcuts`)
* `POST /memories` — the `remember` tool's fields (+ `metadata`, `embed_content`, `store_raw`) → 201
* `POST /recall` — the `recall` tool's fields (time only when `at` is given; unknown channels → 422)
* `DELETE /memories/{id}`
* `GET /recollections/pending?limit=` — peek (never drains); `POST /recollections/drain?limit=` — take
* `GET /healthz`
* Guard on every route including mounted `/mcp`: Host/Origin allow-list (loopback +
  `MADELEINE_ALLOWED_HOSTS`; 421/403), optional bearer token `MADELEINE_API_TOKEN` (all routes
  but `/healthz`), body limit (`MADELEINE_MAX_BODY_BYTES`, default 10 MiB → 413).
  Binds to `127.0.0.1` by default; `serve` refuses a non-loopback bind without a token unless
  `--insecure-no-auth`.
* Errors: `ObservationError`/`ValueError` → 422, `ChannelNotAvailableError` → 400,
  `MemoryNotFoundError` → 404, `StoreError` → 409, other `MadeleineError` → 500 (with message).
* Local files: media signals may reference files on disk only when
  `MadeleineConfig.allow_local_files` (`serve --allow-file-uris`); enforced by the media
  adapters, so it applies to MCP, HTTP and webhooks alike.

## CLI (lead, integration)

`madeleine serve [--transport stdio|http] [--host 127.0.0.1] [--port 8765] [--db PATH]`,
`madeleine remember TEXT [...]`, `madeleine sense [...]`, `madeleine recall QUERY`,
`madeleine pending`, `madeleine demo`, `madeleine bench [...]`, `madeleine hook claude-code`.

## Cross-cutting rules

* No network and no model loading at import time; heavy imports inside functions.
* Everything typed, `mypy --strict` clean; `ruff` clean.
* Unit tests offline with `HashEmbedder`; real-model tests marked `@pytest.mark.models`.
* Raw media not persisted unless `store_raw=True`.
* Logging via `logging.getLogger("madeleine.<module>")`; never `print` outside the CLI.
  The stdio MCP server must never write to stdout except protocol frames.
