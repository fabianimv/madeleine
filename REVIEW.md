# madeleine v0.1.0 pre-release review

Reviewer: independent senior review, 2026-11-03, branch `fmv/dazzling-bell-jmzwuo` at `e47d205`.
Scope: all of `src/madeleine/`, plus `tests/`, `examples/`, the packaging files and the CI and release workflows.
Experiment scripts and temp DBs are in
`/tmp/claude-0/-home-user-madeleine/4beb5bec-73e5-58d3-8123-4b4909992689/scratchpad/review/`
(`exp_files.py`, `exp_forget.py`, `exp_hook.py`, `exp_engine.py`, `exp_race.py`, `stdio_probe.py`).

## Summary

The core is in good shape. The fusion formula and the push rules match ADR 0001 exactly. SQL is
fully parameterised. Multi-process claim and drain are atomic. Imports are lazy. Stdio stdout stays
clean. ruff, format, mypy (strict) and pytest all pass (**441 passed**, 3 `models` tests
deselected). `madeleine demo` and `uvx --from <wheel> madeleine-mcp demo` both work offline.

**Do not release yet.** Three security problems affect the default `serve --transport http`
setup or defeat a documented control, and two problems block the release itself:

1. **The REST routes have no DNS-rebinding protection** (R1). The default setup has no token, so
   any web page can read, write and drain memories through a rebinding attack. `/mcp` is
   protected; `/cues`, `/memories`, `/recall` and `/recollections/pending` are not.
2. **The HTTP API reads arbitrary local files** (R2). `uri` and string `value` signals on
   `/cues`, `/memories` and `/cues/{adapter}` have no file-URI policy. `--allow-file-uris` only
   applies to MCP.
3. **The MCP `allow_file_uris=False` guard can be bypassed** (R3). The guard checks `uri`, but the
   image and audio adapters also treat a string `value` as a path.
4. **Release blockers** (R4, R5): the README is a placeholder, and it lacks the `mcp-name:` marker
   that the MCP Registry needs to verify ownership of a PyPI package.

R2 and R3 matter only when the `image` or `audio` channel is registered (the `[image]` or
`[audio]` extra, or a custom or hash registry). A base install skips those channels.

Severity counts: **critical 0 · high 5 · medium 6 · low 15 · nit 6**.

## Findings

| ID | Sev | Area | file:line | Finding | Evidence | Suggested fix |
|---|---|---|---|---|---|---|
| R1 | high | security / HTTP | `src/madeleine/ingest/app.py:315` (guard), `src/madeleine/serve.py:57` | The REST routes do not validate the `Host` or `Origin` header. Auth is off by default (`MADELEINE_API_TOKEN` unset), so a malicious site can use DNS rebinding to drive `127.0.0.1:8765` as same-origin. It can then read all memories (`/recall`, `/recollections/pending`), plant memories (`/memories`, which then reach Claude Code through the hook, see R11) and delete them. The SDK protects only the mounted `/mcp`. | `curl -X POST :18765/recall -H 'Host: evil.example:18765' -d '{"query":"planted"}'` returns **200** with results. The same request to `/mcp/` returns `421 Invalid Host header`. | Add a Host/Origin allow-list middleware (or Starlette `TrustedHostMiddleware` plus an Origin check) to `create_app` for loopback binds, reusing the hosts the MCP SDK allows. Consider generating a random token on first `serve --transport http` and printing it. |
| R2 | high | security / HTTP + adapters | `src/madeleine/ingest/app.py:237-238`, `src/madeleine/ingest/webhook.py:50` (also reached via the `signals` passthrough in the `home_assistant` adapter), `src/madeleine/channels/_media.py:103-129`, `src/madeleine/channels/hashing.py:174` | The HTTP API has no local-file policy. A signal `{"channel":"image"\|"audio","uri":"file:///…"}` (or a string `value`) makes the server `stat()` the path and then read it in the embedder: `read_media_bytes`, `HashEmbedder._uri_payload`, `sf.read`. That gives (a) a file-existence oracle (`422 "file not found"` versus `200`/`201`), (b) reads of any file the server user can read, and (c) the path is persisted in `observations_json`. An attacker can also confirm a file's content by comparing its embedding with a recall cue. `--allow-file-uris` has no effect on HTTP even though `serve --help` implies a single flag. | `exp_files.py`: `POST /memories` with `uri=file:///etc/passwd` returns **201**, and the spy records the read of `file:///etc/passwd`. `/cues` with a missing file returns `422 image: file not found: 'file:///etc/shadow-nope'`. `/cues/home_assistant` with audio `value=/etc/hostname` returns 200 and the file is read. | Enforce the policy in one place for every surface. Simplest: give `ImageAdapter`/`AudioAdapter` an `allow_local_files` flag (default False) wired from `--allow-file-uris`, so the check cannot be bypassed by the input path. At minimum, call `_inputs.parse_signals(..., allow_file_uris=...)` from the HTTP routes and adapters. Pass the flag through `build_http_app` to `create_app`. |
| R3 | high | security / MCP | `src/madeleine/mcp_server/_inputs.py:87-95`; `src/madeleine/channels/image.py:130`; `src/madeleine/channels/audio.py:115` | `parse_signals` checks only the `uri` key. The adapters also accept a **string `value`** as a path, so `{"channel":"image","value":"/path"}` gets past `allow_file_uris=False` on stdio and `/mcp`. | `exp_files.py`: MCP `remember` with `value=<scratch>/secret.png` succeeds and the embedder reads the file. `sense` with audio `value=/etc/hostname` succeeds and reads it. `sense` with `value=/nonexistent/x.png` returns the tool error "file not found" (an oracle). The `uri=file:///etc/passwd` variant is correctly refused. | Same fix as R2 (enforce in the adapter). In `parse_signals`, also reject a string `value` on media channels, or reject any media signal that carries no `data_b64`. Add a regression test for the `value` path. |
| R4 | high | release | `README.md:1-3` | The README is a placeholder ("README is written in Phase 2"). It becomes the PyPI long description. ADR 0002 also requires the README to state prominently that semantic text needs `[text]`. | `unzip -p …whl METADATA` has no description body. | Write the README (install, `uvx madeleine-mcp serve`, extras, HTTP and auth, privacy, hook) before tagging. |
| R5 | high | release / MCP Registry | `server.json:3`, `README.md` | For PyPI packages, the MCP Registry verifies ownership by looking for `mcp-name: io.github.FabianIMV/madeleine` in the package README. The README lacks it, so `mcp-publisher publish` in `release.yml` should fail after the PyPI upload has already happened. | `grep mcp-name README.md` finds nothing. **Unverified** against the live registry; based on the registry's documented PyPI validation. | Add the line (an HTML comment is fine) to the README, then check with `mcp-publisher publish --dry-run` if available. |
| R6 | medium | security / HTTP | `src/madeleine/ingest/app.py:420-426` | `GET /recollections/pending` drains by default (`drain=true`), so a GET changes state. Without a token, any page can fire a no-CORS GET (for example `<img src=…>`) and silently destroy pending recollections before the hook or agent sees them. | `curl ':18765/recollections/pending?drain=true'` with no auth returns 200 (FastAPI's JSON content-type check does block cross-site POSTs; verified `text/plain` and no-content-type both give 422). | Make drain a `POST` (or `DELETE`), keep `GET` as peek-only, and have the remote hook use the POST. R1's Origin check also mitigates this. |
| R7 | medium | privacy / store | `src/madeleine/store/sqlite.py:558-579` (`delete`), `:199` (WAL) | `forget` removes every row (memories, vectors, raw_blobs, triggers, queue and the vec0 index, all verified at 0), but SQLite keeps freed pages, so the content and raw bytes stay recoverable from the file. "Forget truly deletes" is not true at the byte level. | `exp_forget.py`: after `forget` and `close` (and even after `wal_checkpoint(TRUNCATE)`), the DB file still contains `CONTENT-MARKER` ×10 and `RAWPIXELS-MARKER` ×64. | Set `PRAGMA secure_delete = ON` on every connection (and checkpoint the WAL after `delete`), or document the limitation and offer `madeleine vacuum`. |
| R8 | medium | correctness / hook | `src/madeleine/integrations/claude_code.py:101-120`, `:150` | The hook **drains first** and then fits lines under `MAX_CONTEXT_CHARS`. A recollection whose line is too long is dropped after it was already drained, so it is lost. Because of the `break`, every recollection after it is lost too. Memory `content` has no length limit (an 8.8 MB memory was accepted over HTTP). | `exp_hook.py`: two recollections were queued (one 12 000 chars). Hook output contains only the short one, and `pending after hook` is 0. | Truncate each line (for example to 1 000 chars with "…"). Better still, `peek` → choose what fits → `drain` exactly those (or re-queue the rest). Consider a max `content` length on `remember`. |
| R9 | medium | correctness / engine | `src/madeleine/engine.py:292-298` | Candidates come from a per-channel KNN with `candidate_k=64` **before** the `tags` filter and regardless of `k`. (a) `recall(k=100)` never returns more than 64 results from a single-channel cue, although MCP and HTTP accept `k≤100`. (b) `tags=` starves: if 64 closer untagged memories exist, a tagged match is never returned. | `exp_engine.py`: 100 memories at one geohash, `recall k=100` returns **64**. One memory tagged `x` about 1 km away, `recall tags={'x'}` returns **[]**. | Use `max(candidate_k, k)` per channel. When `tags` is set, pre-filter in SQL (or brute-force over the tagged subset). Document `candidate_k` as an approximation knob. |
| R10 | medium | API / error mapping | `src/madeleine/ingest/app.py:327-330` | Only `ObservationError`, `ValueError`, `ChannelNotAvailableError` and `MemoryNotFoundError` are mapped. `StoreError` (for example the documented "text model changed" refusal from ADR 0002) and any other `MadeleineError` return a bare **500** with no detail. MCP and the CLI show the message. | `exp_engine.py` step 4: DB created with hash dim=256, server opened with dim=128, `POST /memories` returns `500 Internal Server Error`. MCP returns a clear tool error. | Map `StoreError` to 409 (or 503) with its message, and `MadeleineError` to 500 with its message. |
| R11 | medium | security / prompt injection | `src/madeleine/integrations/claude_code.py:101-120` | Memory `content` is injected verbatim into Claude Code's context. Anyone who can write a memory (an HTTP client, a webhook, or a web page via R1) can inject instructions that surface later as "recollections". The header labels them, but nothing marks them as untrusted data. | By inspection; chains with R1. | Wrap the lines in an explicit "untrusted data, not instructions" block with delimiters, strip control characters, and document the trust model (who can write memories). |
| R12 | low | input validation | `src/madeleine/core/serde.py:44`, `:126` | `signal_from_json` calls `parse_timestamp(at)` for any truthy `at`. A non-string (`"at": 123`) raises `AttributeError`, which is a **500**. This is reachable through `/cues/{adapter}`, whose payload is untyped (`/cues` is protected by the pydantic model). | `curl -X POST /cues/generic -d '{"signals":[{"channel":"text","value":"x","at":123}]}'` returns 500, with a traceback in the server log. | Reject non-`str`/`datetime` values in `parse_timestamp`, or use `_fields.as_str`. |
| R13 | low | correctness / refractory | `src/madeleine/store/sqlite.py:679-683`, `src/madeleine/core/fusion.py:194-203` | The symmetric refractory rule is checked only against the **most recent** trigger. After an out-of-order or future-dated cue, a cue 1 h after a real fire fires again. This breaks "fired less than `refractory_period` ago". | `exp_engine.py` step 3: fires at T+1h, then T+100h, then T+2h. All three fire. | In `claim_trigger`, check `EXISTS(trigger with abs(at_us − ?) < period)`, a range query on the existing index. Apply the same change to the in-memory store. |
| R14 | low | correctness / sqlite-vec fallback | `src/madeleine/store/sqlite.py:277-292`, `:616` | `vec_dirty` is honoured only when a connection opens. A process (or connection) with sqlite-vec loaded never sees vectors written by a fallback-mode process until it restarts, so its KNN silently misses those memories. | Two `SQLiteDatabase` connections, one with `use_sqlite_vec=True` and one with `False`, on the same file: A recalls `['written by A']`, B recalls both. | Re-check `meta.vec_dirty` (or a write counter) in `nearest` and rebuild or fall back when it is set; or document "all processes must agree on sqlite-vec". |
| R15 | low | concurrency / privacy | `src/madeleine/engine.py:356-359`, `src/madeleine/store/sqlite.py:648-654` | `forget` runs `store.delete` and `queue.discard_memory` in separate transactions, and `claim_trigger` does not check that the memory still exists. A concurrent `sense` that fetched the memory before the delete can still add an orphan trigger and enqueue a recollection carrying the forgotten content. | **Unverified**; found by inspection of the transaction boundaries. | In `claim_trigger`, insert only `WHERE EXISTS (SELECT 1 FROM memories WHERE id=?)`. Make `forget` delete the memory and its queue rows in one transaction when both share a DB. |
| R16 | low | privacy | `src/madeleine/config.py:47`, `src/madeleine/store/sqlite.py:187` | `~/.madeleine/` and the DB, WAL and SHM files are created with the default umask (`0755`/`0644`), so other local users can read all memories. | `ls -la` on the scratch DBs shows `-rw-r--r--`. | `mkdir(mode=0o700)` and `os.chmod(db, 0o600)` on creation (and for the `-wal`/`-shm` files). |
| R17 | low | DoS | `src/madeleine/channels/_media.py:129`, `src/madeleine/channels/hashing.py:174` | Bytes are capped at `MAX_MEDIA_BYTES` (32 MiB), but a `uri` file is read whole with no size check. Once file URIs are allowed (or via R2/R3), a multi-GB file exhausts memory. | By inspection. | `stat()` and refuse files larger than `MAX_MEDIA_BYTES` in `normalize_file_uri` and `read_media_bytes`. |
| R18 | low | API consistency | `src/madeleine/mcp_server/server.py:257-277,314-327`; `src/madeleine/ingest/app.py:395,403-415` | Surfaces disagree. (a) MCP `sense` with no inputs returns `[]`; HTTP `/cues` returns 422 "cue is empty". (b) MCP `recall` rejects unknown `channels`; HTTP silently returns `[]`. (c) HTTP `recall` treats `query` values "null"/"none"/"unknown"/"unavailable" as missing (`_fields.as_str`). (d) ARCHITECTURE says `POST /memories` has "the same fields as `remember`", but HTTP also takes `metadata`, `embed_content` and `store_raw`, which the MCP tool lacks. (e) HTTP recollections include `prompt_line`; MCP ones do not. (f) `source` is capped at 64 chars on HTTP but unbounded on MCP. | Verified (a)–(c) with curl and in-process MCP calls. | Align on one behaviour each (suggest: reject empty cues everywhere; validate channels in the shared serde; use plain `serde` for `query`; add `metadata` to `remember`; cap `source` everywhere). |
| R19 | low | docs vs behaviour | `src/madeleine/mcp_server/_inputs.py:58`, `:93`; `src/madeleine/core/serde.py:103` | The `uri` field description ("Where the media lives (https://...)"), the refusal message ("use an http(s) URI") and the serde docstring example all suggest remote URIs work. The channels reject every non-file scheme ("madeleine never downloads media"). Models will follow the tool description and fail. | MCP `sense` with `uri=https://example.com/a.png` returns "only file:// URIs or local paths are supported". | Reword: "local file path (only if the server allows it); otherwise send `data_b64`". |
| R20 | low | privacy / data minimisation | `src/madeleine/store/codec.py:158` | Queue payloads store the full `memory_to_dict` including **embeddings** (base64 float32) and observation URIs. They are not raw bytes, but text embeddings are partly invertible, and they bloat the queue. | `exp_forget.py`: `queue payload carries embeddings: True`. | Store `memory_id` plus the score fields and rehydrate from the store when draining (forgotten memories then disappear on their own). Or at least drop `fingerprint.embeddings`. |
| R21 | low | CLI | `src/madeleine/config.py:100-101`, `src/madeleine/cli.py` | `--text-backend bogus` and an unknown `MADELEINE_CHANNELS` entry raise raw tracebacks: override values are not validated, and only `MadeleineError` is caught. `recall --lon 3` gives a misleading "give a query…" error. | `madeleine remember x --text-backend bogus` gives `ValueError: unknown text backend 'bogus'` with a traceback (exit 1). | Validate overrides in `load_config`, catch `ValueError`/`ChannelNotAvailableError` in the CLI, and check `lat`/`lon` together. |
| R22 | low | secrets | `src/madeleine/integrations/claude_code.py:76`, `:170-190` | `--token` is a documented CLI flag, so the token shows up in `ps` and in `settings.json`. In remote mode it is also sent over plain `http://` to non-loopback hosts without a warning. | By inspection. | Recommend the env var only (deprecate `--token` or support `--token-file`). Warn when the token goes over `http://` to a non-loopback host. |
| R23 | low | security defaults | `src/madeleine/serve.py:51-56` | Binding to a non-loopback host with no token only logs a warning. Everything, including `/mcp` and `/docs`, is then open to the network. | By inspection. | Refuse to start unless a token is set (or an explicit `--insecure-no-auth` is given). |
| R24 | low | release / versioning | `src/madeleine/__init__.py:21`, `pyproject.toml:3`, `server.json:10,15` | The version is declared in three places (`0.1.0.dev0`, `0.1.0.dev0`, `0.1.0`). `release.yml` checks only pyproject against server.json, so `__version__` (shown by `/healthz`, MCP and `--version`) can drift. | By inspection. | Make it single-source: hatch `dynamic = ["version"]` from `__init__.py`, or `importlib.metadata.version("madeleine-mcp")`. |
| R25 | low | dependencies | `pyproject.toml:22-29`; `src/madeleine/mcp_server/server.py:456`; `src/madeleine/mcp_server/notifier.py:262` | `pydantic`, `starlette`, `anyio` and `mcp-types` are imported directly but not declared; they arrive only transitively. The server relies on private SDK internals (`_lowlevel_server`, `session._connection`) under a loose `mcp>=2.0,<3`. | It imports and builds with `mcp==2.0.0` on Python 3.11 (checked), but private APIs can break in any minor release. | Declare the direct deps. Narrow to `mcp>=2.3,<2.4` (or add a CI job against the newest minor), with a test that fails loudly if the private attributes disappear. |
| R26 | low | input bounds | `src/madeleine/mcp_server/server.py:332-334` | MCP `pending_recollections.limit` has `ge=1` but no upper bound (HTTP caps at 1000). `limit=10**20` overflows SQLite and returns an opaque "Error executing tool pending_recollections". | `exp_engine.py` step 5. | Add `le=1000`. |
| R27 | nit | packaging | sdist | The sdist ships `CLAUDE.md`, `PLAN.md`, `.github/`, `uv.lock` and `server.json` (internal agent and process files). The wheel is clean: package, `py.typed`, both entry points, license. | `tar tzf …tar.gz` (129 files). | Add `[tool.hatch.build.targets.sdist] exclude = ["CLAUDE.md", "PLAN.md", ".github"]` (keep tests, docs and examples). |
| R28 | nit | docs drift | `ARCHITECTURE.md:129,131,154-163`; `src/madeleine/channels/text.py:83`; `src/madeleine/channels/registry.py:175` | ARCHITECTURE does not list `POST /recall`, `DELETE /memories/{id}`, or the `text`/`biometric`/`salience` fields of `/cues`. It names the tables `vec_<channel>` (code: `vec_ch<id>`, which is safer) and gives `raw_blobs` a `mime_type` column it does not have. `text.py:83` puts the docstring "max(0, cosine)" under `hash_text_similarity`, which is a rescaled cosine. The `hash_registry` docstring says text uses `cosine01`, and the ARCHITECTURE channel table says text uses `max(0, cos)`; the hash backend actually uses `rescaled_cosine(0.10, 0.45)`. | By inspection. | Update the docs to match the code. |
| R29 | nit | code | `src/madeleine/mcp_server/server.py:123-130` | `_engine_now` builds a throwaway cue to read the clock "because `MemoryEngine` does not expose its clock". The contract now has `MemoryEngine.now()`. | By inspection. | Use `engine.now()`. |
| R30 | nit | semantics | `src/madeleine/engine.py:289` | `recall` (documented "no side effects") calls `_ensure_registered`, which writes `channels` rows and vec tables, and can raise the model-change `StoreError` on a read path. | By inspection. | Accept this and document it, or register only on `remember`, treating unknown channels in `nearest` as empty (it already does). |
| R31 | nit | CI supply chain | `.github/workflows/release.yml:18-19,49,60` | Actions are pinned by mutable tags (`checkout@v5`, `setup-uv@v6`, `gh-action-pypi-publish@release/v1`), while `cosign-installer` is pinned by SHA. These jobs hold `id-token: write`. | By inspection. | Pin the release-job actions by commit SHA. |
| R32 | nit | UX | `src/madeleine/channels/registry.py:128`, `src/madeleine/channels/text.py:218` | Every CLI command (`remember`, `recall`…) prints three WARNING lines about missing extras on a base install. | `madeleine remember …` output. | Log optional-channel skips at INFO unless the channel was explicitly requested in `MADELEINE_CHANNELS`. |

## Checked and found fine

- **Quality gates.** `ruff check` and `ruff format --check` are clean, `mypy --strict` reports no
  issues (49 files), and `pytest -q` gives 441 passed and 3 deselected (`models`).
- **Fusion conformance (ADR 0001).** All of these match the normative formula exactly:
  - `base` over S, and `coverage = (ΣS/ΣC)^exp` (the `min(1, ·)` is a harmless no-op).
  - `salience_factor = 1 + w·(s − 0.5)`.
  - Decay with floor, using `age = max(0, Δ)`.
  - Final clamp to [0, 1], with NaN mapped to 0.
  - Zero-weight and non-finite weights are excluded from both S and C.
  - `RecallQuery.channels` restricts both S and C.
- **PUSH rules (ADR 0001).**
  - A memory fires only if `score ≥ trigger_threshold`, `|S| ≥ min_shared_channels` and some
    channel in S has `solo_trigger`.
  - Refractory is symmetric (`abs`), and a period of 0 disables it.
  - At most `max_push_per_cue` fire, highest first. Only memories that fire are claimed.
  - Ties break by newest. PULL applies no refractory filter.
- **Default channel specs** match the ARCHITECTURE table: time has weight 0.5 and
  `solo_trigger=False`; biometric has weight 0.5 and `solo_trigger=False`.
- **SQL injection.** All user data is bound. Interpolated fragments are integer vec-table ids,
  placeholder lists and constants. Channel names are regex-validated
  (`[a-z][a-z0-9_]*`). Table names never come from channel names.
- **Auth.**
  - The bearer token is compared with `hmac.compare_digest` (constant time) and the scheme is
    case-insensitive.
  - The token is required on every route, including mounted `/mcp`, `/docs` and
    `/openapi.json`; only `/healthz` is public, and `/healthz/` and `//healthz` are not.
  - A wrong token returns 401, and the 401 comes before the body is read.
  - The default bind is `127.0.0.1`.
- **`/mcp` DNS-rebinding protection** works: a bad Host returns 421 and a bad Origin returns 403.
- **Body limits.** Content-Length over 10 MiB and chunked bodies both return 413, on REST and on
  `/mcp`.
- **Input validation.** About 35 malformed payloads across `/cues`, `/cues/{generic,
  home_assistant, ios_shortcuts}`, `/memories` and `/recall` were tried: NaN/inf, out-of-range
  lat/lon/salience, wrong types, unknown channels, bad base64, huge epochs, year 1 and year 9999
  timestamps, and so on. All return 4xx except R12.
- **No SSRF.** Nothing fetches URLs. `http(s)`, `s3` and remote `file://host` are rejected. The
  hook's `--url` is limited to http(s) and skips proxies for loopback.
- **Queue deserialisation** is JSON only (no pickle) through a validating codec. Undecodable rows
  are logged and dropped.
- **Raw-media privacy.**
  - `store_raw=False` (the default) strips `data` before storing; only `store_raw=True` writes
    `raw_blobs`.
  - Queue payloads never contain `data_b64`.
  - `Memory.to_dict` excludes observations and vectors.
  - Cues are not persisted.
  - Logs contain only ids and counts, at DEBUG.
- **`forget` completeness at the row level.** memories, vectors, raw_blobs, triggers, queue and
  vec0 rows (`vec_chN`, `_rowids`) are all 0 afterwards (byte residue is R7).
- **Stdio hygiene.** Every stdout line was valid JSON-RPC across initialize, tools/list, tool
  calls, errors and resources/read. Warnings go to stderr.
- **Lazy imports.**
  - Importing `madeleine`, `.engine`, `.channels`, `.mcp_server`, `.ingest`,
    `.integrations.claude_code` and `.evals` (each in a fresh subprocess) loads no
    torch/fastapi/mcp/uvicorn.
  - `create_engine()` loads no torch.
  - Missing extras skip the image and audio channels with a pip hint, and requests that use them
    get 400.
- **Concurrency.** Eight processes sensing the same cue on one SQLite file fired exactly 3 (the
  expected count). Four concurrent drainers got 3 recollections with no duplicates. WAL and
  `BEGIN IMMEDIATE` behave as documented.
- **Timezones and float32.**
  - All timestamps are aware. Naive input is read as UTC (documented). Offsets survive the store
    round trip.
  - Vectors are rounded to float32 before both scoring and storage, so in-memory results match
    persisted ones.
  - The sqlite-vec fallback works within a single mode.
- **Hook.**
  - Always exits 0.
  - Output stays ≤ 9 500 chars.
  - The prompt cue is truncated at 4 000 chars.
  - `--limit` is validated.
- **Packaging.**
  - The wheel contains `py.typed`, the `madeleine` and `madeleine-mcp` entry points, the license
    and the extras.
  - `uvx --from <wheel> madeleine-mcp demo` runs offline.
  - The package builds and imports with `mcp==2.0.0` on Python 3.11.
  - `madeleine demo` cleans up its temp dir.
  - CLI `--help` texts are present and accurate, apart from `--allow-file-uris`, which says
    "MCP clients" (see R2).
- **Module seams** match ARCHITECTURE: `create_engine`, `default_registry`/`hash_registry`,
  `create_server`/`run_stdio`, `mcp_asgi_app` returning `(app, lifespan)`, `create_app(...)`,
  the hook flags, and `run_bench`. The MCP tool names, inputs and resource
  `recollections://pending` match the documented surface. Time is included in sense (MCP, HTTP,
  CLI, hook) and only when `at` is given in recall (MCP, HTTP, CLI), consistently.

## Resolution (lead, after the review)

All 32 findings were addressed on `fmv/dazzling-bell-jmzwuo`, each with a regression test
(635 tests pass, up from 441). The table gives the status and where the fix lives.

| ID | Status | Fix |
|---|---|---|
| R1 | fixed | REST guard: Host allow-list (421) and Origin allow-list (403) on every route, including `/mcp` and `/healthz`. Loopback is allowed by default, plus `MADELEINE_ALLOWED_HOSTS`. `serve` passes the bind host. |
| R2, R3 | fixed | `MadeleineConfig.allow_local_files` (default off) is enforced **inside** the image/audio adapters. It covers `uri` and string `value` and is checked before any filesystem access, with the same message whether or not the file exists. The hash embedder never reads files. MCP keeps its guard as defence in depth. `serve --allow-file-uris` sets the flag for every surface. |
| R4 | fixed | Full README: quickstart, the `[text]` caveat, privacy, benchmark, limitations. |
| R5 | fixed (unverified live) | `<!-- mcp-name: io.github.FabianIMV/madeleine -->` is in the README and therefore in the wheel `METADATA`. Not yet tested against the live registry. |
| R6 | fixed | `GET /recollections/pending` only peeks. Draining is `POST /recollections/drain`. The hook uses the POST. |
| R7 | fixed | `PRAGMA secure_delete=ON` on every connection, plus a WAL checkpoint after `delete`. A test greps the DB files for content and raw markers. |
| R8 | fixed | The hook peeks, fits, then drains exactly what it shows. Lines are truncated to 1000 chars, so nothing is lost. |
| R9 | fixed | Per-channel candidates are `max(candidate_k, k)`. Tags filter before KNN (`nearest_tagged`). |
| R10 | fixed | `StoreError` returns 409; any other `MadeleineError` returns 500 with its message. |
| R11 | mitigated | Recollections are wrapped in `<madeleine-recollections>` delimiters and labelled untrusted. Control and bidi characters are stripped and delimiters inside content are neutralised. The trust model is documented in SECURITY.md and the Claude Code example. |
| R12 | fixed | `parse_timestamp` rejects non-strings, and adapter field checks return 422. A fuzz test shows no 500s. |
| R13 | fixed | `claim_trigger` checks every trigger in the window, not just the latest. |
| R14 | fixed | `nearest` checks `vec_dirty` and uses brute force until the index is rebuilt. |
| R15 | fixed | A claim requires the memory to exist. Delete purges queue rows in the same transaction, and the engine holds one lock across claim and put. |
| R16 | fixed | The data directory is `0700` and the DB, `-wal` and `-shm` files are `0600`. Existing world-readable files are tightened. |
| R17 | fixed | Local media files are size-checked after the policy check and again before embedding. |
| R18 | fixed | Empty cues are rejected everywhere. Unknown recall channels return 422 on HTTP and a tool error on MCP. `query` is literal. MCP `remember` takes `metadata`. `source` is capped at 64 chars everywhere. Both surfaces return `prompt_line`. |
| R19 | fixed | URI descriptions and messages no longer suggest remote URLs. |
| R20 | fixed | Queue rows (schema 2) hold the id plus score explanation, and the memory is rehydrated on read. Includes a v1 to v2 migration. |
| R21 | fixed | Config overrides are validated. CLI errors print a message without a traceback. `--lat` and `--lon` must be given together. |
| R22 | fixed | The hook supports `--token-file` and the env var. `--token` is deprecated and warns. The hook warns when a token would go over cleartext HTTP to a non-loopback host. |
| R23 | fixed | `serve` refuses a non-loopback bind without `MADELEINE_API_TOKEN` unless `--insecure-no-auth` is given. |
| R24 | fixed | The version is defined once in `madeleine.__version__` (hatch dynamic). The release workflow checks the tag, `__init__` and `server.json`. |
| R25 | fixed | Direct dependencies are declared, `mcp`/`mcp-types` are pinned to `>=2.3,<2.4`, and a test fails loudly if the SDK internals move. |
| R26 | fixed | `pending_recollections.limit` is capped with `le=1000`. |
| R27 | fixed | The sdist excludes CLAUDE.md, PLAN.md, REVIEW.md, `.github`, `.claude` and `uv.lock`. |
| R28 | fixed | ARCHITECTURE.md and the docstrings now match the code. |
| R29 | fixed | `engine.now()` is used. |
| R30 | fixed | Only `remember` registers channels. Reads call a read-only `check_channel`. |
| R31 | fixed | Every third-party action is pinned by commit SHA. |
| R32 | fixed | Skipped optional channels are logged at INFO unless they were explicitly requested. |

Deliberately deferred:

* `nearest_tagged` and `check_channel` are optional store extensions that the engine detects
  at runtime, with fallbacks. They are not yet in the `MemoryStore` protocol; adding them
  needs an ADR.
* There is no `madeleine vacuum` command. It is only relevant to databases created before
  R7, and none exist outside development.
* `FakeEngine` keeps simplified refractory semantics. It is a test double for consumers,
  not for the engine.
