# Security and privacy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's **"Report a vulnerability"** button
(Security → Advisories) on <https://github.com/fabianimv/madeleine>. Do not open a public
issue. We aim to acknowledge reports within 7 days. madeleine is pre-1.0, and only the latest
release receives fixes.

## What madeleine stores

madeleine is designed to hold personal context: where you were, when, what you heard, saw and
said. Treat its database like a diary.

| Data | Stored? |
|---|---|
| Memory content, timestamps, geohash, tags, salience, metadata | Yes, in SQLite (`~/.madeleine/madeleine.db` by default) |
| Per-channel embeddings | Yes. Text embeddings are partly invertible: treat them as sensitive. |
| Raw image / audio bytes | **No**, unless `store_raw=True` (per call) or `MADELEINE_STORE_RAW=1` |
| Media URIs | Only the reference string, and only if local files are allowed |
| Cues | **No.** A recollection records only the cue id, source and channel names. |
| Telemetry | None. madeleine never phones home. |

The data directory is created owner-only (`0700`), and database files are `0600`. `forget`
deletes the memory, its vectors, raw bytes, trigger history and queued recollections, with
SQLite `secure_delete` enabled so freed pages are zeroed. Backups, filesystem snapshots and
copies you made remain your responsibility.

## Threat model and defaults

* **Local files.** By default no surface (MCP, HTTP or webhooks) can make madeleine read a
  file from disk. Media must arrive as bytes. `serve --allow-file-uris` /
  `MADELEINE_ALLOW_LOCAL_FILES=1` opts in. Only do this when every client is trusted.
* **No network fetches.** madeleine never downloads media from URLs (no SSRF). Models are
  downloaded by their libraries on first use, only when the corresponding extra is
  installed.
* **HTTP server.**
  * It binds to `127.0.0.1` by default.
  * The `Host` and `Origin` headers are checked against an allow-list (loopback names plus
    `MADELEINE_ALLOWED_HOSTS`) on every route, including `/mcp`. This blocks DNS-rebinding
    attacks from web pages.
  * Set `MADELEINE_API_TOKEN` to require `Authorization: Bearer <token>` on every route
    except `/healthz`. The token is compared in constant time.
  * Binding to a non-loopback address without a token is refused unless you pass
    `--insecure-no-auth`.
  * Request bodies are capped (10 MiB by default, `MADELEINE_MAX_BODY_BYTES`).
* **State-changing requests are POST/DELETE.** `GET /recollections/pending` only peeks;
  draining is `POST /recollections/drain`.
* **Prompt injection.** Recollections are injected into your agent's context. Anyone who can
  write a memory (HTTP clients, webhooks, the MCP `remember` tool) can therefore put text in
  front of your agent later. The Claude Code hook wraps recollections in explicit delimiters
  and labels them as untrusted data, but this is a mitigation, not a guarantee. Only give
  write access to sources you trust.
* **Hook tokens.** Prefer `MADELEINE_API_TOKEN` or `--token-file` over `--token`, because
  command-line arguments are visible in process listings. The hook warns when a token would
  travel over plain HTTP to a non-loopback host.
* **No multi-user isolation.** Everyone who can reach the server, or read the database file,
  sees all memories.

## Supply chain

* Runtime dependencies are declared explicitly, and the MCP SDK is pinned to the tested minor
  version.
* Release workflows pin third-party actions by commit SHA and publish to PyPI through Trusted
  Publishing (no long-lived tokens).
* ImageBind is never a dependency (its license is CC-BY-NC).
