# ADR 0002: Dependency footprint and MCP SDK version

- Status: Accepted
- Date: 2026-11-03
- Deciders: lead

## Context

The mission asks for `uvx madeleine-mcp demo` to run end-to-end offline and for a "small
sentence-transformers model (lazy-loaded)" for text. `sentence-transformers` depends on
`torch` (hundreds of MB to GBs). Making it a base dependency would make every `uvx` launch of
the MCP server download torch, even for users who only need time/geo cues or the demo.

The official MCP Python SDK is at 2.x (2.3.0 at the time of writing). In 2.x the `FastMCP`
class was renamed `MCPServer` (`mcp.server.mcpserver`), and several APIs changed.

## Decision

1. Base install: `numpy`, `sqlite-vec`, `mcp>=2,<3`, `fastapi`, `uvicorn`, `typer`. No ML.
2. Extras:
   - `madeleine-mcp[text]` → `sentence-transformers` (default model
     `sentence-transformers/all-MiniLM-L6-v2`, 384-d).
   - `madeleine-mcp[image]` → `open-clip-torch` (+ pillow, torch).
   - `madeleine-mcp[audio]` → `transformers` CLAP (+ torch, soundfile).
   - `madeleine-mcp[all]`.
3. `text_backend="auto"` (default) uses sentence-transformers when importable, otherwise the
   deterministic `HashEmbedder` (lexical feature hashing) with a one-time warning. `"hash"`
   forces the offline embedder (tests, demo, CI).
4. Use the MCP SDK 2.x API (`MCPServer`). The term "FastMCP" in older docs refers to the same
   high-level server API.
5. Geohash encode/decode is implemented in-tree (tiny, pure Python) rather than depending on
   a third-party geohash package.

## Consequences

* `uvx madeleine-mcp demo` installs in seconds and works offline.
* Real semantic text recall requires `uvx --from 'madeleine-mcp[text]' madeleine serve`;
  README must say so prominently. Benchmarks report both backends where possible.
* Switching text backend on an existing database is refused by the store (different
  `model_id`); users re-create or re-embed (re-embed tooling is future work).
