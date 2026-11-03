"""Compose the runnable servers.

* ``stdio``: MCP over stdio, for clients that spawn the server (Claude Code/Desktop...).
* ``http``: one uvicorn process serving the HTTP ingestion API (``/cues``, ``/memories``,
  ``/recollections/pending``...) and MCP streamable HTTP at ``/mcp``, sharing one engine.

Both open the same SQLite file, so a stdio MCP server and an HTTP ingestion server started
separately still share memories and the involuntary-recollection queue.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_http_app(engine: Any, *, host: str = "127.0.0.1", allow_file_uris: bool = False) -> Any:
    """FastAPI app with the ingestion routes and MCP mounted at ``/mcp`` (clients should use
    ``/mcp/``; ``host`` configures the SDK's DNS-rebinding protection)."""
    from madeleine.ingest.app import create_app
    from madeleine.mcp_server.http import mcp_asgi_app

    mcp_app, mcp_lifespan = mcp_asgi_app(engine, host=host, allow_file_uris=allow_file_uris)
    return create_app(engine, mounts={"/mcp": mcp_app}, lifespans=[mcp_lifespan])


def serve(
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
    db: str | None = None,
    text_backend: str | None = None,
    allow_file_uris: bool = False,
) -> None:
    """Run until interrupted."""
    from madeleine.config import load_config
    from madeleine.engine import create_engine

    engine = create_engine(load_config(db_path=db, text_backend=text_backend))
    try:
        if transport == "stdio":
            from madeleine.mcp_server.server import run_stdio

            run_stdio(engine, allow_file_uris=allow_file_uris)
        elif transport == "http":
            import uvicorn

            if host not in {"127.0.0.1", "localhost", "::1"}:
                logger.warning(
                    "binding to %s exposes your memories on the network; set "
                    "MADELEINE_API_TOKEN to require a bearer token",
                    host,
                )
            app = build_http_app(engine, host=host, allow_file_uris=allow_file_uris)
            uvicorn.run(app, host=host, port=port, log_level="warning")
        else:
            raise ValueError(f"unknown transport {transport!r}")
    finally:
        engine.close()
