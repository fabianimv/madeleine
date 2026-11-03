"""Compose the runnable servers.

* ``stdio``: MCP over stdio, for clients that spawn the server (Claude Code/Desktop...).
* ``http``: one uvicorn process serving the HTTP ingestion API (``/cues``, ``/memories``,
  ``/recollections/...``) and MCP streamable HTTP at ``/mcp/``, sharing one engine.

Both open the same SQLite file, so a stdio MCP server and an HTTP ingestion server started
separately still share memories and the involuntary-recollection queue.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class InsecureBindError(RuntimeError):
    """Refusing to expose memories on the network without a token."""


def build_http_app(engine: Any, *, host: str = "127.0.0.1") -> Any:
    """FastAPI app with the ingestion routes and MCP mounted at ``/mcp/``.

    ``host`` feeds both DNS-rebinding protections (the MCP SDK's and the REST guard): the
    loopback names are always allowed, a non-loopback bind address is added.
    """
    from madeleine.ingest.app import create_app
    from madeleine.mcp_server.http import mcp_asgi_app

    allow_files = bool(engine.config.allow_local_files)
    mcp_app, mcp_lifespan = mcp_asgi_app(engine, host=host, allow_file_uris=allow_files)
    extra: dict[str, Any] = {}
    if host not in LOOPBACK_HOSTS:
        # TODO(R1 merge): pass allowed_hosts directly once create_app gains the parameter.
        extra["allowed_hosts"] = [*sorted(LOOPBACK_HOSTS), host]
    return create_app(engine, mounts={"/mcp": mcp_app}, lifespans=[mcp_lifespan], **extra)


def check_bind(host: str, *, insecure_no_auth: bool = False) -> None:
    """Refuse a non-loopback bind without ``MADELEINE_API_TOKEN`` unless explicitly allowed."""
    if host in LOOPBACK_HOSTS:
        return
    if os.environ.get("MADELEINE_API_TOKEN"):
        return
    if insecure_no_auth:
        logger.warning(
            "binding to %s WITHOUT authentication: anyone on the network can read, "
            "write and delete your memories",
            host,
        )
        return
    raise InsecureBindError(
        f"refusing to bind to {host!r} without MADELEINE_API_TOKEN; set a token "
        "(recommended) or pass --insecure-no-auth"
    )


def serve(
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
    db: str | None = None,
    text_backend: str | None = None,
    allow_file_uris: bool = False,
    insecure_no_auth: bool = False,
) -> None:
    """Run until interrupted."""
    from madeleine.config import load_config
    from madeleine.engine import create_engine

    if transport == "http":
        check_bind(host, insecure_no_auth=insecure_no_auth)
    elif transport != "stdio":
        raise ValueError(f"unknown transport {transport!r}")
    config = load_config(
        db_path=db, text_backend=text_backend, allow_local_files=True if allow_file_uris else None
    )
    engine = create_engine(config)
    try:
        if transport == "stdio":
            from madeleine.mcp_server.server import run_stdio

            run_stdio(engine, allow_file_uris=config.allow_local_files)
        else:
            import uvicorn

            uvicorn.run(
                build_http_app(engine, host=host), host=host, port=port, log_level="warning"
            )
    finally:
        engine.close()
