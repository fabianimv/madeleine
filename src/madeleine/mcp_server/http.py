"""MCP over streamable HTTP.

:func:`mcp_asgi_app` returns an ASGI app plus a lifespan factory so the MCP endpoint can be
mounted inside another application (the ingestion FastAPI app mounts it at ``/mcp``)::

    mcp_app, mcp_lifespan = mcp_asgi_app(engine)
    app = create_app(engine, mounts={"/mcp": mcp_app}, lifespans=[mcp_lifespan])

Starlette does not run the lifespan of mounted sub-apps, hence the separate factory: the
host app must enter it (it starts the MCP session manager and the recollection notifier).
The endpoint is the mount root, so clients connect to ``http://host:port/mcp/``.

:func:`run_http` serves only MCP (at ``/mcp``) with uvicorn.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Any

from starlette.types import ASGIApp

from madeleine.core.types import MemoryEngine
from madeleine.mcp_server.server import create_server

__all__ = ["mcp_asgi_app", "run_http"]

logger = logging.getLogger(__name__)


def mcp_asgi_app(
    engine: MemoryEngine,
    *,
    host: str = "127.0.0.1",
    json_response: bool = False,
    **kwargs: Any,
) -> tuple[ASGIApp, Callable[[], AbstractAsyncContextManager[None]]]:
    """Return ``(app, lifespan)`` for mounting MCP streamable HTTP inside another app.

    * ``app`` serves the MCP endpoint at its root path (mount it at ``/mcp``).
    * ``lifespan()`` must be entered by the host application's lifespan, exactly once.
    * ``host`` is the interface the host app binds: for loopback hosts DNS-rebinding
      protection (Host/Origin checks) is enabled, as the MCP spec recommends.
    * ``kwargs`` go to :func:`~madeleine.mcp_server.server.create_server`
      (``poll_interval``, ``allow_file_uris``...).
    """
    server = create_server(engine, **kwargs)
    app = server.streamable_http_app(
        streamable_http_path="/", host=host, json_response=json_response
    )

    def lifespan() -> AbstractAsyncContextManager[None]:
        return server.session_manager.run()

    return app, lifespan


def run_http(
    engine: MemoryEngine, host: str = "127.0.0.1", port: int = 8765, **kwargs: Any
) -> None:
    """Serve only MCP over streamable HTTP at ``http://host:port/mcp`` (blocking)."""
    import uvicorn

    server = create_server(engine, **kwargs)
    app = server.streamable_http_app(streamable_http_path="/mcp", host=host)
    logger.info("madeleine MCP server on http://%s:%d/mcp", host, port)
    # log_config=None: uvicorn's default config sends access logs to stdout.
    uvicorn.run(app, host=host, port=port, log_config=None, access_log=False)
