"""MCP server for madeleine (tools, the ``recollections://pending`` resource, transports).

Importing this package is cheap; the MCP SDK is imported only when a name is used::

    from madeleine.mcp_server import create_server, run_stdio, mcp_asgi_app
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from madeleine.mcp_server.http import mcp_asgi_app, run_http
    from madeleine.mcp_server.notifier import PENDING_URI
    from madeleine.mcp_server.server import MadeleineMCPServer, create_server, run_stdio

__all__ = [
    "PENDING_URI",
    "MadeleineMCPServer",
    "create_server",
    "mcp_asgi_app",
    "run_http",
    "run_stdio",
]

_LAZY = {
    "PENDING_URI": "madeleine.mcp_server.notifier",
    "MadeleineMCPServer": "madeleine.mcp_server.server",
    "create_server": "madeleine.mcp_server.server",
    "run_stdio": "madeleine.mcp_server.server",
    "mcp_asgi_app": "madeleine.mcp_server.http",
    "run_http": "madeleine.mcp_server.http",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)
