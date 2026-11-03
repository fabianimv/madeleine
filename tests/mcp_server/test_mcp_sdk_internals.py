"""Canary for the private MCP SDK internals madeleine relies on (R25).

madeleine uses two non-public SDK details, each with a guarded fallback that would
otherwise degrade *silently*:

* ``MCPServer._lowlevel_server.add_request_handler`` - to serve handshake-era
  ``resources/subscribe`` / ``resources/unsubscribe`` (server.py ``_register_resources``).
* ``ServerSession._connection`` (an ``mcp.server.connection.Connection`` with an
  ``exit_stack``) - to key legacy subscribers by connection and drop them on disconnect
  (notifier.py ``_connection_key``).

If an SDK upgrade removes or renames them, these tests fail with an explicit message.
"""

from __future__ import annotations

from typing import Any

import pytest

from madeleine.mcp_server import create_server
from madeleine.mcp_server.notifier import _connection_key
from madeleine.testing import FakeEngine
from tests.mcp_server.test_mcp_notifications import LegacyInbox, legacy_client

SDK_CHANGED = (
    "The installed mcp SDK no longer provides {what}, a private API madeleine depends on. "
    "Update {where} for this SDK version (or pin mcp to a version that has it)."
)


def test_lowlevel_server_request_handler_api_exists() -> None:
    server = create_server(FakeEngine())
    lowlevel = getattr(server, "_lowlevel_server", None)
    if lowlevel is None or not callable(getattr(lowlevel, "add_request_handler", None)):
        pytest.fail(
            SDK_CHANGED.format(
                what="MCPServer._lowlevel_server.add_request_handler",
                where="madeleine.mcp_server.server.MadeleineMCPServer._register_resources",
            )
        )


def test_connection_class_api_exists() -> None:
    try:
        from mcp.server.connection import Connection
    except ImportError:  # pragma: no cover - only on an incompatible SDK
        pytest.fail(
            SDK_CHANGED.format(
                what="mcp.server.connection.Connection",
                where="madeleine.mcp_server.notifier",
            )
        )
    if "exit_stack" not in dir(Connection) and "exit_stack" not in getattr(
        Connection, "__annotations__", {}
    ):
        pytest.fail(
            SDK_CHANGED.format(
                what="Connection.exit_stack", where="madeleine.mcp_server.notifier.add_legacy"
            )
        )


async def test_server_session_exposes_its_connection() -> None:
    server = create_server(FakeEngine(), poll_interval=30)
    sessions: list[Any] = []
    real_add = server.notifier.add_legacy

    def spy(session: Any, uri: str) -> None:
        sessions.append(session)
        real_add(session, uri)

    server.notifier.add_legacy = spy  # type: ignore[method-assign]
    async with legacy_client(server, LegacyInbox()):
        assert sessions, "resources/subscribe never reached the notifier"
        key, connection = _connection_key(sessions[0])
        if connection is None:
            pytest.fail(
                SDK_CHANGED.format(
                    what="ServerSession._connection (a Connection)",
                    where="madeleine.mcp_server.notifier._connection_key",
                )
            )
        assert key is connection
        assert hasattr(connection, "exit_stack")
