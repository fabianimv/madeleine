"""Helpers shared by the mcp_server tests."""

from __future__ import annotations

import json
from typing import Any

from mcp import Client
from mcp_types import CallToolResult, TextContent

# Both protocol eras the SDK serves: "auto" negotiates 2026-07-28 in-process, "legacy"
# forces the 2025 initialize handshake over in-memory streams.
MODES = ("auto", "legacy")


def payload(result: CallToolResult) -> Any:
    """The JSON a tool returned (its single text block)."""
    assert not result.is_error, text(result)
    return json.loads(text(result))


def text(result: CallToolResult) -> str:
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, TextContent)
    return block.text


async def call(client: Client, name: str, args: dict[str, Any] | None = None) -> Any:
    return payload(await client.call_tool(name, args or {}))


async def call_error(client: Client, name: str, args: dict[str, Any]) -> str:
    result = await client.call_tool(name, args)
    assert result.is_error, f"expected a tool error, got {text(result)}"
    return text(result)
