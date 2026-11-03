"""The MCP server against the real engine (skipped until the engine and channels exist)."""

from __future__ import annotations

from typing import Any

import pytest
from mcp import Client

from madeleine.core.types import MadeleineConfig
from madeleine.mcp_server import create_server
from tests.mcp_server.mcp_helpers import call, call_error


def _real_engine() -> Any:
    engine_mod = pytest.importorskip("madeleine.engine")
    registry_mod = pytest.importorskip("madeleine.channels.registry")
    return engine_mod.create_engine(
        MadeleineConfig(db_path=":memory:", text_backend="hash"),
        registry=registry_mod.hash_registry(),
    )


async def test_round_trip_with_real_engine() -> None:
    engine = _real_engine()
    try:
        async with Client(create_server(engine)) as client:
            mem = await call(
                client,
                "remember",
                {
                    "content": "Ana proposed at the Eiffel Tower",
                    "geohash": "u09tunq",
                    "at": "2024-06-01T21:30:00+02:00",
                    "salience": 0.9,
                },
            )
            await call(client, "remember", {"content": "groceries: milk and eggs"})
            found = await call(client, "recall", {"query": "Ana proposed at the Eiffel Tower"})
            assert found[0]["memory"]["id"] == mem["id"]
            by_place = await call(client, "recall", {"geohash": "u09tunq"})
            assert by_place[0]["memory"]["id"] == mem["id"]

            fired = await call(
                client,
                "sense",
                {
                    "text": "Ana proposed at the Eiffel Tower",
                    "geohash": "u09tunq",
                    "at": "2026-06-01T21:30:00+02:00",
                },
            )
            assert mem["id"] in {r["memory"]["id"] for r in fired}
            drained = await call(client, "pending_recollections")
            assert {r["id"] for r in drained} == {r["id"] for r in fired}
            assert await call(client, "pending_recollections") == []
            assert "geohash" in await call_error(client, "recall", {"geohash": "nope!"})
            assert await call(client, "forget", {"memory_id": mem["id"]}) == {"deleted": True}
    finally:
        engine.close()
