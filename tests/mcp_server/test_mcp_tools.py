"""Tools and resources through an in-process MCP client (both protocol eras)."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any

import pytest
from mcp import Client
from mcp_types import TextResourceContents

from madeleine.mcp_server import PENDING_URI, create_server
from madeleine.testing import FakeClock, FakeEngine
from tests.mcp_server.mcp_helpers import MODES, call, call_error, payload, text

TOOLS = {"remember", "recall", "sense", "pending_recollections", "forget"}
EIFFEL = "u09tunq"


@pytest.fixture
def engine(clock: FakeClock) -> FakeEngine:
    return FakeEngine(clock=clock)


@pytest.fixture(params=MODES)
def mode(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@asynccontextmanager
async def connect(engine: FakeEngine, mode: str = "auto", **kwargs: Any) -> AsyncIterator[Client]:
    # Not a yield-fixture: the client's task group must open and close in the test's task.
    async with Client(create_server(engine, poll_interval=0.05, **kwargs), mode=mode) as c:
        yield c


async def test_server_identity_and_instructions(engine: FakeEngine, mode: str) -> None:
    async with connect(engine, mode) as client:
        assert client.instructions is not None
        assert "involuntary" in client.instructions
        assert "pending_recollections" in client.instructions
        caps = client.server_capabilities
        assert caps.tools is not None
        assert caps.resources is not None
        assert caps.resources.subscribe is True


async def test_tools_list_exposes_exactly_the_five_tools(engine: FakeEngine, mode: str) -> None:
    async with connect(engine, mode) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        assert set(tools) == TOOLS
        for tool in tools.values():
            assert tool.description
            assert tool.input_schema["type"] == "object"
        assert tools["remember"].input_schema["required"] == ["content"]
        assert tools["forget"].input_schema["required"] == ["memory_id"]
        remember_props = tools["remember"].input_schema["properties"]
        assert {"content", "geohash", "lat", "lon", "at", "tags", "salience", "signals"} <= set(
            remember_props
        )
        assert "media_ref" in remember_props
        recall_props = tools["recall"].input_schema["properties"]
        assert {"query", "geohash", "lat", "lon", "at", "channels", "tags", "k"} <= set(
            recall_props
        )
        assert recall_props["k"]["default"] == 5
        sense_props = tools["sense"].input_schema["properties"]
        assert {"text", "geohash", "lat", "lon", "at", "biometric", "signals", "source"} <= set(
            sense_props
        )
        pending_props = tools["pending_recollections"].input_schema["properties"]
        assert pending_props["drain"]["default"] is True
        assert tools["recall"].annotations is not None
        assert tools["recall"].annotations.read_only_hint is True
        assert tools["forget"].annotations is not None
        assert tools["forget"].annotations.destructive_hint is True


async def test_resources_list_has_pending(engine: FakeEngine, mode: str) -> None:
    async with connect(engine, mode) as client:
        resources = (await client.list_resources()).resources
        assert [str(r.uri) for r in resources] == [PENDING_URI]
        assert resources[0].mime_type == "application/json"


async def test_remember_then_recall_round_trip(engine: FakeEngine, mode: str) -> None:
    async with connect(engine, mode) as client:
        mem = await call(
            client,
            "remember",
            {
                "content": "Ana proposed at the Eiffel Tower",
                "geohash": EIFFEL,
                "at": "2024-06-01T21:30:00+02:00",
                "tags": ["family"],
                "salience": 0.9,
            },
        )
        assert mem["id"].startswith("mem_")
        assert mem["content"] == "Ana proposed at the Eiffel Tower"
        assert mem["context"]["geohash"] == EIFFEL
        assert mem["context"]["timestamp"] == "2024-06-01T21:30:00+02:00"
        assert mem["context"]["tags"] == ["family"]
        assert mem["context"]["salience"] == 0.9
        assert {"geo", "text", "time"} <= set(mem["channels"])
        await call(client, "remember", {"content": "bought bread at the bakery"})

        by_text = await call(client, "recall", {"query": "Eiffel Tower proposal"})
        assert by_text[0]["memory"]["id"] == mem["id"]
        assert set(by_text[0]["channel_scores"]) == {"text"}  # no time unless `at` given

        by_place = await call(client, "recall", {"geohash": EIFFEL, "channels": ["geo"]})
        assert by_place[0]["memory"]["id"] == mem["id"]
        assert set(by_place[0]["channel_scores"]) == {"geo"}

        by_tag = await call(client, "recall", {"query": "bread bakery tower", "tags": ["family"]})
        assert [r["memory"]["id"] for r in by_tag] == [mem["id"]]

        with_time = await call(
            client, "recall", {"query": "Eiffel", "at": "2026-06-02T21:00:00+02:00"}
        )
        assert "time" in with_time[0]["channel_scores"]

        limited = await call(client, "recall", {"query": "the", "k": 1})
        assert len(limited) <= 1


async def test_lat_lon_is_encoded_to_geohash(engine: FakeEngine, mode: str) -> None:
    async with connect(engine, mode) as client:
        mem = await call(client, "remember", {"content": "picnic", "lat": 48.8584, "lon": 2.2945})
        assert mem["context"]["geohash"].startswith("u09tun")


async def test_remember_with_signals(engine: FakeEngine, mode: str) -> None:
    async with connect(engine, mode) as client:
        png = base64.b64encode(b"\x89PNG fake").decode()
        mem = await call(
            client,
            "remember",
            {
                "content": "sunset photo",
                "signals": [
                    {"channel": "image", "data_b64": png, "mime_type": "image/png"},
                    {"channel": "biometric", "value": {"heart_rate": 72}},
                ],
            },
        )
        assert {"image", "biometric", "text"} <= set(mem["channels"])
        assert "data_b64" not in json.dumps(mem)  # raw bytes are never echoed


async def test_sense_fires_and_pending_drains(mode: str, engine: FakeEngine) -> None:
    async with connect(engine, mode) as client:
        mem = await call(client, "remember", {"content": "Ana proposed at the Eiffel Tower"})
        fired = await call(client, "sense", {"text": "the Eiffel Tower", "geohash": EIFFEL})
        assert len(fired) == 1
        rec = fired[0]
        assert rec["id"].startswith("rec_")
        assert rec["mode"] == "push"
        assert rec["cue_source"] == "mcp"
        assert "time" in rec["cue_channels"]  # sense always includes time
        assert rec["memory"]["id"] == mem["id"]

        peek = await call(client, "pending_recollections", {"drain": False})
        assert [r["id"] for r in peek] == [rec["id"]]
        resource = await client.read_resource(PENDING_URI)
        contents = resource.contents[0]
        assert isinstance(contents, TextResourceContents)
        assert contents.mime_type == "application/json"
        assert [r["id"] for r in json.loads(contents.text)] == [rec["id"]]

        drained = await call(client, "pending_recollections")
        assert [r["id"] for r in drained] == [rec["id"]]
        assert await call(client, "pending_recollections") == []
        assert engine.queue.pending_count() == 0

        # Habituation: the same cue right away does not fire again.
        assert await call(client, "sense", {"text": "the Eiffel Tower", "source": "test"}) == []


async def test_sense_source_biometric_and_limit(
    mode: str, engine: FakeEngine, clock: FakeClock
) -> None:
    async with connect(engine, mode) as client:
        for i in range(3):
            await call(client, "remember", {"content": f"tea and cake {i}"})
        fired = await call(
            client,
            "sense",
            {"text": "tea and cake", "biometric": {"heart_rate": 80}, "source": "x"},
        )
        assert fired
        assert all(r["cue_source"] == "x" for r in fired)
        assert all("biometric" in r["cue_channels"] for r in fired)
        first = await call(client, "pending_recollections", {"limit": 1})
        assert len(first) == 1
        assert engine.queue.pending_count() == len(fired) - 1
        clock.advance(timedelta(hours=2))


async def test_forget(mode: str, engine: FakeEngine) -> None:
    async with connect(engine, mode) as client:
        mem = await call(client, "remember", {"content": "a secret"})
        await call(client, "sense", {"text": "a secret"})
        assert engine.queue.pending_count() == 1
        assert await call(client, "forget", {"memory_id": mem["id"]}) == {"deleted": True}
        assert engine.queue.pending_count() == 0
        assert await call(client, "forget", {"memory_id": mem["id"]}) == {"deleted": False}
        assert await call(client, "recall", {"query": "a secret"}) == []


async def test_structured_content_mirrors_text(engine: FakeEngine, mode: str) -> None:
    async with connect(engine, mode) as client:
        result = await client.call_tool("forget", {"memory_id": "mem_nope"})
        assert result.structured_content == {"deleted": False}
        result = await client.call_tool("pending_recollections", {})
        assert result.structured_content == {"result": []}
        assert payload(result) == []


@pytest.mark.parametrize(
    ("tool", "args", "needle"),
    [
        ("remember", {"content": "x", "geohash": "not-a-geohash!"}, "invalid geohash"),
        ("remember", {"content": "x", "lat": 48.8}, "lat and lon"),
        ("recall", {"query": "x", "lon": 2.0}, "lat and lon"),
        ("remember", {"content": "x", "at": "yesterday-ish"}, "ISO-8601"),
        ("remember", {"content": "x", "salience": 1.5}, "salience"),
        ("remember", {"content": ""}, "content"),
        ("sense", {"signals": [{"channel": "smell", "value": "rain"}]}, "unknown channel"),
        ("sense", {"signals": [{"channel": "Bad Name", "value": 1}]}, "channel"),
        ("recall", {"query": "x", "channels": ["smell"]}, "unknown channel"),
        ("sense", {"signals": [{"channel": "image", "data_b64": "%%%"}]}, "base64"),
        (
            "remember",
            {"content": "x", "signals": [{"channel": "image", "uri": "file:///etc/passwd"}]},
            "file",
        ),
        ("sense", {"signals": [{"channel": "audio", "uri": "/home/me/a.wav"}]}, "file"),
        ("sense", {"text": "a", "signals": [{"channel": "text", "value": "b"}]}, "more than one"),
        ("sense", {"signals": [{"channel": "text", "valu": "typo"}]}, "valu"),
        ("recall", {}, "at least one"),
        ("recall", {"query": "x", "k": 0}, "k"),
        ("pending_recollections", {"limit": 0}, "limit"),
        ("forget", {}, "memory_id"),
    ],
)
async def test_invalid_inputs_are_tool_errors(
    engine: FakeEngine, mode: str, tool: str, args: dict[str, object], needle: str
) -> None:
    async with connect(engine, mode) as client:
        message = await call_error(client, tool, args)
        assert needle.lower() in message.lower()
        # The server keeps working after the error.
        assert await call(client, "pending_recollections", {"drain": False}) == []


async def test_https_uri_is_allowed(engine: FakeEngine, mode: str) -> None:
    async with connect(engine, mode) as client:
        mem = await call(
            client,
            "remember",
            {"content": "x", "signals": [{"channel": "image", "uri": "https://example.org/a.png"}]},
        )
        assert "image" in mem["channels"]


async def test_file_uris_allowed_when_enabled(engine: FakeEngine) -> None:
    server = create_server(engine, allow_file_uris=True)
    async with Client(server) as c:
        mem = await call(
            c,
            "remember",
            {"content": "x", "signals": [{"channel": "image", "uri": "file:///tmp/a.png"}]},
        )
        assert "image" in mem["channels"]


async def test_unexpected_engine_errors_do_not_crash(engine: FakeEngine) -> None:
    def boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("disk on fire")

    engine.recall = boom  # type: ignore[method-assign]
    async with Client(create_server(engine)) as c:
        result = await c.call_tool("recall", {"query": "x"})
        assert result.is_error
        assert "Error executing tool recall" in text(result)  # details are only logged
        assert await call(c, "pending_recollections") == []


async def test_tool_defaults_use_the_engine_clock(engine: FakeEngine, clock: FakeClock) -> None:
    clock.advance(timedelta(days=3))
    async with Client(create_server(engine)) as c:
        mem = await call(c, "remember", {"content": "x", "geohash": EIFFEL})
        assert mem["context"]["timestamp"] == clock.now().isoformat()
        await call(c, "remember", {"content": "Eiffel tower"})
        fired = await call(c, "sense", {"text": "Eiffel tower"})
        assert fired[0]["triggered_at"] == clock.now().isoformat()


async def test_tool_defaults_read_engine_now_without_building_cues(
    engine: FakeEngine, clock: FakeClock
) -> None:
    """R29: "now" comes from ``engine.now()``, not from a throwaway cue."""
    built: list[str] = []
    real_build = engine.build_cue

    def spy(*args: Any, **kwargs: Any) -> Any:
        built.append(str(kwargs.get("source")))
        return real_build(*args, **kwargs)

    engine.build_cue = spy  # type: ignore[method-assign]
    clock.advance(timedelta(days=1))
    async with Client(create_server(engine)) as c:
        mem = await call(c, "remember", {"content": "x"})
        assert mem["context"]["timestamp"] == clock.now().isoformat()
        await call(c, "recall", {"query": "x"})
        await call(c, "sense", {"text": "x"})
    assert built == ["mcp", "mcp"]  # one cue per recall/sense, none just to read the clock
