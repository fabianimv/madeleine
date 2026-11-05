"""Real transports: stdio subprocess and streamable HTTP over a socket."""

from __future__ import annotations

import socket
import sys
import threading
import warnings
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager

import anyio
import pytest
import uvicorn
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import ASGIApp

from madeleine.mcp_server import PENDING_URI, mcp_asgi_app
from madeleine.testing import FakeEngine
from tests.mcp_server.mcp_helpers import call
from tests.mcp_server.test_mcp_notifications import LegacyInbox

TOOLS = {
    "remember",
    "recall",
    "sense",
    "pending_recollections",
    "get_memory",
    "update_memory",
    "forget",
}

STDIO_SERVER = """
import logging
logging.basicConfig(level=logging.DEBUG)  # logs must still go to stderr only
logging.getLogger("madeleine").info("starting")
from madeleine.mcp_server import run_stdio
from madeleine.testing import FakeEngine
run_stdio(FakeEngine(), poll_interval=0.1)
"""


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_stdio_subprocess_speaks_clean_protocol(mode: str) -> None:
    params = StdioServerParameters(command=sys.executable, args=["-c", STDIO_SERVER])
    with anyio.fail_after(60):
        async with Client(params, mode=mode) as client:
            assert client.server_info is not None
            assert client.server_info.name == "madeleine"
            tools = await client.list_tools()
            assert {t.name for t in tools.tools} == TOOLS
            await call(client, "remember", {"content": "rain on a tin roof"})
            fired = await call(client, "sense", {"text": "rain on a tin roof"})
            assert len(fired) == 1
            assert len(await call(client, "pending_recollections")) == 1


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextmanager
def _serve(app: ASGIApp) -> Iterator[str]:
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, lifespan="on")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(500):
            if server.started:
                break
            if not thread.is_alive():
                raise RuntimeError("uvicorn failed to start")
            threading.Event().wait(0.01)
        else:
            raise RuntimeError("uvicorn did not start")
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def _host_app(engine: FakeEngine) -> Starlette:
    """A host application mounting MCP at /mcp, as the ingestion app does."""
    mcp_app, mcp_lifespan = mcp_asgi_app(engine, allow_file_uris=False, poll_interval=0.1)

    @asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with mcp_lifespan():
            yield

    return Starlette(routes=[Mount("/mcp", app=mcp_app)], lifespan=lifespan)


@pytest.mark.parametrize("mode", ["auto", "legacy"])
async def test_streamable_http_mounted_round_trip(mode: str) -> None:
    engine = FakeEngine()
    with _serve(_host_app(engine)) as base:
        with anyio.fail_after(30):
            async with Client(f"{base}/mcp/", mode=mode) as client:
                tools = await client.list_tools()
                assert {t.name for t in tools.tools} == TOOLS
                mem = await call(client, "remember", {"content": "jazz at the corner bar"})
                found = await call(client, "recall", {"query": "jazz bar"})
                assert found[0]["memory"]["id"] == mem["id"]
                assert len(await call(client, "sense", {"text": "jazz bar"})) == 1
                peek = await call(client, "pending_recollections", {"drain": False})
                assert len(peek) == 1
                assert await call(client, "forget", {"memory_id": mem["id"]}) == {"deleted": True}


async def test_streamable_http_listen_notification() -> None:
    engine = FakeEngine()
    with _serve(_host_app(engine)) as base:
        with anyio.fail_after(30):
            async with Client(f"{base}/mcp/") as client:
                await call(client, "remember", {"content": "fresh bread smell"})
                async with client.listen(resource_subscriptions=[PENDING_URI]) as sub:
                    await call(client, "sense", {"text": "fresh bread"})
                    event = await anext(aiter(sub))
                    assert event.uri == PENDING_URI  # type: ignore[union-attr]


async def test_streamable_http_legacy_subscription_notification() -> None:
    engine = FakeEngine()
    inbox = LegacyInbox()
    with _serve(_host_app(engine)) as base:
        with anyio.fail_after(30):
            async with Client(f"{base}/mcp/", mode="legacy", message_handler=inbox) as client:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    await client.subscribe_resource(PENDING_URI)
                await call(client, "remember", {"content": "fresh bread smell"})
                await call(client, "sense", {"text": "fresh bread"})
                await inbox.wait()
                assert inbox.uris == [PENDING_URI]
