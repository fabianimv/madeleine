"""``notifications/resources/updated`` for ``recollections://pending``."""

from __future__ import annotations

import asyncio
import threading
import warnings
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
import pytest
from mcp import Client
from mcp.client.session import IncomingMessage
from mcp_types import ResourceUpdatedNotification

from madeleine.core.types import RecallMode, Recollection, ScoredMemory, new_id
from madeleine.mcp_server import PENDING_URI, MadeleineMCPServer, create_server
from madeleine.testing import FakeEngine
from tests.mcp_server.mcp_helpers import call

TIMEOUT = 5.0


def _recollection(engine: FakeEngine, content: str = "a memory") -> Recollection:
    memory = engine.remember(content)
    return Recollection(
        id=new_id("rec"),
        match=ScoredMemory(memory=memory, score=0.9, channel_scores={"text": 0.9}),
        cue_id=new_id("cue"),
        cue_channels=frozenset({"text"}),
        triggered_at=memory.created_at,
        mode=RecallMode.PUSH,
        cue_source="other-process",
    )


def _put_from_thread(engine: FakeEngine, recs: list[Recollection]) -> None:
    """Enqueue from a plain (non-anyio) thread, as another component would."""

    def worker() -> None:
        for rec in recs:
            engine.queue.put(rec)

    t = threading.Thread(target=worker)
    t.start()
    t.join()


class LegacyInbox:
    """message_handler for handshake-era clients: records resources/updated URIs."""

    def __init__(self) -> None:
        self.uris: list[str] = []
        self._event = anyio.Event()

    async def __call__(self, message: IncomingMessage) -> None:
        if isinstance(message, ResourceUpdatedNotification):
            self.uris.append(message.params.uri)
            self._event.set()

    async def wait(self) -> None:
        with anyio.fail_after(TIMEOUT):
            await self._event.wait()
        self._event = anyio.Event()


@asynccontextmanager
async def legacy_client(server: MadeleineMCPServer, inbox: LegacyInbox) -> AsyncIterator[Client]:
    async with Client(server, mode="legacy", message_handler=inbox) as client:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # resources/subscribe is deprecated at 2026-07-28
            await client.subscribe_resource(PENDING_URI)
        yield client


async def _wait_until(predicate: object, timeout: float = TIMEOUT) -> None:
    assert callable(predicate)
    with anyio.fail_after(timeout):
        while not predicate():
            await anyio.sleep(0.01)


async def test_listen_receives_update_when_sense_fires() -> None:
    engine = FakeEngine()
    server = create_server(engine, poll_interval=30)
    async with Client(server) as client:
        await call(client, "remember", {"content": "the smell of madeleines"})
        async with client.listen(resource_subscriptions=[PENDING_URI]) as sub:
            assert sub.honored.resource_subscriptions == [PENDING_URI]
            fired = await call(client, "sense", {"text": "madeleines"})
            assert len(fired) == 1
            with anyio.fail_after(TIMEOUT):
                event = await anext(aiter(sub))
            assert event.uri == PENDING_URI  # type: ignore[union-attr]
    assert server.notifier.notifications_sent == 1


async def test_legacy_subscribe_receives_update_when_sense_fires() -> None:
    engine = FakeEngine()
    server = create_server(engine, poll_interval=30)
    inbox = LegacyInbox()
    async with legacy_client(server, inbox) as client:
        assert server.notifier.legacy_subscriber_count == 1
        await call(client, "remember", {"content": "the smell of madeleines"})
        await call(client, "sense", {"text": "madeleines"})
        await inbox.wait()
        assert inbox.uris == [PENDING_URI]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            await client.unsubscribe_resource(PENDING_URI)
        assert server.notifier.legacy_subscriber_count == 0
        sent = server.notifier.notifications_sent
        _put_from_thread(engine, [_recollection(engine)])
        await _wait_until(lambda: server.notifier.notifications_sent > sent)
        await anyio.sleep(0.05)
        assert inbox.uris == [PENDING_URI]  # unsubscribed: nothing more delivered
    assert server.notifier.legacy_subscriber_count == 0


async def test_put_from_another_thread_uses_in_process_listener() -> None:
    engine = FakeEngine()
    server = create_server(engine, poll_interval=30)  # the poller would be far too slow
    inbox = LegacyInbox()
    async with legacy_client(server, inbox):
        assert server.notifier.listener_attached
        _put_from_thread(engine, [_recollection(engine)])
        await inbox.wait()
        assert inbox.uris == [PENDING_URI]


async def test_poller_detects_puts_when_listener_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulates another process sharing the DB: no in-process listener fires."""
    engine = FakeEngine()

    def no_listener(_listener: object) -> object:
        raise NotImplementedError("cross-process queue")

    monkeypatch.setattr(engine.queue, "subscribe", no_listener)
    server = create_server(engine, poll_interval=0.05)
    async with Client(server) as client:
        async with client.listen(resource_subscriptions=[PENDING_URI]) as sub:
            await _wait_until(lambda: server.notifier.running)
            assert not server.notifier.listener_attached
            _put_from_thread(engine, [_recollection(engine)])
            with anyio.fail_after(TIMEOUT):
                event = await anext(aiter(sub))
            assert event.uri == PENDING_URI  # type: ignore[union-attr]
        # A drain (count goes down) is not a new recollection: no notification.
        sent = server.notifier.notifications_sent
        assert len(await call(client, "pending_recollections")) == 1
        await anyio.sleep(0.2)
        assert server.notifier.notifications_sent == sent


async def test_bursts_are_debounced() -> None:
    engine = FakeEngine()
    server = create_server(engine, poll_interval=30, debounce=0.2)
    inbox = LegacyInbox()
    async with legacy_client(server, inbox):
        _put_from_thread(engine, [_recollection(engine, f"m{i}") for i in range(20)])
        await inbox.wait()
        await anyio.sleep(0.3)
        assert inbox.uris == [PENDING_URI]
        assert server.notifier.notifications_sent == 1


async def test_overlapping_connections_share_one_watcher() -> None:
    engine = FakeEngine()
    server = create_server(engine, poll_interval=30)
    first, second = LegacyInbox(), LegacyInbox()
    async with legacy_client(server, second):
        async with legacy_client(server, first):
            assert server.notifier.legacy_subscriber_count == 2
            _put_from_thread(engine, [_recollection(engine)])
            await first.wait()
            await second.wait()
            await anyio.sleep(0.1)
            assert server.notifier.notifications_sent == 1
        # The first connection's lifespan ended: the second one takes over watching.
        await _wait_until(lambda: server.notifier.running)
        assert server.notifier.legacy_subscriber_count == 1
        _put_from_thread(engine, [_recollection(engine)])
        await second.wait()
        assert second.uris == [PENDING_URI, PENDING_URI]
        assert first.uris == [PENDING_URI]


async def test_lifecycle_stops_watcher_without_leaking_tasks() -> None:
    engine = FakeEngine()
    server = create_server(engine, poll_interval=0.01)
    before = set(asyncio.all_tasks())
    assert not server.notifier.running
    async with Client(server) as client:
        await _wait_until(lambda: server.notifier.running)
        assert engine.queue._listeners  # subscribed while running
        await call(client, "pending_recollections")
    assert not server.notifier.running
    assert not engine.queue._listeners
    await anyio.sleep(0.05)
    assert set(asyncio.all_tasks()) - before == set()
    # A put after shutdown is harmless.
    engine.queue.put(_recollection(engine))


def test_invalid_notifier_settings() -> None:
    with pytest.raises(ValueError, match="poll_interval"):
        create_server(FakeEngine(), poll_interval=0)
    with pytest.raises(ValueError, match="debounce"):
        create_server(FakeEngine(), debounce=-1)
