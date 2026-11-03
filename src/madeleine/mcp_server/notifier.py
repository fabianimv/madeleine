"""``notifications/resources/updated`` for ``recollections://pending``.

Two sources of "the queue changed":

* **in-process** - :meth:`RecollectionQueue.subscribe` calls us on every ``put`` made in
  this process. The engine runs in worker threads, so the callback may arrive on any
  thread; it only wakes the notifier loop, hopping to the event loop thread safely.
* **other processes** sharing the SQLite file - discovered by polling
  ``queue.pending_count()`` every ``poll_interval`` seconds; growth triggers a notification.

Bursts are debounced: one notification per ``debounce`` window, however many recollections
fired. Delivery reaches both protocol eras the MCP SDK serves:

* 2026-07-28+ clients listen with ``subscriptions/listen``; we publish a
  :class:`ResourceUpdated` event on the server's :class:`SubscriptionBus`.
* earlier (handshake-era) clients send ``resources/subscribe``; we remember their sessions
  and send ``notifications/resources/updated`` on each connection's standalone channel.

The loop runs for as long as at least one server lifespan is active. When several
lifespans overlap (e.g. several legacy in-memory connections to the same server) only one
of them - the *leader* - runs the loop, so notifications are never duplicated; when the
leader's lifespan ends, the next one takes over.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import anyio
import anyio.from_thread
import anyio.lowlevel
import anyio.to_thread
from mcp.server.connection import Connection
from mcp.server.session import ServerSession
from mcp.server.subscriptions import ResourceUpdated, SubscriptionBus

from madeleine.core.types import Recollection, RecollectionQueue, Unsubscribe

__all__ = ["PENDING_URI", "PendingNotifier"]

logger = logging.getLogger(__name__)

PENDING_URI = "recollections://pending"
"""The resource listing queued involuntary recollections."""

_SEND_TIMEOUT = 5.0
"""Seconds a legacy notification send may take before the subscriber is dropped."""


@dataclass
class _LegacySubscriber:
    session: ServerSession
    uris: set[str] = field(default_factory=set)


class PendingNotifier:
    """Watches the recollection queue and notifies subscribed MCP sessions."""

    def __init__(
        self,
        queue: RecollectionQueue,
        bus: SubscriptionBus,
        *,
        poll_interval: float = 1.0,
        debounce: float = 0.05,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        if debounce < 0:
            raise ValueError("debounce must be >= 0")
        self._queue = queue
        self._bus = bus
        self.poll_interval = poll_interval
        self.debounce = debounce
        self._legacy: dict[object, _LegacySubscriber] = {}
        self._active = 0
        self._leader_lock: anyio.Lock | None = None
        self._wake: anyio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: int | None = None
        self._token: anyio.lowlevel.EventLoopToken | None = None
        self._state_lock = threading.Lock()
        self.running = False
        """True while a leader loop is watching the queue."""
        self.listener_attached = False
        """True when in-process queue events are received (False: poller only)."""
        self.notifications_sent = 0
        """Number of change notifications published (one per debounced burst)."""

    # -- legacy (resources/subscribe) sessions -------------------------------------------
    def add_legacy(self, session: ServerSession, uri: str) -> None:
        """Record a ``resources/subscribe`` from a handshake-era session."""
        key, connection = _connection_key(session)
        sub = self._legacy.get(key)
        if sub is None:
            sub = self._legacy[key] = _LegacySubscriber(session)
            if connection is not None:
                connection.exit_stack.callback(self._legacy.pop, key, None)
        sub.session = session
        sub.uris.add(uri)

    def remove_legacy(self, session: ServerSession, uri: str) -> None:
        """Handle ``resources/unsubscribe``."""
        key, _ = _connection_key(session)
        sub = self._legacy.get(key)
        if sub is not None:
            sub.uris.discard(uri)
            if not sub.uris:
                self._legacy.pop(key, None)

    @property
    def legacy_subscriber_count(self) -> int:
        return len(self._legacy)

    # -- lifecycle -----------------------------------------------------------------------
    @asynccontextmanager
    async def run(self) -> AsyncIterator[None]:
        """Watch the queue while the context is open (enter it in the server lifespan)."""
        if self._active == 0 or self._leader_lock is None:
            self._leader_lock = anyio.Lock()
        lock = self._leader_lock
        self._active += 1
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(self._lead, lock)
                try:
                    yield
                finally:
                    tg.cancel_scope.cancel()
        finally:
            self._active -= 1
            if self._active == 0:
                self._leader_lock = None

    async def _lead(self, lock: anyio.Lock) -> None:
        async with lock:
            unsubscribe = self._attach()
            try:
                # Read the baseline *before* declaring ourselves running, so a put that
                # lands during start-up counts as growth instead of hiding in the baseline.
                baseline = await self._count() or 0
                self.running = True
                await self._watch(baseline)
            finally:
                self.running = False
                self._detach(unsubscribe)

    def _attach(self) -> Unsubscribe | None:
        with self._state_lock:
            self._wake = anyio.Event()
            self._loop_thread = threading.get_ident()
            self._token = anyio.lowlevel.current_token()
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:  # not asyncio (e.g. trio)
                self._loop = None
        try:
            unsubscribe = self._queue.subscribe(self._on_put)
        except Exception as exc:
            logger.warning(
                "in-process recollection listener unavailable (%s); relying on polling every %.2fs",
                exc,
                self.poll_interval,
            )
            self.listener_attached = False
            return None
        self.listener_attached = True
        return unsubscribe

    def _detach(self, unsubscribe: Unsubscribe | None) -> None:
        if unsubscribe is not None:
            try:
                unsubscribe()
            except Exception:
                logger.exception("failed to unsubscribe from the recollection queue")
        with self._state_lock:
            self._wake = None
            self._loop = None
            self._loop_thread = None
            self._token = None
        self.listener_attached = False

    # -- wake-ups --------------------------------------------------------------------------
    def _set_wake(self) -> None:
        wake = self._wake
        if wake is not None:
            wake.set()

    def _on_put(self, _recollection: Recollection) -> None:
        """Queue listener: may run on any thread; never raises, never blocks for long."""
        with self._state_lock:
            loop, loop_thread, token = self._loop, self._loop_thread, self._token
        if loop_thread is None:
            return
        try:
            if threading.get_ident() == loop_thread:
                self._set_wake()
            elif loop is not None:
                loop.call_soon_threadsafe(self._set_wake)
            elif token is not None:
                anyio.from_thread.run_sync(self._set_wake, token=token)
        except Exception:  # loop closed while shutting down: nothing left to notify
            logger.debug("recollection wake-up dropped", exc_info=True)

    # -- the loop --------------------------------------------------------------------------
    async def _count(self) -> int | None:
        try:
            return await anyio.to_thread.run_sync(self._queue.pending_count)
        except Exception:
            logger.warning("pending_count() failed; will retry", exc_info=True)
            return None

    async def _watch(self, baseline: int) -> None:
        while True:
            woke = False
            wake = self._wake
            assert wake is not None
            with anyio.move_on_after(self.poll_interval):
                await wake.wait()
                woke = True
            if woke:
                # Coalesce a burst (one sense() may enqueue several recollections).
                await anyio.sleep(self.debounce)
                self._wake = anyio.Event()
            count = await self._count()
            if count is None:
                continue
            if woke or count > baseline:
                await self.publish()
            baseline = count

    async def publish(self) -> None:
        """Tell every subscribed session that ``recollections://pending`` changed."""
        self.notifications_sent += 1
        try:
            await self._bus.publish(ResourceUpdated(uri=PENDING_URI))
        except Exception:
            logger.exception("publishing to the subscription bus failed")
        for key, sub in list(self._legacy.items()):
            if PENDING_URI not in sub.uris:
                continue
            try:
                with anyio.fail_after(_SEND_TIMEOUT):
                    await sub.session.send_resource_updated(PENDING_URI)
            except Exception:
                logger.info("dropping a resource subscriber whose connection failed")
                self._legacy.pop(key, None)


def _connection_key(session: ServerSession) -> tuple[object, Connection | None]:
    """Identify the client connection behind a per-request ``ServerSession`` proxy.

    The SDK builds a new ``ServerSession`` for every request but they share one
    ``Connection``; it is not exposed publicly, hence the guarded lookup.
    """
    connection: Any = getattr(session, "_connection", None)
    if isinstance(connection, Connection):
        return connection, connection
    return session, None
