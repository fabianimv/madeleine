"""The madeleine MCP server: tools, the pending-recollections resource, and stdio.

Tools (names are the public API):

* ``remember`` - encode a memory with its sensory context.
* ``recall`` - PULL: search memories by text, place and/or time.
* ``sense`` - PUSH: report what is being sensed now; returns the memories that fired.
* ``pending_recollections`` - read (and by default drain) queued involuntary recollections.
* ``forget`` - delete a memory everywhere.

Resource ``recollections://pending`` (JSON) is a non-draining peek of the queue; subscribed
clients get ``notifications/resources/updated`` when it grows (see :mod:`.notifier`).

All engine calls are blocking and run in worker threads. Every input is parsed with
:mod:`madeleine.core.serde`; madeleine errors become MCP tool errors (``isError``) and never
crash the server. Logging goes to stderr only.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any, TypeVar

import anyio
import anyio.to_thread
import mcp_types as types
from mcp.server.context import ServerRequestContext
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.subscriptions import InMemorySubscriptionBus
from mcp_types import CallToolResult, TextContent, ToolAnnotations
from pydantic import Field

from madeleine import __version__
from madeleine.core import serde
from madeleine.core.types import (
    TEXT,
    ChannelNotAvailableError,
    JSONValue,
    MadeleineError,
    MemoryEngine,
    ObservationError,
    RawSignal,
    RecallQuery,
)
from madeleine.mcp_server._inputs import SignalInput, check_unique_channels, parse_signals
from madeleine.mcp_server.notifier import PENDING_URI, PendingNotifier

__all__ = ["INSTRUCTIONS", "PENDING_URI", "MadeleineMCPServer", "create_server", "run_stdio"]

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

INSTRUCTIONS = """\
madeleine gives you involuntary, cue-triggered memory (the Proust effect).

Besides searching memory on purpose (`recall`), memories can come back on their own: when a
sensory cue arrives - a place, a time of day, a sound, an image, a heart rate, a webhook from
a phone or a smart home - madeleine compares it with the context each memory was encoded in
and, if they match strongly enough, an *involuntary recollection* fires and is queued for
you, even if you never asked. Cues may come from you (`sense`) or from other processes
(HTTP webhooks, hooks) sharing the same memory store.

How to use it:
- `remember` things worth keeping, with the context they happened in: where (geohash or
  lat/lon), when (`at`, ISO 8601 with the user's local UTC offset), tags, salience 0..1, and
  optional extra signals (image/audio/biometric). Rich context makes later cues work.
- `sense` what is being perceived right now (text description, location, biometrics...);
  it returns any recollections that fired. Weave them into your reply naturally, as a
  memory that "comes back", only when relevant.
- Call `pending_recollections` at the start of each turn (and whenever you are notified that
  `recollections://pending` changed) to receive recollections triggered by cues from other
  sources. By default this drains the queue, so each recollection is delivered once.
- `recall` searches memory deliberately; `forget` deletes a memory for good.
Scores are in [0, 1]; `channel_scores` explains which senses matched.
"""

# --------------------------------------------------------------------------------------
# Argument types (descriptions are what the model reads)
# --------------------------------------------------------------------------------------

Geohash = Annotated[
    str | None,
    Field(description="Where, as a geohash (any precision 1-12). Wins over lat/lon."),
]
Lat = Annotated[
    float | None,
    Field(description="Latitude in degrees (WGS84). Requires lon.", ge=-90, le=90),
]
Lon = Annotated[
    float | None,
    Field(description="Longitude in degrees (WGS84). Requires lat.", ge=-180, le=180),
]
Tags = Annotated[list[str] | None, Field(description="Free-form labels, e.g. ['family'].")]
Signals = Annotated[
    list[SignalInput] | None,
    Field(description="Extra raw readings, at most one per channel (see each field)."),
]


def _json_result(payload: JSONValue) -> CallToolResult:
    """One JSON text block (what the model reads) plus the same data as structured content."""
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    structured = payload if isinstance(payload, dict) else {"result": payload}
    return CallToolResult(
        content=[TextContent(type="text", text=text)], structured_content=structured
    )


async def _in_thread(fn: Callable[[], _T]) -> _T:
    """Run a blocking engine call off the event loop; madeleine errors -> tool errors."""
    try:
        return await anyio.to_thread.run_sync(fn)
    except (MadeleineError, ValueError) as exc:
        raise ToolError(str(exc)) from exc


def _engine_now(engine: MemoryEngine) -> datetime:
    """The engine's notion of "now".

    ``MemoryEngine`` does not expose its clock, but ``build_cue`` stamps an empty cue
    with it; this keeps tool defaults consistent with injected (fake) clocks.
    """
    cue = engine.build_cue((), include_time=False, include_geo=False, source="mcp")
    return cue.context.timestamp


class MadeleineMCPServer(MCPServer[dict[str, Any]]):
    """An :class:`MCPServer` bound to a :class:`MemoryEngine` (see :func:`create_server`)."""

    def __init__(
        self,
        engine: MemoryEngine,
        *,
        poll_interval: float = 1.0,
        allow_file_uris: bool = False,
        debounce: float = 0.05,
    ) -> None:
        self.engine = engine
        self.allow_file_uris = allow_file_uris
        bus = InMemorySubscriptionBus()
        self.notifier = PendingNotifier(
            engine.queue, bus, poll_interval=poll_interval, debounce=debounce
        )
        notifier = self.notifier

        @asynccontextmanager
        async def lifespan(_: MCPServer[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
            async with notifier.run():
                yield {}

        super().__init__(
            name="madeleine",
            title="madeleine - involuntary memory",
            instructions=INSTRUCTIONS,
            version=__version__,
            lifespan=lifespan,
            subscriptions=bus,
        )
        self._register_tools()
        self._register_resources()

    # -- tools ---------------------------------------------------------------------------
    def _signals(
        self, items: Sequence[SignalInput | Mapping[str, Any]] | None, now: datetime
    ) -> list[RawSignal]:
        signals = parse_signals(items, now=now, allow_file_uris=self.allow_file_uris)
        check_unique_channels(signals)
        return signals

    def _register_tools(self) -> None:
        engine = self.engine

        async def remember(
            content: Annotated[
                str,
                Field(
                    description=(
                        "What to remember, as you want it back later (usually one or two "
                        "sentences). Its text is also embedded as the 'text' channel."
                    ),
                    min_length=1,
                ),
            ],
            geohash: Geohash = None,
            lat: Lat = None,
            lon: Lon = None,
            at: Annotated[
                str | None,
                Field(
                    description=(
                        "When it happened, ISO 8601 with the user's local offset, e.g. "
                        "'2026-06-01T21:30:00+02:00'. Default: now."
                    )
                ),
            ] = None,
            tags: Tags = None,
            salience: Annotated[
                float | None,
                Field(
                    description=(
                        "How significant it is, 0 (trivial) to 1 (life-changing). "
                        "Default 0.5. Salient memories surface more easily."
                    ),
                    ge=0,
                    le=1,
                ),
            ] = None,
            signals: Signals = None,
            media_ref: Annotated[
                str | None,
                Field(description="Optional pointer (URI/path) to related media; never opened."),
            ] = None,
        ) -> CallToolResult:
            def run() -> JSONValue:
                now = _engine_now(engine)
                context = serde.context_from_fields(
                    now=now, at=at, geohash=geohash, lat=lat, lon=lon, tags=tags, salience=salience
                )
                raw = self._signals(signals, now)
                memory = engine.remember(content, raw, context=context, media_ref=media_ref)
                return memory.to_dict()

            return _json_result(await _in_thread(run))

        async def recall(
            query: Annotated[
                str | None, Field(description="Text to search for (semantic/lexical match).")
            ] = None,
            geohash: Geohash = None,
            lat: Lat = None,
            lon: Lon = None,
            at: Annotated[
                str | None,
                Field(
                    description=(
                        "Match memories from a similar time of day/week/year (ISO 8601, local "
                        "offset). Time is only used when given."
                    )
                ),
            ] = None,
            channels: Annotated[
                list[str] | None,
                Field(description="Only fuse these channels, e.g. ['geo']. Default: all shared."),
            ] = None,
            tags: Annotated[
                list[str] | None, Field(description="Only memories having all of these tags.")
            ] = None,
            k: Annotated[int, Field(description="Maximum number of results.", ge=1, le=100)] = 5,
        ) -> CallToolResult:
            def run() -> JSONValue:
                has_place = bool(geohash) or lat is not None or lon is not None
                if not query and not has_place and not at:
                    raise ObservationError(
                        "recall needs at least one of: query, geohash or lat/lon, at"
                    )
                now = _engine_now(engine)
                context = serde.context_from_fields(
                    now=now, at=at, geohash=geohash, lat=lat, lon=lon
                )
                raw = serde.signals_from_json(
                    [{"channel": TEXT, "value": query}] if query else [], now=now
                )
                only: frozenset[str] | None = None
                if channels:
                    known = engine.channels.names()
                    unknown = sorted(set(channels) - set(known))
                    if unknown:
                        raise ChannelNotAvailableError(
                            f"unknown channel(s) {unknown}; available: {list(known)}"
                        )
                    only = frozenset(channels)
                cue = engine.build_cue(
                    raw, context=context, include_time=at is not None, source="mcp"
                )
                results = engine.recall(
                    RecallQuery(cue=cue, k=k, channels=only, tags=frozenset(tags) if tags else None)
                )
                return [s.to_dict() for s in results]

            return _json_result(await _in_thread(run))

        async def sense(
            text: Annotated[
                str | None,
                Field(description="What is perceived right now, in words ('smell of rain')."),
            ] = None,
            geohash: Geohash = None,
            lat: Lat = None,
            lon: Lon = None,
            at: Annotated[
                str | None,
                Field(description="When it is sensed (ISO 8601, local offset). Default: now."),
            ] = None,
            biometric: Annotated[
                dict[str, float] | None,
                Field(
                    description=(
                        "Bodily state: heart_rate (bpm), hrv (ms), skin_temp (deg C), eda "
                        "(microsiemens), steps_per_min, respiration_rate (breaths/min)."
                    )
                ),
            ] = None,
            signals: Signals = None,
            source: Annotated[
                str, Field(description="Who is reporting the cue (free text).")
            ] = "mcp",
        ) -> CallToolResult:
            def run() -> JSONValue:
                now = _engine_now(engine)
                context = serde.context_from_fields(
                    now=now, at=at, geohash=geohash, lat=lat, lon=lon
                )
                items: list[SignalInput | Mapping[str, Any]] = []
                if text is not None:
                    items.append({"channel": TEXT, "value": text})
                if biometric is not None:
                    items.append({"channel": "biometric", "value": biometric})
                items.extend(signals or ())
                raw = self._signals(items, now)
                cue = engine.build_cue(raw, context=context, include_time=True, source=source)
                return [r.to_dict() for r in engine.sense(cue)]

            return _json_result(await _in_thread(run))

        async def pending_recollections(
            limit: Annotated[
                int | None, Field(description="Maximum number to return (oldest first).", ge=1)
            ] = None,
            drain: Annotated[
                bool,
                Field(
                    description=(
                        "True (default): remove them from the queue so each is delivered "
                        "once. False: just peek."
                    )
                ),
            ] = True,
        ) -> CallToolResult:
            def run() -> JSONValue:
                recs = engine.pending_recollections(limit=limit, drain=drain)
                return [r.to_dict() for r in recs]

            return _json_result(await _in_thread(run))

        async def forget(
            memory_id: Annotated[
                str, Field(description="The memory's id (e.g. 'mem_0189a3f2c1d4_9f3c2a7b1e').")
            ],
        ) -> CallToolResult:
            def run() -> JSONValue:
                return {"deleted": engine.forget(memory_id)}

            return _json_result(await _in_thread(run))

        self.add_tool(
            remember,
            description=(
                "Store a memory together with the sensory context it happened in (place, "
                "time, tags, salience, optional image/audio/biometric signals), so that "
                "similar cues can bring it back later. Returns the stored memory as JSON."
            ),
            annotations=ToolAnnotations(title="Remember", read_only_hint=False),
            structured_output=False,
        )
        self.add_tool(
            recall,
            description=(
                "Deliberately search memories (PULL). Give at least one of: query (text), "
                "a location (geohash or lat+lon), at (time). Only the cues you give are "
                "matched. Returns up to k memories, best first, each with a score in [0, 1] "
                "and per-channel similarities explaining the match. No side effects."
            ),
            annotations=ToolAnnotations(
                title="Recall", read_only_hint=True, idempotent_hint=True, open_world_hint=False
            ),
            structured_output=False,
        )
        self.add_tool(
            sense,
            description=(
                "Report what is being sensed right now (PUSH): text, location, biometrics or "
                "raw signals. The current time is always part of the cue. Memories whose "
                "encoding context matches strongly fire as involuntary recollections: they "
                "are returned here and also queued (see pending_recollections). A memory "
                "that just fired will not fire again for a while (habituation)."
            ),
            annotations=ToolAnnotations(title="Sense", read_only_hint=False, idempotent_hint=False),
            structured_output=False,
        )
        self.add_tool(
            pending_recollections,
            description=(
                "Get involuntary recollections that fired since you last checked, including "
                "those triggered by other sources (webhooks, sensors, other processes). Call "
                "it at the start of each turn. Oldest first. drain=true (default) removes "
                "them so each is delivered once; drain=false peeks."
            ),
            annotations=ToolAnnotations(title="Pending recollections", read_only_hint=False),
            structured_output=False,
        )
        self.add_tool(
            forget,
            description=(
                "Permanently delete a memory (and any queued recollection of it). "
                'Returns {"deleted": true} if it existed, false otherwise.'
            ),
            annotations=ToolAnnotations(
                title="Forget", read_only_hint=False, destructive_hint=True, idempotent_hint=True
            ),
            structured_output=False,
        )

    # -- resources -----------------------------------------------------------------------
    def _register_resources(self) -> None:
        engine = self.engine
        notifier = self.notifier

        async def pending() -> str:
            recs = await anyio.to_thread.run_sync(lambda: engine.pending_recollections(drain=False))
            return json.dumps([r.to_dict() for r in recs], ensure_ascii=False)

        self.resource(
            PENDING_URI,
            name="pending_recollections",
            title="Pending involuntary recollections",
            description=(
                "Involuntary recollections waiting to be delivered, oldest first (JSON list). "
                "Reading does not remove them; use the pending_recollections tool to drain. "
                "Subscribe to be notified when new ones fire."
            ),
            mime_type="application/json",
        )(pending)

        # Handshake-era (<= 2025-11-25) clients subscribe with resources/subscribe; the
        # 2026-07-28 subscriptions/listen path is served by MCPServer itself via the bus.
        # MCPServer has no public hook for these two methods yet, so register them on the
        # low-level server it wraps.
        async def on_subscribe(
            ctx: ServerRequestContext[Any, Any], params: types.SubscribeRequestParams
        ) -> types.EmptyResult:
            notifier.add_legacy(ctx.session, str(params.uri))
            return types.EmptyResult()

        async def on_unsubscribe(
            ctx: ServerRequestContext[Any, Any], params: types.UnsubscribeRequestParams
        ) -> types.EmptyResult:
            notifier.remove_legacy(ctx.session, str(params.uri))
            return types.EmptyResult()

        lowlevel = self._lowlevel_server
        lowlevel.add_request_handler(
            "resources/subscribe", types.SubscribeRequestParams, on_subscribe
        )
        lowlevel.add_request_handler(
            "resources/unsubscribe", types.UnsubscribeRequestParams, on_unsubscribe
        )


def create_server(
    engine: MemoryEngine,
    *,
    poll_interval: float = 1.0,
    allow_file_uris: bool = False,
    debounce: float = 0.05,
) -> MadeleineMCPServer:
    """Build the madeleine MCP server for ``engine``.

    * ``poll_interval`` - seconds between ``pending_count()`` polls that detect
      recollections enqueued by *other processes* sharing the database.
    * ``allow_file_uris`` - accept signals whose ``uri`` is a local path / ``file://``.
      Off by default: a remote client must not make the server read local files.
    * ``debounce`` - seconds to coalesce a burst of recollections into one notification.

    The server does not own ``engine``: closing it is the caller's job.
    """
    return MadeleineMCPServer(
        engine, poll_interval=poll_interval, allow_file_uris=allow_file_uris, debounce=debounce
    )


def run_stdio(engine: MemoryEngine, **kwargs: Any) -> None:
    """Serve MCP over stdin/stdout until the client disconnects (blocking).

    ``kwargs`` go to :func:`create_server`. Only protocol frames are written to stdout;
    logs go to stderr.
    """
    server = create_server(engine, **kwargs)
    logger.info("madeleine MCP server on stdio")
    server.run("stdio")
