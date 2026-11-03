"""The FastAPI ingestion app: webhooks in, involuntary recollections out.

Routes (see ``ARCHITECTURE.md`` "HTTP surface"):

* ``POST /cues`` - generic cue (``{signals, text?, biometric?, at?, geohash?, lat?, lon?,
  tags?, salience?, source?}``) -> ``engine.sense`` -> ``{cue_id, recollections}``.
* ``POST /cues/{adapter}`` - adapter-specific payload (``generic``, ``home_assistant``,
  ``ios_shortcuts`` or any adapter passed to :func:`create_app`).
* ``POST /memories`` - same fields as the MCP ``remember`` tool -> ``Memory.to_dict()``.
* ``POST /recall`` - PULL; same fields as the MCP ``recall`` tool -> ``{results}``.
* ``GET /recollections/pending?drain=true&limit=`` -> ``{recollections}``.
* ``DELETE /memories/{memory_id}`` -> ``{deleted: true}`` (404 when unknown).
* ``GET /healthz`` -> ``{status: "ok", version}`` (never requires the token).

Every serialised recollection is ``Recollection.to_dict()`` plus ``prompt_line``
(``Recollection.to_prompt_line()``), which remote hooks inject verbatim.

Errors: malformed input (``ObservationError``/``ValueError``) -> 422, an unavailable channel
-> 400, unknown adapter or memory -> 404, a storage refusal (``StoreError``, e.g. "the text
model changed") -> 409, any other ``MadeleineError`` -> 500 (all with the error message as
``detail``), missing/invalid bearer token -> 401, body larger than ``max_body_bytes`` -> 413.
Request bodies and media are never logged.
"""

from __future__ import annotations

import hmac
import logging
import os
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from datetime import datetime
from typing import Annotated, Any

from fastapi import Body, FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from madeleine import __version__
from madeleine.core import serde
from madeleine.core.types import (
    TEXT,
    ChannelNotAvailableError,
    Clock,
    CueAdapter,
    JSONValue,
    MadeleineError,
    MemoryEngine,
    MemoryNotFoundError,
    ObservationError,
    RawSignal,
    RecallQuery,
    Recollection,
    SensoryContext,
    StoreError,
    SystemClock,
)
from madeleine.ingest import _fields as f
from madeleine.ingest import default_adapters
from madeleine.ingest.webhook import parse_generic_payload

__all__ = ["DEFAULT_MAX_BODY_BYTES", "create_app"]

_log = logging.getLogger("madeleine.ingest")

DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024
"""Default request body limit (10 MiB) - bounds base64 photos / audio clips."""

_PUBLIC_PATHS = frozenset({"/healthz"})

LifespanFactory = Callable[[], AbstractAsyncContextManager[None]]


# --------------------------------------------------------------------------------------
# Request models (field names shared with the MCP tools; parsing via madeleine.core.serde)
# --------------------------------------------------------------------------------------


class SignalIn(BaseModel):
    """One raw signal: ``{channel, value?, data_b64?, mime_type?, uri?, at?}``."""

    model_config = ConfigDict(extra="forbid")

    channel: str
    value: Any = None
    data_b64: str | None = None
    mime_type: str | None = None
    uri: str | None = None
    at: str | None = None


class _ContextFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    at: str | None = Field(None, description="ISO 8601 timestamp with offset; default now")
    geohash: str | None = None
    lat: float | None = None
    lon: float | None = None


class CueIn(_ContextFields):
    """Body of ``POST /cues`` (the generic adapter's payload)."""

    signals: list[SignalIn] = Field(default_factory=list)
    text: str | None = None
    biometric: dict[str, float] | None = None
    tags: list[str] | None = None
    salience: float | None = None
    source: str = Field("http", max_length=64)


class MemoryIn(_ContextFields):
    """Body of ``POST /memories`` (the MCP ``remember`` tool's fields)."""

    content: str = Field(min_length=1)
    signals: list[SignalIn] = Field(default_factory=list)
    tags: list[str] | None = None
    salience: float | None = None
    media_ref: str | None = None
    metadata: dict[str, str] | None = None
    embed_content: bool = True
    store_raw: bool | None = None


class RecallIn(_ContextFields):
    """Body of ``POST /recall`` (the MCP ``recall`` tool's fields).

    ``time`` takes part in the match only when ``at`` is given. ``query`` is used literally
    (``""`` means no query). Unknown ``channels`` are rejected with 422.
    """

    query: str | None = None
    signals: list[SignalIn] = Field(default_factory=list)
    channels: list[str] | None = None
    tags: list[str] | None = None
    k: int = Field(5, ge=1, le=100)
    min_score: float | None = Field(None, ge=0.0, le=1.0)


# --------------------------------------------------------------------------------------
# ASGI guard: bearer token + body size limit (also covers mounted sub-apps such as /mcp)
# --------------------------------------------------------------------------------------


class _BodyTooLargeError(HTTPException):
    def __init__(self, limit: int) -> None:
        super().__init__(status_code=413, detail=f"request body exceeds {limit} bytes")


class _GuardMiddleware:
    """Pure ASGI middleware (streaming-safe, unlike ``BaseHTTPMiddleware``)."""

    def __init__(self, app: ASGIApp, *, token: str | None, max_body_bytes: int) -> None:
        self.app = app
        self.token = token.encode() if token else None
        self.max_body_bytes = max_body_bytes

    def _authorized(self, scope: Scope) -> bool:
        if self.token is None:
            return True
        path: str = scope.get("path", "")
        root: str = scope.get("root_path", "")
        if root and path.startswith(root):
            path = path[len(root) :]
        if path in _PUBLIC_PATHS:
            return True
        for key, value in scope.get("headers", ()):
            if key == b"authorization":
                scheme, _, credentials = value.partition(b" ")
                if scheme.lower() == b"bearer":
                    return hmac.compare_digest(credentials.strip(), self.token)
        return False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        if not self._authorized(scope):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1008})
                return
            response = JSONResponse(
                {"detail": "missing or invalid bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        limit = self.max_body_bytes
        for key, value in scope.get("headers", ()):
            if key == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = 0
                if declared > limit:
                    too_large = {"detail": f"request body exceeds {limit} bytes"}
                    await JSONResponse(too_large, 413)(scope, receive, send)
                    return
        received = 0
        started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise _BodyTooLargeError(limit)
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLargeError as exc:  # raised in a sub-app that did not handle it
            if started:
                raise
            await JSONResponse({"detail": exc.detail}, 413)(scope, receive, send)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _recollection_json(r: Recollection) -> dict[str, JSONValue]:
    out = r.to_dict()
    out["prompt_line"] = r.to_prompt_line()
    return out


def _signals(items: Sequence[SignalIn], now: datetime) -> list[RawSignal]:
    return serde.signals_from_json([s.model_dump(exclude_none=True) for s in items], now=now)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer number of bytes") from None
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


# --------------------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------------------


def create_app(
    engine: MemoryEngine,
    *,
    mounts: Mapping[str, ASGIApp] | None = None,
    lifespans: Sequence[LifespanFactory] = (),
    adapters: Sequence[CueAdapter] | None = None,
    api_token: str | None = None,
    max_body_bytes: int | None = None,
    clock: Clock | None = None,
) -> FastAPI:
    """Build the ingestion app around ``engine`` (which the caller owns and closes).

    * ``mounts`` - extra ASGI apps mounted by path prefix (e.g. ``{"/mcp": mcp_app}``);
      they sit behind the same bearer-token and body-size guard.
    * ``lifespans`` - async context-manager factories entered (in order) when the app
      starts and exited when it stops (mounted apps' own lifespans are not run by Starlette,
      so pass them here).
    * ``adapters`` - cue adapters for ``POST /cues/{name}``; default
      :func:`madeleine.ingest.default_adapters`. Names must be unique.
    * ``api_token`` - bearer token required on every route except ``/healthz``.
      ``None`` reads ``MADELEINE_API_TOKEN``; ``""`` disables auth explicitly.
    * ``max_body_bytes`` - request body limit (413 above it). ``None`` reads
      ``MADELEINE_MAX_BODY_BYTES``, default :data:`DEFAULT_MAX_BODY_BYTES`.
    * ``clock`` - "now" for context fields given without ``at`` (default system clock;
      cues with no context fields use the engine's own clock).
    """
    clk: Clock = clock or SystemClock()
    raw_token = os.environ.get("MADELEINE_API_TOKEN", "") if api_token is None else api_token
    token = raw_token.strip() or None
    limit = (
        _env_int("MADELEINE_MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES)
        if max_body_bytes is None
        else max_body_bytes
    )
    if limit < 1:
        raise ValueError("max_body_bytes must be >= 1")
    by_name: dict[str, CueAdapter] = {}
    for adapter in default_adapters(clock=clk) if adapters is None else adapters:
        if adapter.name in by_name:
            raise ValueError(f"duplicate cue adapter name {adapter.name!r}")
        by_name[adapter.name] = adapter
    factories = tuple(lifespans)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            for factory in factories:
                await stack.enter_async_context(factory())
            yield

    app = FastAPI(
        title="madeleine",
        version=__version__,
        description="Involuntary, cue-triggered memory for AI agents: HTTP ingestion.",
        lifespan=lifespan,
    )
    app.add_middleware(_GuardMiddleware, token=token, max_body_bytes=limit)

    async def _unprocessable(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    async def _unavailable(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    async def _not_found(_: Request, exc: Exception) -> JSONResponse:
        detail = exc.args[0] if exc.args else "not found"
        return JSONResponse({"detail": f"memory not found: {detail}"}, status_code=404)

    async def _conflict(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    async def _internal(_: Request, exc: Exception) -> JSONResponse:
        _log.warning("request failed: %s", type(exc).__name__)
        return JSONResponse({"detail": str(exc) or type(exc).__name__}, status_code=500)

    # Starlette picks the handler of the first class in the exception's MRO, so the
    # specific MadeleineError subclasses must be registered explicitly (as they are here).
    app.add_exception_handler(ObservationError, _unprocessable)
    app.add_exception_handler(ValueError, _unprocessable)
    app.add_exception_handler(ChannelNotAvailableError, _unavailable)
    app.add_exception_handler(MemoryNotFoundError, _not_found)
    app.add_exception_handler(StoreError, _conflict)
    app.add_exception_handler(MadeleineError, _internal)

    def _sense(
        signals: list[RawSignal], context: SensoryContext | None, source: str
    ) -> dict[str, JSONValue]:
        cue = engine.build_cue(signals, context=context, source=source)
        fired = engine.sense(cue)
        _log.debug("cue %s from %s: %d recollection(s)", cue.id, source, len(fired))
        return {"cue_id": cue.id, "recollections": [_recollection_json(r) for r in fired]}

    @app.get("/healthz", tags=["meta"])
    def healthz() -> dict[str, str]:
        """Liveness probe (no token required)."""
        return {"status": "ok", "version": __version__}

    @app.post("/cues", tags=["cues"])
    def post_cue(body: CueIn) -> dict[str, Any]:
        """Sense a generic cue; returns the involuntary recollections it fired."""
        payload = body.model_dump(exclude_none=True, exclude={"source"})
        signals, context = parse_generic_payload(payload, now=clk.now())
        return _sense(signals, context, body.source)

    @app.post("/cues/{adapter}", tags=["cues"])
    def post_adapter_cue(
        adapter: str, payload: Annotated[dict[str, Any], Body()]
    ) -> dict[str, Any]:
        """Sense a cue from an adapter-specific payload (``POST /cues/home_assistant``)."""
        found = by_name.get(adapter)
        if found is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown cue adapter {adapter!r}; available: {sorted(by_name)}",
            )
        signals, context = found.to_signals(payload)
        return _sense(signals, context, f"webhook:{found.name}")

    @app.post("/memories", status_code=201, tags=["memories"])
    def post_memory(body: MemoryIn) -> dict[str, Any]:
        """Encode a memory with its sensory fingerprint (fields of the ``remember`` tool)."""
        now = clk.now()
        context = f.build_context(
            now=now,
            at=body.at,
            geohash=body.geohash,
            lat=body.lat,
            lon=body.lon,
            tags=body.tags or (),
            salience=body.salience,
        )
        memory = engine.remember(
            body.content,
            _signals(body.signals, now),
            context=context,
            media_ref=body.media_ref,
            metadata=body.metadata,
            embed_content=body.embed_content,
            store_raw=body.store_raw,
        )
        return memory.to_dict()

    @app.post("/recall", tags=["memories"])
    def post_recall(body: RecallIn) -> dict[str, Any]:
        """PULL recall (fields of the ``recall`` tool); no side effects."""
        now = clk.now()
        signals = _signals(body.signals, now)
        query = body.query or None  # taken literally ("" means "no query", as in MCP)
        if query is not None:
            if any(s.channel == TEXT for s in signals):
                raise ObservationError("text given both as 'query' and inside 'signals'")
            signals.append(RawSignal(TEXT, query))
        context = f.build_context(
            now=now, at=body.at, geohash=body.geohash, lat=body.lat, lon=body.lon
        )
        if not signals and context is None:
            raise ObservationError("recall needs a query, signals, a location or 'at'")
        only: frozenset[str] | None = None
        if body.channels:
            known = engine.channels.names()
            unknown = sorted(set(body.channels) - set(known))
            if unknown:
                raise ObservationError(f"unknown channel(s) {unknown}; available: {list(known)}")
            only = frozenset(body.channels)
        cue = engine.build_cue(
            signals, context=context, include_time=body.at is not None, source="http:recall"
        )
        results = engine.recall(
            RecallQuery(
                cue=cue,
                k=body.k,
                channels=only,
                tags=frozenset(body.tags) if body.tags else None,
                min_score=body.min_score,
            )
        )
        out: list[JSONValue] = [s.to_dict() for s in results]
        return {"results": out}

    @app.get("/recollections/pending", tags=["recollections"])
    def get_pending(
        drain: bool = True, limit: Annotated[int | None, Query(ge=1, le=1000)] = None
    ) -> dict[str, Any]:
        """Queued involuntary recollections, oldest first (``drain=true`` removes them)."""
        items = engine.pending_recollections(limit=limit, drain=drain)
        return {"recollections": [_recollection_json(r) for r in items]}

    @app.delete("/memories/{memory_id}", tags=["memories"])
    def delete_memory(memory_id: str) -> dict[str, Any]:
        """Forget a memory everywhere (store, vectors, raw data, pending recollections)."""
        if not engine.forget(memory_id):
            raise HTTPException(status_code=404, detail=f"memory not found: {memory_id}")
        return {"deleted": True}

    for path, sub_app in (mounts or {}).items():
        app.mount(path, sub_app)
    return app
