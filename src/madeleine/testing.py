"""Test helpers shared by madeleine's own tests and by downstream users."""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from madeleine.core import geohash as _gh
from madeleine.core.types import (
    GEO,
    TEXT,
    TIME,
    Channel,
    ChannelName,
    ChannelNotAvailableError,
    Clock,
    Cue,
    MadeleineConfig,
    Memory,
    MemoryNotFoundError,
    MemoryStore,
    Observation,
    ObservationError,
    RawSignal,
    RecallMode,
    RecallQuery,
    Recollection,
    RecollectionListener,
    ScoredMemory,
    SensoryContext,
    SensoryFingerprint,
    Unsubscribe,
    _require_aware,
    new_id,
)


class FakeClock:
    """A controllable :class:`~madeleine.core.types.Clock`."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
        _require_aware(self._now, "FakeClock start")

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> datetime:
        self._now += delta
        return self._now

    def set(self, when: datetime) -> None:
        _require_aware(when, "FakeClock.set")
        self._now = when


# --------------------------------------------------------------------------------------
# FakeEngine: a tiny, dependency-free MemoryEngine for testing *consumers* of the engine
# (MCP server, HTTP ingestion, hooks, evals plumbing). It is NOT the real recall engine:
# similarity is crude (token Jaccard for text, shared geohash prefix for geo, hour
# distance for time) and there are no vectors. Use ``madeleine.engine.create_engine`` with
# the hash registry for realistic behaviour.
# --------------------------------------------------------------------------------------

_log = logging.getLogger(__name__)

_FAKE_CHANNELS: tuple[ChannelName, ...] = ("audio", "biometric", "geo", "image", "text", "time")


class FakeRegistry:
    """Knows channel *names* only; ``get`` always raises (there are no real channels)."""

    def __init__(self, names: Sequence[ChannelName] = _FAKE_CHANNELS) -> None:
        self._names = tuple(sorted(names))

    def register(self, channel: Channel, *, replace: bool = False) -> None:
        raise NotImplementedError("FakeRegistry is read-only")

    def get(self, name: ChannelName) -> Channel:
        raise ChannelNotAvailableError(f"FakeRegistry has no channel objects ({name!r})")

    def names(self) -> tuple[ChannelName, ...]:
        return self._names

    def __contains__(self, name: object) -> bool:
        return name in self._names


class FakeQueue:
    """In-memory :class:`~madeleine.core.types.RecollectionQueue`."""

    def __init__(self) -> None:
        self._items: list[Recollection] = []
        self._listeners: list[RecollectionListener] = []
        self._lock = threading.Lock()

    def put(self, recollection: Recollection) -> None:
        with self._lock:
            self._items.append(recollection)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(recollection)
            except Exception:
                _log.exception("recollection listener failed")

    def peek(self, limit: int | None = None) -> list[Recollection]:
        with self._lock:
            return list(self._items[:limit])

    def drain(self, limit: int | None = None) -> list[Recollection]:
        with self._lock:
            n = len(self._items) if limit is None else min(limit, len(self._items))
            out, self._items = self._items[:n], self._items[n:]
            return out

    def pending_count(self) -> int:
        with self._lock:
            return len(self._items)

    def discard_memory(self, memory_id: str) -> int:
        with self._lock:
            before = len(self._items)
            self._items = [r for r in self._items if r.memory.id != memory_id]
            return before - len(self._items)

    def subscribe(self, listener: RecollectionListener) -> Unsubscribe:
        with self._lock:
            self._listeners.append(listener)

        def _unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return _unsubscribe


def _fake_normalize(signal: RawSignal, now: datetime) -> Observation:
    ch, v = signal.channel, signal.value
    if ch == TEXT:
        if not isinstance(v, str) or not v.strip():
            raise ObservationError("text signal needs a non-empty string")
        return Observation(TEXT, text=v.strip())
    if ch == GEO:
        if isinstance(v, str) and _gh.is_valid(v):
            return Observation(GEO, text=v.lower())
        if isinstance(v, tuple) and len(v) == 2:
            lat, lon = float(v[0]), float(v[1])
            return Observation(GEO, text=_gh.encode(lat, lon), numbers={"lat": lat, "lon": lon})
        raise ObservationError("geo signal needs a geohash or (lat, lon)")
    if ch == TIME:
        if isinstance(v, datetime):
            return Observation(TIME, timestamp=v if v.tzinfo else v.replace(tzinfo=UTC))
        return Observation(TIME, timestamp=signal.observed_at or now)
    if ch == "biometric":
        if not isinstance(v, Mapping):
            raise ObservationError("biometric signal needs a mapping of numbers")
        return Observation(ch, numbers={str(k): float(x) for k, x in v.items()})
    if ch in ("image", "audio"):
        data = v if isinstance(v, bytes) else None
        if data is None and signal.uri is None:
            raise ObservationError(f"{ch} signal needs bytes or a uri")
        return Observation(ch, data=data, uri=signal.uri, mime_type=signal.mime_type)
    raise ChannelNotAvailableError(f"unknown channel {ch!r}")


def _tokens(text: str | None) -> set[str]:
    return set("".join(c.lower() if c.isalnum() else " " for c in text or "").split())


def _fake_similarity(a: Observation, b: Observation) -> float:
    if a.channel == TEXT:
        ta, tb = _tokens(a.text), _tokens(b.text)
        return len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    if a.channel == GEO:
        ga, gb = a.text or "", b.text or ""
        common = 0
        for x, y in zip(ga, gb, strict=False):
            if x != y:
                break
            common += 1
        return common / max(1, min(len(ga), len(gb)))
    if a.channel == TIME and a.timestamp and b.timestamp:
        diff = abs(a.timestamp.hour + a.timestamp.minute / 60 - b.timestamp.hour)
        diff = min(diff, 24 - diff)
        return max(0.0, 1.0 - diff / 12)
    return 1.0 if a == b else 0.0


class FakeEngine:
    """A deliberately simple, in-memory :class:`~madeleine.core.types.MemoryEngine`.

    * PULL score = mean similarity over shared channels.
    * PUSH fires when score >= ``trigger_threshold``, some shared channel other than
      ``time`` exists, and the memory has not fired within ``refractory``.
    """

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        trigger_threshold: float = 0.6,
        refractory: timedelta = timedelta(hours=1),
    ) -> None:
        self._clock: Clock = clock or FakeClock()
        self._config = MadeleineConfig(db_path=":memory:", text_backend="hash")
        self._registry = FakeRegistry()
        self._queue = FakeQueue()
        self._memories: dict[str, Memory] = {}
        self._fired: dict[str, datetime] = {}
        self._threshold = trigger_threshold
        self._refractory = refractory
        self._lock = threading.Lock()
        self.closed = False

    # -- protocol properties ---------------------------------------------------------
    @property
    def config(self) -> MadeleineConfig:
        return self._config

    @property
    def channels(self) -> FakeRegistry:
        return self._registry

    @property
    def store(self) -> MemoryStore:
        raise NotImplementedError("FakeEngine has no MemoryStore; use the facade methods")

    @property
    def queue(self) -> FakeQueue:
        return self._queue

    def now(self) -> datetime:
        return self._clock.now()

    # -- helpers ---------------------------------------------------------------------
    def _observations(
        self,
        signals: Sequence[RawSignal],
        context: SensoryContext,
        include_time: bool,
        include_geo: bool,
    ) -> tuple[dict[ChannelName, Observation], SensoryContext]:
        obs: dict[ChannelName, Observation] = {}
        if include_time:
            obs[TIME] = Observation(TIME, timestamp=context.timestamp)
        if include_geo and context.geohash:
            obs[GEO] = Observation(GEO, text=context.geohash)
        for s in signals:
            obs[s.channel] = _fake_normalize(s, context.timestamp)
        if context.geohash is None and GEO in obs and obs[GEO].text:
            context = replace(context, geohash=obs[GEO].text)
        return obs, context

    def _score(self, cue: Cue, memory: Memory, channels: frozenset[str] | None) -> ScoredMemory:
        shared = set(cue.observations) & set(memory.fingerprint.observations)
        if channels is not None:
            shared &= channels
        scores = {
            c: _fake_similarity(cue.observations[c], memory.fingerprint.observations[c])
            for c in shared
        }
        score = sum(scores.values()) / len(scores) if scores else 0.0
        return ScoredMemory(memory=memory, score=score, channel_scores=scores)

    # -- MemoryEngine ----------------------------------------------------------------
    def build_cue(
        self,
        signals: Sequence[RawSignal] = (),
        *,
        context: SensoryContext | None = None,
        include_time: bool = True,
        include_geo: bool = True,
        source: str = "unknown",
    ) -> Cue:
        ctx = context or SensoryContext(timestamp=self._clock.now())
        obs, ctx = self._observations(signals, ctx, include_time, include_geo)
        return Cue(id=new_id("cue"), observations=obs, context=ctx, source=source)

    def remember(
        self,
        content: str,
        signals: Sequence[RawSignal] = (),
        *,
        context: SensoryContext | None = None,
        media_ref: str | None = None,
        metadata: Mapping[str, str] | None = None,
        embed_content: bool = True,
        store_raw: bool | None = None,
    ) -> Memory:
        if not content.strip():
            raise ObservationError("content must not be empty")
        ctx = context or SensoryContext(timestamp=self._clock.now())
        sigs = list(signals)
        if embed_content and not any(s.channel == TEXT for s in sigs):
            sigs.append(RawSignal(TEXT, content))
        obs, ctx = self._observations(sigs, ctx, True, True)
        keep_raw = self._config.store_raw if store_raw is None else store_raw
        if not keep_raw:
            obs = {k: o.without_raw() for k, o in obs.items()}
        fp = SensoryFingerprint(
            embeddings=dict.fromkeys(obs, (1.0,)), context=ctx, observations=obs
        )
        mem = Memory(
            id=new_id("mem"),
            content=content,
            fingerprint=fp,
            created_at=self._clock.now(),
            media_ref=media_ref,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._memories[mem.id] = mem
        return mem

    def recall(self, query: RecallQuery) -> list[ScoredMemory]:
        with self._lock:
            memories = list(self._memories.values())
        if query.tags:
            memories = [m for m in memories if query.tags <= m.context.tags]
        scored = [self._score(query.cue, m, query.channels) for m in memories]
        floor = query.min_score if query.min_score is not None else 0.0
        scored = [s for s in scored if s.channel_scores and s.score >= floor]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[: query.k]

    def sense(self, cue: Cue) -> list[Recollection]:
        fired: list[Recollection] = []
        now = cue.context.timestamp
        with self._lock:
            memories = list(self._memories.values())
            candidates = []
            for m in memories:
                s = self._score(cue, m, None)
                if s.score < self._threshold or not (set(s.channel_scores) - {TIME}):
                    continue
                last = self._fired.get(m.id)
                if last is not None and abs(now - last) < self._refractory:
                    continue
                candidates.append(s)
            candidates.sort(key=lambda s: s.score, reverse=True)
            for s in candidates[:3]:
                self._fired[s.memory.id] = now
                fired.append(
                    Recollection(
                        id=new_id("rec"),
                        match=s,
                        cue_id=cue.id,
                        cue_channels=cue.channels,
                        triggered_at=now,
                        mode=RecallMode.PUSH,
                        cue_source=cue.source,
                    )
                )
        for r in fired:
            self._queue.put(r)
        return fired

    def pending_recollections(
        self, *, limit: int | None = None, drain: bool = True
    ) -> list[Recollection]:
        return self._queue.drain(limit) if drain else self._queue.peek(limit)

    def forget(self, memory_id: str) -> bool:
        with self._lock:
            existed = self._memories.pop(memory_id, None) is not None
            self._fired.pop(memory_id, None)
        self._queue.discard_memory(memory_id)
        return existed

    def get(self, memory_id: str) -> Memory:
        """Convenience for tests (not part of the protocol)."""
        try:
            return self._memories[memory_id]
        except KeyError:
            raise MemoryNotFoundError(memory_id) from None

    def close(self) -> None:
        self.closed = True


def _static_conformance(engine: FakeEngine) -> None:
    """mypy-only check that the fakes satisfy the contract protocols."""
    from madeleine.core.types import ChannelRegistry, MemoryEngine, RecollectionQueue

    _e: MemoryEngine = engine
    _q: RecollectionQueue = engine.queue
    _r: ChannelRegistry = engine.channels
