"""The recall engine: :class:`Madeleine` implements :class:`~madeleine.core.types.MemoryEngine`.

Composition::

    engine = create_engine(load_config())        # SQLite at config.db_path, default channels
    mem = engine.remember("Ana proposed here", context=SensoryContext(ts, geohash="u09tunq"))
    fired = engine.sense(engine.build_cue(context=SensoryContext(now, geohash="u09tunq")))
    engine.pending_recollections()                # drained by hooks / MCP

See ADR 0001 for the semantics (fusion formula, PULL vs PUSH, refractory period).
"""

from __future__ import annotations

import importlib
import logging
import math
import threading
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Protocol, cast, runtime_checkable

from madeleine.core.fusion import channel_weight, fuse, push_eligible
from madeleine.core.types import (
    GEO,
    TEXT,
    TIME,
    Channel,
    ChannelName,
    ChannelNotAvailableError,
    ChannelRegistry,
    ChannelSpec,
    Clock,
    Cue,
    MadeleineConfig,
    MadeleineError,
    Memory,
    MemoryStore,
    Observation,
    ObservationError,
    RawSignal,
    RecallMode,
    RecallQuery,
    Recollection,
    RecollectionQueue,
    ScoredMemory,
    SensoryContext,
    SensoryFingerprint,
    SimilarityFn,
    SystemClock,
    Vector,
    new_id,
)

__all__ = ["Madeleine", "create_engine", "rank"]

_log = logging.getLogger(__name__)


@runtime_checkable
class _Closeable(Protocol):
    def close(self) -> None: ...


@runtime_checkable
class _ChannelChecker(Protocol):
    """Optional store extension: read-only ``register_channel`` check (R30)."""

    def check_channel(self, spec: ChannelSpec) -> bool: ...


@runtime_checkable
class _TaggedNearest(Protocol):
    """Optional store extension: KNN restricted to memories having all ``tags`` (R9)."""

    def nearest_tagged(
        self, channel: ChannelName, vector: Vector, k: int, tags: frozenset[str]
    ) -> list[tuple[str, float]]: ...


def _to_f32(vector: Sequence[float], channel: Channel) -> Vector:
    """Validate an embedder's output and round it to float32 (what stores persist)."""
    import numpy as np

    if len(vector) != channel.spec.dim:
        raise MadeleineError(
            f"embedder for {channel.name!r} returned {len(vector)} values, expected "
            f"{channel.spec.dim}"
        )
    arr = np.asarray(vector, dtype=np.float32)
    if not bool(np.isfinite(arr).all()):
        raise MadeleineError(f"embedder for {channel.name!r} returned non-finite values")
    return tuple(float(x) for x in arr.tolist())


def _sort_key(s: ScoredMemory) -> tuple[float, float, float, str]:
    m = s.memory
    return (s.score, m.context.timestamp.timestamp(), m.created_at.timestamp(), m.id)


def rank(scored: Sequence[ScoredMemory]) -> list[ScoredMemory]:
    """Score descending; ties broken by newer memory (context timestamp, created_at, id)."""
    return sorted(scored, key=_sort_key, reverse=True)


class Madeleine:
    """The reference :class:`~madeleine.core.types.MemoryEngine`.

    Thread-safe. Construct with :func:`create_engine`. ``close`` closes the store/queue only
    if the engine created them (``owns_store`` / ``owns_queue``).
    """

    def __init__(
        self,
        config: MadeleineConfig,
        *,
        registry: ChannelRegistry,
        store: MemoryStore,
        queue: RecollectionQueue,
        clock: Clock | None = None,
        owns_store: bool = False,
        owns_queue: bool = False,
    ) -> None:
        self._config = config
        self._registry = registry
        self._store = store
        self._queue = queue
        self._clock: Clock = clock or SystemClock()
        self._owns_store = owns_store
        self._owns_queue = owns_queue
        self._registered: set[ChannelName] = set()
        self._register_lock = threading.Lock()
        self._push_lock = threading.RLock()  # re-entrant: queue listeners may call forget
        self._closed = False

    # -- protocol properties -------------------------------------------------------------
    @property
    def config(self) -> MadeleineConfig:
        return self._config

    @property
    def channels(self) -> ChannelRegistry:
        return self._registry

    @property
    def store(self) -> MemoryStore:
        return self._store

    @property
    def queue(self) -> RecollectionQueue:
        return self._queue

    @property
    def clock(self) -> Clock:
        return self._clock

    def now(self) -> datetime:
        """Current time from the injected clock (timezone-aware)."""
        return self._clock.now()

    # -- observations --------------------------------------------------------------------
    def _channel(self, name: ChannelName) -> Channel:
        if name not in self._registry:
            raise ChannelNotAvailableError(
                f"channel {name!r} is not enabled (available: {', '.join(self._registry.names())})"
            )
        return self._registry.get(name)

    def _normalize(self, channel: Channel, signal: RawSignal) -> Observation:
        obs = channel.adapter.normalize(signal)
        if obs.channel != channel.name:
            raise ObservationError(
                f"adapter for {channel.name!r} returned an observation for {obs.channel!r}"
            )
        return obs

    def _observe(
        self,
        signals: Sequence[RawSignal],
        context: SensoryContext,
        *,
        include_time: bool,
        include_geo: bool,
    ) -> tuple[dict[ChannelName, Observation], SensoryContext]:
        now = self._clock.now()
        explicit: dict[ChannelName, Observation] = {}
        for signal in signals:
            if signal.channel in explicit:
                raise ObservationError(f"more than one signal for channel {signal.channel!r}")
            channel = self._channel(signal.channel)
            if signal.observed_at is None:
                signal = replace(signal, observed_at=now)
            explicit[signal.channel] = self._normalize(channel, signal)
        observations: dict[ChannelName, Observation] = {}
        if include_time and TIME not in explicit and TIME in self._registry:
            observations[TIME] = self._normalize(
                self._registry.get(TIME),
                RawSignal(TIME, context.timestamp, observed_at=context.timestamp),
            )
        if include_geo and context.geohash and GEO not in explicit and GEO in self._registry:
            observations[GEO] = self._normalize(
                self._registry.get(GEO), RawSignal(GEO, context.geohash)
            )
        observations.update(explicit)
        geo = explicit.get(GEO)
        if context.geohash is None and geo is not None and geo.text:
            try:
                context = replace(context, geohash=geo.text)
            except ValueError as exc:
                raise ObservationError(f"geo observation is not a geohash: {exc}") from exc
        return observations, context

    def _embed(self, channel: Channel, observation: Observation) -> Vector:
        return _to_f32(channel.embedder.embed(observation), channel)

    def _ensure_registered(self, names: Collection[ChannelName]) -> None:
        missing = [n for n in names if n not in self._registered]
        if not missing:
            return
        with self._register_lock:
            for name in missing:
                if name not in self._registered:
                    self._store.register_channel(self._channel(name).spec)
                    self._registered.add(name)

    def _searchable(self, names: Collection[ChannelName]) -> list[ChannelName]:
        """Channels worth a KNN query, without writing anything (R30).

        Read paths never register channels: a channel this engine has not registered is
        only *checked* (``StoreError`` if the store holds vectors from another model/dim);
        a channel the store does not know (or holds no vectors for under this spec) has no
        candidates. Stores without ``check_channel`` are queried as is.
        """
        store = self._store
        if not isinstance(store, _ChannelChecker):
            return list(names)
        return [
            name
            for name in names
            if name in self._registered or store.check_channel(self._channel(name).spec)
        ]

    # -- MemoryEngine --------------------------------------------------------------------
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
        observations, ctx = self._observe(
            signals, ctx, include_time=include_time, include_geo=include_geo
        )
        return Cue(id=new_id("cue"), observations=observations, context=ctx, source=source)

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
        if not content or not content.strip():
            raise ObservationError("memory content must not be empty")
        ctx = context or SensoryContext(timestamp=self._clock.now())
        sigs = list(signals)
        if embed_content and TEXT in self._registry and not any(s.channel == TEXT for s in sigs):
            sigs.append(RawSignal(TEXT, content))
        observations, ctx = self._observe(sigs, ctx, include_time=True, include_geo=True)
        embeddings = {
            name: self._embed(self._registry.get(name), obs) for name, obs in observations.items()
        }
        keep_raw = self._config.store_raw if store_raw is None else store_raw
        if not keep_raw:
            observations = {name: obs.without_raw() for name, obs in observations.items()}
        self._ensure_registered(embeddings)
        memory = Memory(
            id=new_id("mem"),
            content=content,
            fingerprint=SensoryFingerprint(
                embeddings=embeddings, context=ctx, observations=observations
            ),
            created_at=self._clock.now(),
            media_ref=media_ref,
            metadata=dict(metadata or {}),
        )
        self._store.add(memory)
        _log.debug("remembered %s on channels %s", memory.id, sorted(embeddings))
        return memory

    def _score(
        self,
        cue: Cue,
        *,
        channels: frozenset[ChannelName] | None,
        tags: frozenset[str] | None,
        k: int,
    ) -> list[ScoredMemory]:
        """Candidate generation (per-channel KNN union) then exact fusion; unsorted."""
        recall = self._config.recall
        weights: dict[ChannelName, float] = {}
        similarities: dict[ChannelName, SimilarityFn] = {}
        vectors: dict[ChannelName, Vector] = {}
        for name, obs in cue.observations.items():
            if channels is not None and name not in channels:
                continue
            channel = self._channel(name)
            w = channel_weight(name, channel.spec.default_weight, recall)
            if not (w > 0.0 and math.isfinite(w)):
                continue
            weights[name] = w
            similarities[name] = channel.similarity
            vectors[name] = self._embed(channel, obs)
        if not vectors:
            return []
        searchable = {name: vectors[name] for name in self._searchable(vectors)}
        candidate_ids = self._candidates(searchable, max(recall.candidate_k, k), tags)
        memories = self._store.get_many(list(candidate_ids))
        now = cue.context.timestamp
        out: list[ScoredMemory] = []
        for memory in memories:
            if tags and not tags <= memory.context.tags:
                continue
            scored = fuse(vectors, memory, similarities, weights, recall, now)
            if scored.channel_scores:
                out.append(scored)
        return out

    def _candidates(
        self, vectors: Mapping[ChannelName, Vector], k: int, tags: frozenset[str] | None
    ) -> dict[str, None]:
        """Union of per-channel nearest neighbours (insertion-ordered ids).

        ``k`` per channel is ``max(candidate_k, requested results)``. With ``tags`` the KNN
        runs over the tagged subset only (``nearest_tagged`` when the store has it, else
        every tagged memory is a candidate), so tagged matches are never starved (R9).
        """
        store = self._store
        out: dict[str, None] = {}
        if tags and not isinstance(store, _TaggedNearest):
            return dict.fromkeys(m.id for m in store.iter_memories(tags=tags))
        for name, vec in vectors.items():
            if tags and isinstance(store, _TaggedNearest):
                hits = store.nearest_tagged(name, vec, k, tags)
            else:
                hits = store.nearest(name, vec, k)
            out.update(dict.fromkeys(memory_id for memory_id, _dist in hits))
        return out

    def recall(self, query: RecallQuery) -> list[ScoredMemory]:
        floor = (
            query.min_score if query.min_score is not None else self._config.recall.pull_threshold
        )
        scored = self._score(query.cue, channels=query.channels, tags=query.tags, k=query.k)
        return rank([s for s in scored if s.score >= floor])[: query.k]

    def sense(self, cue: Cue) -> list[Recollection]:
        recall = self._config.recall
        solo = {name for name in cue.observations if self._channel(name).spec.solo_trigger}
        eligible = rank(
            [
                s
                for s in self._score(cue, channels=None, tags=None, k=recall.max_push_per_cue)
                if push_eligible(s, solo, recall)
            ]
        )
        at = cue.context.timestamp
        fired: list[Recollection] = []
        with self._push_lock:
            for scored in eligible:
                if len(fired) >= recall.max_push_per_cue:
                    break
                if not self._claim(scored.memory.id, at):
                    continue
                rec = Recollection(
                    id=new_id("rec"),
                    match=scored,
                    cue_id=cue.id,
                    cue_channels=cue.channels,
                    triggered_at=at,
                    mode=RecallMode.PUSH,
                    cue_source=cue.source,
                )
                # Enqueued under the push lock so an in-process ``forget`` cannot slip in
                # between the claim and the put (R15). Across processes, the store refuses to
                # claim a deleted memory and the SQLite queue refuses orphans.
                self._queue.put(rec)
                fired.append(rec)
        if fired:
            _log.debug("cue %s fired %d recollection(s)", cue.id, len(fired))
        return fired

    def _claim(self, memory_id: str, at: datetime) -> bool:
        # claim_trigger is atomic across processes (see MemoryStore.claim_trigger).
        return self._store.claim_trigger(memory_id, at, self._config.recall.refractory_period)

    def pending_recollections(
        self, *, limit: int | None = None, drain: bool = True
    ) -> list[Recollection]:
        return self._queue.drain(limit) if drain else self._queue.peek(limit)

    def get_memory(self, memory_id: str) -> Memory:
        """One memory by id (raises :class:`MemoryNotFoundError`). No side effects."""
        return self._store.get(memory_id)

    def update_memory(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        tags: frozenset[str] | None = None,
        salience: float | None = None,
        metadata: Mapping[str, str] | None = None,
        media_ref: str | None = None,
    ) -> Memory:
        """Edit a memory's annotations; the sensory fingerprint stays as encoded (ADR 0003)."""
        if content is not None and not content.strip():
            raise ObservationError("content must not be empty")
        if salience is not None and not 0.0 <= salience <= 1.0:
            raise ObservationError(f"salience must be in [0, 1], got {salience}")
        with self._push_lock:
            old = self._store.get(memory_id)
            fp = old.fingerprint
            ctx = replace(
                fp.context,
                tags=frozenset(t.strip() for t in tags if t.strip())
                if tags is not None
                else fp.context.tags,
                salience=fp.context.salience if salience is None else float(salience),
            )
            embeddings = dict(fp.embeddings)
            observations = dict(fp.observations)
            new_content = old.content if content is None else content.strip()
            text_obs = observations.get(TEXT)
            derived = text_obs is not None and text_obs.text == old.content.strip()
            if new_content != old.content and derived and TEXT in self._registry:
                channel = self._registry.get(TEXT)
                obs = channel.adapter.normalize(RawSignal(TEXT, new_content))
                embeddings[TEXT] = self._embed(channel, obs)
                observations[TEXT] = obs.without_raw()
            updated = replace(
                old,
                content=new_content,
                fingerprint=SensoryFingerprint(
                    embeddings=embeddings, context=ctx, observations=observations
                ),
                metadata=dict(old.metadata) if metadata is None else dict(metadata),
                media_ref=old.media_ref if media_ref is None else media_ref,
            )
            self._store.update(updated)
        _log.debug("updated %s", memory_id)
        return updated

    def forget(self, memory_id: str) -> bool:
        """Delete a memory everywhere. Serialised with ``sense`` in this process; the SQLite
        store also removes the memory's queue rows in its delete transaction."""
        with self._push_lock:
            existed = self._store.delete(memory_id)
            self._queue.discard_memory(memory_id)
        return existed

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_queue and isinstance(self._queue, _Closeable):
            self._queue.close()
        if self._owns_store:
            self._store.close()

    def __enter__(self) -> Madeleine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def create_engine(
    config: MadeleineConfig,
    *,
    clock: Clock | None = None,
    registry: ChannelRegistry | None = None,
    store: MemoryStore | None = None,
    queue: RecollectionQueue | None = None,
) -> Madeleine:
    """Build a :class:`Madeleine`.

    * ``registry`` ``None`` -> ``madeleine.channels.registry.default_registry(config)``
      (imported lazily).
    * ``store`` / ``queue`` ``None`` -> SQLite at ``resolve_db_path(config.db_path)``; when
      both are created here they share one connection (so ``":memory:"`` works). With an
      injected store and no queue, the queue shares the store's SQLite database, or is
      in-process for other stores (queued recollections are rehydrated from the store's
      file). Whatever the engine creates, ``close()`` closes.
    """
    from madeleine.config import resolve_db_path
    from madeleine.store.sqlite import SQLiteDatabase, SQLiteMemoryStore, SQLiteRecollectionQueue

    if registry is None:
        module = importlib.import_module("madeleine.channels.registry")
        registry = cast(ChannelRegistry, module.default_registry(config))
    owns_store = store is None
    owns_queue = queue is None
    if store is None:
        db = SQLiteDatabase(resolve_db_path(config.db_path))
        store = SQLiteMemoryStore(db)
        if queue is None:
            queue = SQLiteRecollectionQueue(db)
    elif queue is None:
        # The SQLite queue rehydrates memories from its own file (R20), so it must share the
        # injected store's database; any other store gets an in-process queue.
        if isinstance(store, SQLiteMemoryStore):
            queue = SQLiteRecollectionQueue(store.database)
        else:
            from madeleine.store.memory import InMemoryRecollectionQueue

            _log.info("injected non-SQLite store: using an in-process recollection queue")
            queue = InMemoryRecollectionQueue()
    return Madeleine(
        config,
        registry=registry,
        store=store,
        queue=queue,
        clock=clock,
        owns_store=owns_store,
        owns_queue=owns_queue,
    )
