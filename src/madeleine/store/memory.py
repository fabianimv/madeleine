"""In-memory, thread-safe :class:`MemoryStore` and :class:`RecollectionQueue` (tests, ephemeral)."""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta

import numpy as np

from madeleine.core.fusion import is_refractory
from madeleine.core.types import (
    ChannelName,
    ChannelSpec,
    Memory,
    MemoryNotFoundError,
    Recollection,
    RecollectionListener,
    StoreError,
    Unsubscribe,
    Vector,
)

__all__ = ["InMemoryRecollectionQueue", "InMemoryStore", "l2_nearest"]

_log = logging.getLogger(__name__)


def l2_nearest(
    ids: Sequence[str], matrix: np.ndarray, vector: Vector, k: int
) -> list[tuple[str, float]]:
    """Exact float32 L2 k-nearest over the rows of ``matrix`` (ties broken by id)."""
    if k <= 0 or not ids:
        return []
    q = np.asarray(vector, dtype=np.float32)
    if matrix.ndim != 2 or q.shape != (matrix.shape[1],):
        raise ValueError(f"query vector has length {q.shape}, expected {matrix.shape[1:]}")
    diff = matrix - q
    dist = np.sqrt(np.einsum("ij,ij->i", diff, diff, dtype=np.float32))
    order = sorted(range(len(ids)), key=lambda i: (float(dist[i]), ids[i]))
    return [(ids[i], float(dist[i])) for i in order[:k]]


class InMemoryStore:
    """A :class:`~madeleine.core.types.MemoryStore` kept in process memory.

    ``nearest`` is exact L2 brute force with numpy. Memories are kept as given (callers that
    want parity with SQLite should pass float32-representable vectors, as the engine does).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._memories: dict[str, Memory] = {}
        self._specs: dict[ChannelName, ChannelSpec] = {}
        self._vectors: dict[ChannelName, dict[str, np.ndarray]] = {}
        self._triggers: dict[str, list[datetime]] = {}
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise StoreError("store is closed")

    def register_channel(self, spec: ChannelSpec) -> None:
        with self._lock:
            self._check_open()
            known = self._specs.get(spec.name)
            changed = known is not None and (known.dim, known.model_id) != (
                spec.dim,
                spec.model_id,
            )
            if changed and known is not None and self._vectors.get(spec.name):
                raise StoreError(
                    f"channel {spec.name!r} holds vectors with dim={known.dim}, "
                    f"model_id={known.model_id!r}; cannot use dim={spec.dim}, "
                    f"model_id={spec.model_id!r}"
                )
            self._specs[spec.name] = spec
            self._vectors.setdefault(spec.name, {})

    def check_channel(self, spec: ChannelSpec) -> bool:
        """Read-only counterpart of :meth:`register_channel` (for recall paths, R30).

        ``True`` if ``spec`` is registered as is; ``False`` if the channel is unknown or holds
        no vectors under a different ``dim``/``model_id`` (nothing to match: treat as
        empty). Raises :class:`StoreError` exactly where :meth:`register_channel` would.
        Never writes.
        """
        with self._lock:
            self._check_open()
            known = self._specs.get(spec.name)
            if known is None:
                return False
            if (known.dim, known.model_id) == (spec.dim, spec.model_id):
                return True
            if self._vectors.get(spec.name):
                raise StoreError(
                    f"channel {spec.name!r} holds vectors with dim={known.dim}, "
                    f"model_id={known.model_id!r}; cannot use dim={spec.dim}, "
                    f"model_id={spec.model_id!r}"
                )
            return False

    def add(self, memory: Memory) -> None:
        with self._lock:
            self._check_open()
            if memory.id in self._memories:
                raise ValueError(f"memory {memory.id!r} already exists")
            for name, vec in memory.fingerprint.embeddings.items():
                spec = self._specs.get(name)
                if spec is None:
                    raise ValueError(f"channel {name!r} is not registered")
                if len(vec) != spec.dim:
                    raise ValueError(f"vector for {name!r} has length {len(vec)}, not {spec.dim}")
            self._memories[memory.id] = memory
            for name, vec in memory.fingerprint.embeddings.items():
                self._vectors[name][memory.id] = np.asarray(vec, dtype=np.float32)

    def get(self, memory_id: str) -> Memory:
        with self._lock:
            self._check_open()
            try:
                return self._memories[memory_id]
            except KeyError:
                raise MemoryNotFoundError(memory_id) from None

    def get_many(self, memory_ids: Sequence[str]) -> list[Memory]:
        with self._lock:
            self._check_open()
            return [self._memories[i] for i in memory_ids if i in self._memories]

    def delete(self, memory_id: str) -> bool:
        with self._lock:
            self._check_open()
            existed = self._memories.pop(memory_id, None) is not None
            for vecs in self._vectors.values():
                vecs.pop(memory_id, None)
            self._triggers.pop(memory_id, None)
            return existed

    def iter_memories(self, *, tags: frozenset[str] | None = None) -> Iterator[Memory]:
        with self._lock:
            self._check_open()
            snapshot = list(self._memories.values())
        for m in snapshot:
            if not tags or tags <= m.context.tags:
                yield m

    def count(self) -> int:
        with self._lock:
            return len(self._memories)

    def nearest(self, channel: ChannelName, vector: Vector, k: int) -> list[tuple[str, float]]:
        with self._lock:
            self._check_open()
            vecs = self._vectors.get(channel)
            if not vecs or k <= 0:
                return []
            ids = list(vecs)
            matrix = np.stack([vecs[i] for i in ids])
        return l2_nearest(ids, matrix, vector, k)

    def nearest_tagged(
        self, channel: ChannelName, vector: Vector, k: int, tags: frozenset[str]
    ) -> list[tuple[str, float]]:
        """Like :meth:`nearest`, restricted to memories having *all* ``tags`` (exact).

        An extension of the :class:`~madeleine.core.types.MemoryStore` protocol used by the
        engine so that tagged recall cannot be starved by closer untagged memories.
        """
        if not tags:
            return self.nearest(channel, vector, k)
        with self._lock:
            self._check_open()
            vecs = self._vectors.get(channel)
            if not vecs or k <= 0:
                return []
            ids = [i for i in vecs if tags <= self._memories[i].context.tags]
            if not ids:
                return []
            matrix = np.stack([vecs[i] for i in ids])
        return l2_nearest(ids, matrix, vector, k)

    def record_trigger(self, memory_id: str, at: datetime) -> None:
        with self._lock:
            self._check_open()
            self._triggers.setdefault(memory_id, []).append(at)

    def last_triggered(self, memory_id: str) -> datetime | None:
        with self._lock:
            self._check_open()
            times = self._triggers.get(memory_id)
            return max(times) if times else None

    def claim_trigger(self, memory_id: str, at: datetime, refractory_period: timedelta) -> bool:
        """Atomically record a trigger unless the memory is refractory: any recorded trigger
        within ``refractory_period`` of ``at`` (not only the latest) blocks it, and so does a
        memory that does not exist (any more)."""
        with self._lock:
            self._check_open()
            if memory_id not in self._memories:
                return False
            times = self._triggers.get(memory_id, ())
            if any(is_refractory(t, at, refractory_period) for t in times):
                return False
            self.record_trigger(memory_id, at)
            return True

    def close(self) -> None:
        with self._lock:
            self._closed = True


class InMemoryRecollectionQueue:
    """A thread-safe, in-process :class:`~madeleine.core.types.RecollectionQueue`."""

    def __init__(self) -> None:
        self._items: list[Recollection] = []
        self._listeners: list[RecollectionListener] = []
        self._lock = threading.Lock()

    def put(self, recollection: Recollection) -> None:
        with self._lock:
            self._items.append(recollection)
            listeners = list(self._listeners)
        _notify(listeners, recollection)

    def peek(self, limit: int | None = None) -> list[Recollection]:
        n = _check_limit(limit)
        with self._lock:
            return list(self._items if n is None else self._items[:n])

    def drain(self, limit: int | None = None) -> list[Recollection]:
        n = _check_limit(limit)
        with self._lock:
            n = len(self._items) if n is None else min(n, len(self._items))
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

    def close(self) -> None:
        """No-op (present for symmetry with the SQLite queue)."""


def _check_limit(limit: int | None) -> int | None:
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")
    return limit


def _notify(listeners: Sequence[RecollectionListener], recollection: Recollection) -> None:
    for listener in listeners:
        try:
            listener(recollection)
        except Exception:
            _log.exception("recollection listener failed")
