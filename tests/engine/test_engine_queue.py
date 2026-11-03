from __future__ import annotations

import json
import multiprocessing as mp
import sqlite3
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from madeleine.core.types import (
    Memory,
    Observation,
    RecallMode,
    Recollection,
    RecollectionQueue,
    ScoredMemory,
    SensoryContext,
    SensoryFingerprint,
)
from madeleine.store import InMemoryRecollectionQueue, SQLiteRecollectionQueue, open_sqlite
from madeleine.store.codec import recollection_from_dict, recollection_to_dict

T0 = datetime(2026, 6, 1, 21, 30, tzinfo=UTC)


def make_rec(i: int, memory_id: str | None = None, *, raw: bytes | None = None) -> Recollection:
    mid = memory_id or f"mem_{i}"
    obs = {"text": Observation("text", text=f"memory {i}")}
    emb = {"text": (0.5, 0.25), "geo": (1.0, 0.0, 0.0)}
    if raw is not None:
        obs["image"] = Observation("image", data=raw, mime_type="image/png")
        emb["image"] = (1.0,)
    mem = Memory(
        id=mid,
        content=f"memory {i}",
        fingerprint=SensoryFingerprint(
            embeddings=emb,
            context=SensoryContext(
                T0 - timedelta(days=i), geohash="u09tunq", tags=frozenset({"t"})
            ),
            observations=obs,
        ),
        created_at=T0,
        metadata={"n": str(i)},
    )
    return Recollection(
        id=f"rec_{i:05d}",
        match=ScoredMemory(
            memory=mem,
            score=0.123456789 + i * 1e-9,
            channel_scores={"geo": 1.0, "text": 0.1 / 3},
            salience_factor=1.04,
            decay_factor=0.987654321,
        ),
        cue_id=f"cue_{i}",
        cue_channels=frozenset({"geo", "text"}),
        triggered_at=T0,
        mode=RecallMode.PUSH,
        cue_source="test",
    )


@pytest.fixture(params=["memory", "sqlite"])
def queue(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[RecollectionQueue]:
    if request.param == "memory":
        q: RecollectionQueue = InMemoryRecollectionQueue()
    else:
        q = SQLiteRecollectionQueue(tmp_path / "q.db")
    yield q
    q.close()  # type: ignore[attr-defined]


def test_codec_round_trip_exact() -> None:
    r = make_rec(3)
    assert recollection_from_dict(json.loads(json.dumps(recollection_to_dict(r)))) == r


def test_fifo_peek_drain_limit(queue: RecollectionQueue) -> None:
    recs = [make_rec(i) for i in range(5)]
    for r in recs:
        queue.put(r)
    assert queue.pending_count() == 5
    assert queue.peek(2) == recs[:2]
    assert queue.peek() == recs
    assert queue.drain(0) == []
    assert queue.drain(2) == recs[:2]
    assert queue.drain() == recs[2:]
    assert queue.drain() == []
    assert queue.pending_count() == 0
    with pytest.raises(ValueError, match="limit"):
        queue.peek(-1)


def test_listeners_and_unsubscribe(queue: RecollectionQueue) -> None:
    seen: list[Recollection] = []

    def boom(r: Recollection) -> None:
        raise RuntimeError("listener failure is logged, not raised")

    unsub = queue.subscribe(seen.append)
    queue.subscribe(boom)
    queue.put(make_rec(1))
    unsub()
    queue.put(make_rec(2))
    assert [r.id for r in seen] == ["rec_00001"]
    assert queue.pending_count() == 2


def test_discard_memory(queue: RecollectionQueue) -> None:
    queue.put(make_rec(1, "mem_x"))
    queue.put(make_rec(2, "mem_y"))
    queue.put(make_rec(3, "mem_x"))
    assert queue.discard_memory("mem_x") == 2
    assert queue.discard_memory("mem_x") == 0
    assert [r.id for r in queue.peek()] == ["rec_00002"]


def test_queue_never_persists_raw_bytes(tmp_path: Path) -> None:
    secret = b"RAW-AUDIO-BYTES-DO-NOT-PERSIST"
    q = SQLiteRecollectionQueue(tmp_path / "q.db")
    q.put(make_rec(1, raw=secret))
    got = q.peek()[0]
    assert got.memory.fingerprint.observations["image"].data is None
    q.close()
    assert all(secret not in p.read_bytes() for p in tmp_path.iterdir())


def test_shared_memory_connection() -> None:
    store, q = open_sqlite(":memory:")
    q.put(make_rec(1))
    assert q.pending_count() == 1
    q.close()
    assert store.count() == 0  # connection kept alive while the store still uses it
    store.close()


def test_drain_atomic_across_two_connections(tmp_path: Path) -> None:
    path = tmp_path / "shared.db"
    queues = [SQLiteRecollectionQueue(path), SQLiteRecollectionQueue(path)]
    n = 300
    for i in range(n):
        queues[i % 2].put(make_rec(i))
    got: list[str] = []
    lock = threading.Lock()

    def worker(q: SQLiteRecollectionQueue, limit: int) -> None:
        while True:
            batch = q.drain(limit)
            if not batch:
                if q.pending_count() == 0:
                    return
                continue
            with lock:
                got.extend(r.id for r in batch)

    threads = [threading.Thread(target=worker, args=(queues[i % 2], 1 + i % 4)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(got) == n
    assert sorted(got) == sorted(f"rec_{i:05d}" for i in range(n))
    for q in queues:
        q.close()


def _drain_in_child(path: str, out: mp.Queue) -> None:  # type: ignore[type-arg]
    q = SQLiteRecollectionQueue(path)
    ids: list[str] = []
    while True:
        batch = q.drain(3)
        if not batch:
            break
        ids.extend(r.id for r in batch)
    q.close()
    out.put(ids)


def test_drain_atomic_across_processes(tmp_path: Path) -> None:
    path = str(tmp_path / "procs.db")
    q = SQLiteRecollectionQueue(path)
    n = 200
    for i in range(n):
        q.put(make_rec(i))
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    procs = [ctx.Process(target=_drain_in_child, args=(path, out)) for _ in range(3)]
    for p in procs:
        p.start()
    local: list[str] = []
    while batch := q.drain(2):
        local.extend(r.id for r in batch)
    results = [out.get(timeout=60) for _ in procs]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    every = local + [i for ids in results for i in ids]
    assert sorted(every) == sorted(f"rec_{i:05d}" for i in range(n))
    assert len(every) == n
    q.close()


def test_undecodable_entry_is_dropped(tmp_path: Path) -> None:
    path = tmp_path / "bad.db"
    q = SQLiteRecollectionQueue(path)
    q.put(make_rec(1))
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO queue(rec_id, memory_id, payload_json, enqueued_at) VALUES ('x','y','{}','z')"
    )
    conn.commit()
    conn.close()
    q.put(make_rec(2))
    assert [r.id for r in q.drain()] == ["rec_00001", "rec_00002"]
    assert q.pending_count() == 0
    q.close()
