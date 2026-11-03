from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from madeleine.core.types import (
    ChannelSpec,
    Memory,
    MemoryNotFoundError,
    MemoryStore,
    Observation,
    SensoryContext,
    SensoryFingerprint,
    StoreError,
)
from madeleine.store import InMemoryStore, SQLiteMemoryStore, open_sqlite
from madeleine.store import sqlite as sqlite_mod

T0 = datetime(2025, 6, 1, 21, 30, tzinfo=timezone(timedelta(hours=2)))
RAW = b"\x89PNG-SECRET-RAW-BYTES-0123456789"


def f32(values: list[float]) -> tuple[float, ...]:
    return tuple(float(x) for x in np.asarray(values, dtype=np.float32).tolist())


def make_memory(
    mid: str,
    *,
    text: tuple[float, ...] | None = None,
    geo: tuple[float, ...] | None = None,
    raw: bytes | None = None,
    tags: frozenset[str] = frozenset(),
    ts: datetime = T0,
) -> Memory:
    emb: dict[str, tuple[float, ...]] = {}
    obs: dict[str, Observation] = {}
    if text is not None:
        emb["text"] = text
        obs["text"] = Observation("text", text=f"content of {mid}")
    if geo is not None:
        emb["geo"] = geo
        obs["geo"] = Observation("geo", text="u09tunq", numbers={"lat": 48.8, "lon": 2.29})
    if raw is not None:
        emb["image"] = f32([1.0, 0.0])
        obs["image"] = Observation("image", data=raw, mime_type="image/png", uri="file:///x.png")
    ctx = SensoryContext(timestamp=ts, geohash="u09tunq", tags=tags, salience=0.8)
    return Memory(
        id=mid,
        content=f"content of {mid}",
        fingerprint=SensoryFingerprint(embeddings=emb, context=ctx, observations=obs),
        created_at=datetime(2026, 1, 2, 3, 4, 5, 678901, tzinfo=UTC),
        media_ref="file:///photos/x.jpg",
        metadata={"source": "test", "k": "v"},
    )


SPECS = (
    ChannelSpec("text", dim=4, model_id="m1"),
    ChannelSpec("geo", dim=3),
    ChannelSpec("image", dim=2, model_id="img"),
)


def _register(store: MemoryStore) -> None:
    for s in SPECS:
        store.register_channel(s)


@pytest.fixture(params=["memory", "sqlite-vec", "sqlite-fallback"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[MemoryStore]:
    s: MemoryStore
    if request.param == "memory":
        s = InMemoryStore()
    else:
        s = SQLiteMemoryStore(tmp_path / "m.db", use_sqlite_vec=request.param == "sqlite-vec")
    _register(s)
    yield s
    s.close()


def test_round_trip_all_fields(store: MemoryStore) -> None:
    m = make_memory(
        "mem_a", text=f32([0.1, 0.2, 0.3, 0.4]), geo=f32([1, 0, 0]), tags=frozenset({"a", "b"})
    )
    store.add(m)
    got = store.get("mem_a")
    assert got == m
    assert got.context.timestamp.utcoffset() == timedelta(hours=2)
    assert store.get_many(["nope", "mem_a"]) == [m]
    assert store.count() == 1
    with pytest.raises(MemoryNotFoundError):
        store.get("nope")


def test_add_validation(store: MemoryStore) -> None:
    store.add(make_memory("mem_a", text=f32([1, 0, 0, 0])))
    with pytest.raises(ValueError, match="already exists"):
        store.add(make_memory("mem_a", text=f32([1, 0, 0, 0])))
    with pytest.raises(ValueError, match="length"):
        store.add(make_memory("mem_b", text=f32([1, 0, 0])))
    bad = Memory(
        id="mem_c",
        content="x",
        fingerprint=SensoryFingerprint(embeddings={"audio": (1.0,)}, context=SensoryContext(T0)),
        created_at=T0,
    )
    with pytest.raises(ValueError, match="not registered"):
        store.add(bad)
    assert store.count() == 1


def test_register_channel_mismatch(store: MemoryStore) -> None:
    store.register_channel(ChannelSpec("text", dim=4, model_id="m1"))  # idempotent
    # no vectors yet: changing is allowed
    store.register_channel(ChannelSpec("audio", dim=4, model_id="a1"))
    store.register_channel(ChannelSpec("audio", dim=8, model_id="a2"))
    store.add(make_memory("mem_a", text=f32([1, 0, 0, 0])))
    with pytest.raises(StoreError):
        store.register_channel(ChannelSpec("text", dim=4, model_id="other"))
    with pytest.raises(StoreError):
        store.register_channel(ChannelSpec("text", dim=5, model_id="m1"))


def test_iter_tags_order_and_delete_cascade(store: MemoryStore) -> None:
    for i in range(5):
        store.add(
            make_memory(
                f"mem_{i}",
                text=f32([1, i, 0, 0]),
                tags=frozenset({"even"} if i % 2 == 0 else {"odd"}) | {"all"},
            )
        )
    assert [m.id for m in store.iter_memories()] == [f"mem_{i}" for i in range(5)]
    assert [m.id for m in store.iter_memories(tags=frozenset({"even", "all"}))] == [
        "mem_0",
        "mem_2",
        "mem_4",
    ]
    t = datetime(2026, 1, 1, tzinfo=UTC)
    store.record_trigger("mem_2", t)
    store.record_trigger("mem_2", t + timedelta(hours=1))
    store.record_trigger("mem_2", t - timedelta(hours=1))
    assert store.last_triggered("mem_2") == t + timedelta(hours=1)
    assert store.delete("mem_2")
    assert not store.delete("mem_2")
    assert store.last_triggered("mem_2") is None
    assert store.count() == 4
    assert "mem_2" not in [i for i, _ in store.nearest("text", f32([1, 2, 0, 0]), 10)]


def test_nearest_basics(store: MemoryStore) -> None:
    store.add(make_memory("mem_a", text=f32([1, 0, 0, 0])))
    store.add(make_memory("mem_b", text=f32([0, 1, 0, 0])))
    store.add(make_memory("mem_c", geo=f32([0, 0, 1])))
    hits = store.nearest("text", f32([0.9, 0.1, 0, 0]), 5)
    assert [h[0] for h in hits] == ["mem_a", "mem_b"]
    assert hits[0][1] == pytest.approx(float(np.hypot(0.1, 0.1)), rel=1e-5)
    assert store.nearest("text", f32([1, 0, 0, 0]), 1)[0][0] == "mem_a"
    assert store.nearest("audio", (1.0,), 5) == []
    assert store.nearest("text", f32([1, 0, 0, 0]), 0) == []


def test_sqlite_vec_and_fallback_give_same_nearest(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    path = tmp_path / "same.db"
    vec_store = SQLiteMemoryStore(path, use_sqlite_vec=True)
    assert vec_store.uses_sqlite_vec
    _register(vec_store)
    for i in range(200):
        vec_store.add(make_memory(f"mem_{i:03d}", text=f32(list(rng.normal(size=4)))))
    fb_store = SQLiteMemoryStore(path, use_sqlite_vec=False)
    assert not fb_store.uses_sqlite_vec
    mem_store = InMemoryStore()
    _register(mem_store)
    for m in vec_store.iter_memories():
        mem_store.add(m)
    for _ in range(20):
        q = f32(list(rng.normal(size=4)))
        a = vec_store.nearest("text", q, 25)
        b = fb_store.nearest("text", q, 25)
        c = mem_store.nearest("text", q, 25)
        assert [x[0] for x in a] == [x[0] for x in b] == [x[0] for x in c]
        np.testing.assert_allclose([x[1] for x in a], [x[1] for x in b], rtol=1e-5)
    # k above the sqlite-vec limit uses brute force transparently
    assert len(vec_store.nearest("text", f32([1, 0, 0, 0]), sqlite_mod.VEC_MAX_K + 1)) == 200
    vec_store.close()
    fb_store.close()


def test_fallback_writes_are_reindexed_by_vec_connection(tmp_path: Path) -> None:
    path = tmp_path / "dirty.db"
    s1 = SQLiteMemoryStore(path, use_sqlite_vec=True)
    _register(s1)
    s1.add(make_memory("mem_a", text=f32([1, 0, 0, 0])))
    s1.close()
    fb = SQLiteMemoryStore(path, use_sqlite_vec=False)
    fb.add(make_memory("mem_b", text=f32([0, 1, 0, 0])))
    assert fb.delete("mem_a")
    fb.close()
    s2 = SQLiteMemoryStore(path, use_sqlite_vec=True)
    assert s2.nearest("text", f32([1, 0, 0, 0]), 5) == [
        ("mem_b", pytest.approx(float(np.sqrt(2)), rel=1e-6))
    ]
    s2.close()


def test_fallback_warns_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    monkeypatch.setattr(sqlite_mod, "_load_sqlite_vec", lambda conn: "forced for test")
    monkeypatch.setattr(sqlite_mod, "_fallback_warned", False)
    with caplog.at_level(logging.WARNING, logger="madeleine.store.sqlite"):
        a = SQLiteMemoryStore(tmp_path / "w.db")
        b = SQLiteMemoryStore(tmp_path / "w.db")
    assert not a.uses_sqlite_vec
    assert sum("sqlite-vec could not be loaded" in r.message for r in caplog.records) == 1
    with pytest.raises(StoreError):
        SQLiteMemoryStore(tmp_path / "w.db", use_sqlite_vec=True)
    a.close()
    b.close()


def test_persistence_across_reopen(tmp_path: Path) -> None:
    path = tmp_path / "p.db"
    store, queue = open_sqlite(path)
    _register(store)
    m = make_memory("mem_a", text=f32([1, 0, 0, 0]), geo=f32([0, 1, 0]))
    store.add(m)
    store.record_trigger("mem_a", T0)
    store.close()
    queue.close()
    store2, queue2 = open_sqlite(path)
    assert store2.get("mem_a") == m
    assert store2.last_triggered("mem_a") == T0
    assert store2.nearest("geo", f32([0, 1, 0]), 1)[0][0] == "mem_a"
    with pytest.raises(StoreError):
        store2.register_channel(ChannelSpec("text", dim=4, model_id="m2"))
    store2.close()
    queue2.close()
    with pytest.raises(StoreError):
        store2.count()


def test_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "v.db"
    SQLiteMemoryStore(path).close()
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == sqlite_mod.SCHEMA_VERSION
    assert conn.execute("SELECT version FROM schema_version").fetchall() == [
        (sqlite_mod.SCHEMA_VERSION,)
    ]
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(StoreError, match="newer"):
        SQLiteMemoryStore(path)


def _all_db_bytes(path: Path) -> bytes:
    out = b""
    for p in path.parent.iterdir():
        if p.name.startswith(path.name):
            out += p.read_bytes()
    return out


@pytest.mark.parametrize("use_vec", [True, False])
def test_raw_bytes_only_persisted_when_present(tmp_path: Path, use_vec: bool) -> None:
    path = tmp_path / "raw.db"
    store = SQLiteMemoryStore(path, use_sqlite_vec=use_vec)
    _register(store)
    stripped = make_memory("mem_s", raw=RAW)
    stripped = Memory(
        id=stripped.id,
        content=stripped.content,
        fingerprint=SensoryFingerprint(
            embeddings=stripped.fingerprint.embeddings,
            context=stripped.context,
            observations={k: o.without_raw() for k, o in stripped.fingerprint.observations.items()},
        ),
        created_at=stripped.created_at,
    )
    store.add(stripped)
    store.close()
    assert RAW not in _all_db_bytes(path)
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM raw_blobs").fetchone()[0] == 0
    conn.close()

    store = SQLiteMemoryStore(path, use_sqlite_vec=use_vec)
    kept = make_memory("mem_k", raw=RAW)
    store.add(kept)
    assert store.get("mem_k") == kept
    assert store.get("mem_k").fingerprint.observations["image"].data == RAW
    assert store.get("mem_s").fingerprint.observations["image"].data is None
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT memory_id, data FROM raw_blobs").fetchall() == [("mem_k", RAW)]
    conn.close()
    assert store.delete("mem_k")
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM raw_blobs").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0] == 1
    conn.close()
    store.close()


def test_claim_trigger_is_atomic_across_connections(tmp_path: Path) -> None:
    import threading

    path = tmp_path / "claim.db"
    stores = [SQLiteMemoryStore(path) for _ in range(2)]
    _register(stores[0])
    stores[0].add(make_memory("mem_x", text=f32([1, 0, 0, 0])))
    wins: list[bool] = []
    lock = threading.Lock()

    def worker(s: SQLiteMemoryStore) -> None:
        for _ in range(10):
            ok = s.claim_trigger("mem_x", T0, timedelta(hours=6))
            with lock:
                wins.append(ok)

    threads = [threading.Thread(target=worker, args=(stores[i % 2],)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert wins.count(True) == 1
    assert stores[0].claim_trigger("mem_x", T0 + timedelta(hours=6), timedelta(hours=6))
    for s in stores:
        s.close()
