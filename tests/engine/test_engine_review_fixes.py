"""Regression tests for the pre-release review findings on the store and engine."""

from __future__ import annotations

import os
import sqlite3
import stat
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from madeleine.core import geohash
from madeleine.core.types import (
    ChannelSpec,
    MadeleineConfig,
    Memory,
    MemoryStore,
    RawSignal,
    RecallConfig,
    RecallQuery,
    SensoryContext,
    SensoryFingerprint,
)
from madeleine.engine import create_engine
from madeleine.store import (
    InMemoryRecollectionQueue,
    InMemoryStore,
    SQLiteDatabase,
    SQLiteMemoryStore,
    open_sqlite,
)
from madeleine.testing import FakeClock

from .conftest import EngineFactory
from .fakes import fake_registry

T = datetime(2026, 6, 1, 20, 0, tzinfo=UTC)
PARIS = "u09tunq"


def _db_files(path: Path) -> list[Path]:
    return [p for p in path.parent.iterdir() if p.name.startswith(path.name)]


# -- R7: forget leaves no recoverable bytes ---------------------------------------------
@pytest.mark.parametrize("use_vec", [True, False])
def test_r7_forget_leaves_no_bytes_in_db_files(
    tmp_path: Path, clock: FakeClock, use_vec: bool
) -> None:
    path = tmp_path / "secure.db"
    content = "CONTENT-MARKER-7f3a9c the secret we must forget"
    raw = b"RAWPIXELS-MARKER-" + bytes(range(256)) * 8
    store, queue = open_sqlite(path, use_sqlite_vec=use_vec)
    eng = create_engine(
        MadeleineConfig(db_path=str(path)),
        clock=clock,
        registry=fake_registry(),
        store=store,
        queue=queue,
    )
    keep = eng.remember("an unrelated memory that stays", context=SensoryContext(T))
    m = eng.remember(
        content,
        [RawSignal("image", raw, mime_type="image/png")],
        context=SensoryContext(T, geohash=PARIS),
        store_raw=True,
    )
    assert eng.sense(eng.build_cue(context=SensoryContext(T, geohash=PARIS)))
    assert eng.forget(m.id)
    eng.close()
    store.close()
    queue.close()
    blob = b"".join(p.read_bytes() for p in _db_files(path))
    assert b"CONTENT-MARKER-7f3a9c" not in blob
    assert b"RAWPIXELS-MARKER-" not in blob
    reopened = SQLiteMemoryStore(path, use_sqlite_vec=use_vec)
    assert [x.id for x in reopened.iter_memories()] == [keep.id]
    reopened.close()


def test_r7_secure_delete_pragma_on_every_connection(tmp_path: Path) -> None:
    db = SQLiteDatabase(tmp_path / "p.db")
    with db.read() as conn:
        assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
    db.acquire().release()


# -- R16: owner-only database files ---------------------------------------------------
@pytest.fixture
def umask_022() -> Iterator[None]:
    old = os.umask(0o022)
    yield
    os.umask(old)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
@pytest.mark.usefixtures("umask_022")
def test_r16_new_db_and_wal_files_are_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "private.db"
    store, queue = open_sqlite(path)
    store.register_channel(ChannelSpec("text", dim=2))
    files = _db_files(path)
    assert {p.name for p in files} >= {"private.db", "private.db-wal", "private.db-shm"}
    for p in files:
        assert stat.S_IMODE(p.stat().st_mode) == 0o600, p.name
    store.close()
    queue.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
@pytest.mark.usefixtures("umask_022")
def test_r16_existing_world_readable_db_is_tightened(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    sqlite3.connect(path).close()
    path.chmod(0o644)
    SQLiteMemoryStore(path).close()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


# -- R13: refractory against every trigger, not only the latest -------------------------
def _one_memory_store(kind: str, tmp_path: Path) -> MemoryStore:
    store: MemoryStore = (
        InMemoryStore() if kind == "memory" else SQLiteMemoryStore(tmp_path / "r.db")
    )
    store.register_channel(ChannelSpec("geo", dim=3))
    store.add(
        Memory(
            id="mem_a",
            content="a",
            fingerprint=SensoryFingerprint(
                embeddings={"geo": (1.0, 0.0, 0.0)}, context=SensoryContext(T)
            ),
            created_at=T,
        )
    )
    return store


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_r13_claim_checks_all_triggers_in_window(kind: str, tmp_path: Path) -> None:
    store = _one_memory_store(kind, tmp_path)
    period = timedelta(hours=6)
    assert store.claim_trigger("mem_a", T + timedelta(hours=1), period)
    assert store.claim_trigger("mem_a", T + timedelta(hours=100), period)  # out-of-order future
    assert not store.claim_trigger("mem_a", T + timedelta(hours=2), period)  # 1 h after a fire
    assert not store.claim_trigger("mem_a", T - timedelta(hours=4), period)  # 5 h before it
    assert store.claim_trigger("mem_a", T - timedelta(hours=5), period)  # exactly 6 h: fires
    assert store.claim_trigger("mem_a", T + timedelta(hours=2), timedelta(0))  # disabled
    store.close()


def test_r13_engine_out_of_order_cues(make_engine: EngineFactory) -> None:
    eng = make_engine()
    eng.remember("the cafe", context=SensoryContext(T, geohash=PARIS))

    def fire(hours: int) -> int:
        ctx = SensoryContext(T + timedelta(hours=hours), geohash=PARIS)
        return len(eng.sense(eng.build_cue(context=ctx, include_time=False)))

    assert [fire(1), fire(100), fire(2)] == [1, 1, 0]


# -- R15: forget / sense race ------------------------------------------------------------
@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_r15_claim_requires_existing_memory(kind: str, tmp_path: Path) -> None:
    store = _one_memory_store(kind, tmp_path)
    assert not store.claim_trigger("mem_missing", T, timedelta(hours=6))
    assert store.last_triggered("mem_missing") is None
    assert store.delete("mem_a")
    assert not store.claim_trigger("mem_a", T, timedelta(hours=6))
    assert store.last_triggered("mem_a") is None
    store.close()


def test_r15_forget_during_sense_enqueues_nothing(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "race.db"
    config = MadeleineConfig(db_path=str(path))
    sensing = create_engine(config, clock=clock, registry=fake_registry())
    forgetting = create_engine(config, clock=clock, registry=fake_registry())
    m = sensing.remember("forget me", context=SensoryContext(T, geohash=PARIS))
    store = sensing.store
    original = store.get_many

    def get_many_then_forget(ids: Sequence[str]) -> list[Memory]:
        got = original(ids)
        assert forgetting.forget(m.id)  # another process forgets after the fetch
        return got

    store.get_many = get_many_then_forget  # type: ignore[method-assign]
    fired = sensing.sense(sensing.build_cue(context=SensoryContext(T, geohash=PARIS)))
    assert fired == []
    assert sensing.queue.pending_count() == 0
    assert sensing.store.last_triggered(m.id) is None
    sensing.close()
    forgetting.close()


def test_r15_sqlite_delete_purges_queue_in_same_transaction(
    tmp_path: Path, clock: FakeClock
) -> None:
    path = tmp_path / "purge.db"
    eng = create_engine(MadeleineConfig(db_path=str(path)), clock=clock, registry=fake_registry())
    m = eng.remember("x", context=SensoryContext(T, geohash=PARIS))
    other = eng.remember("y", context=SensoryContext(T, geohash=PARIS))
    assert len(eng.sense(eng.build_cue(context=SensoryContext(T, geohash=PARIS)))) == 2
    assert eng.store.delete(m.id)  # store alone, no queue.discard_memory
    assert [r.memory.id for r in eng.queue.peek()] == [other.id]
    eng.close()


def test_r15_listener_may_forget_during_sense(make_engine: EngineFactory) -> None:
    eng = make_engine()
    m = eng.remember("x", context=SensoryContext(T, geohash=PARIS))
    eng.queue.subscribe(lambda rec: eng.forget(rec.memory.id))  # re-entrant, no deadlock
    eng.sense(eng.build_cue(context=SensoryContext(T, geohash=PARIS)))
    assert eng.store.count() == 0
    assert eng.queue.pending_count() == 0
    assert eng.store.last_triggered(m.id) is None


# -- R14: a live sqlite-vec connection sees fallback-mode writes -------------------------
def test_r14_vec_connection_sees_fallback_writes_without_reopen(tmp_path: Path) -> None:
    path = tmp_path / "mixed.db"
    a = SQLiteMemoryStore(path, use_sqlite_vec=True)
    b = SQLiteMemoryStore(path, use_sqlite_vec=False)
    spec = ChannelSpec("geo", dim=3)
    a.register_channel(spec)
    b.register_channel(spec)
    q = (1.0, 0.0, 0.0)

    def mem(mid: str, vec: tuple[float, ...]) -> Memory:
        fp = SensoryFingerprint(embeddings={"geo": vec}, context=SensoryContext(T))
        return Memory(id=mid, content=mid, fingerprint=fp, created_at=T)

    a.add(mem("mem_by_a", (1.0, 0.0, 0.0)))
    assert [i for i, _ in a.nearest("geo", q, 5)] == ["mem_by_a"]
    b.add(mem("mem_by_b", (0.0, 1.0, 0.0)))
    assert [i for i, _ in a.nearest("geo", q, 5)] == ["mem_by_a", "mem_by_b"]
    assert [i for i, _ in b.nearest("geo", q, 5)] == ["mem_by_a", "mem_by_b"]
    b.register_channel(ChannelSpec("text", dim=2))  # channel created without a vec0 table
    b.add(
        Memory(
            id="mem_text",
            content="t",
            fingerprint=SensoryFingerprint(
                embeddings={"text": (1.0, 0.0)}, context=SensoryContext(T)
            ),
            created_at=T,
        )
    )
    assert [i for i, _ in a.nearest("text", (1.0, 0.0), 5)] == ["mem_text"]
    assert b.delete("mem_by_a")
    assert [i for i, _ in a.nearest("geo", q, 5)] == ["mem_by_b"]
    assert not a.database.vec_index_current()  # reads never rebuild ...
    a.add(mem("mem_by_a2", (0.9, 0.1, 0.0)))  # ... the next sqlite-vec write does
    assert a.database.vec_index_current()
    assert [i for i, _ in a.nearest("geo", q, 5)] == ["mem_by_a2", "mem_by_b"]
    assert [i for i, _ in a.nearest("text", (1.0, 0.0), 5)] == ["mem_text"]
    a.close()
    b.close()


# -- R9: candidate count follows k, tags cannot be starved --------------------------------
def test_r9_recall_k_above_candidate_k(make_engine: EngineFactory) -> None:
    eng = make_engine(recall=RecallConfig(candidate_k=64))
    for i in range(100):
        eng.remember(f"memory {i}", context=SensoryContext(T, geohash=PARIS), embed_content=False)
    cue = eng.build_cue(context=SensoryContext(T, geohash=PARIS), include_time=False)
    assert len(eng.recall(RecallQuery(cue, k=100))) == 100


def test_r9_tagged_match_not_starved_by_closer_untagged(make_engine: EngineFactory) -> None:
    eng = make_engine(recall=RecallConfig(candidate_k=8))
    for i in range(20):
        eng.remember(f"near {i}", context=SensoryContext(T, geohash=PARIS), embed_content=False)
    lat, lon = geohash.decode(PARIS)
    far = geohash.encode(lat + 0.009, lon, 7)  # about 1 km north
    tagged = eng.remember(
        "tagged", context=SensoryContext(T, geohash=far, tags=frozenset({"x", "y"}))
    )
    cue = eng.build_cue(context=SensoryContext(T, geohash=PARIS), include_time=False)
    hits = eng.recall(RecallQuery(cue, k=5, tags=frozenset({"x"})))
    assert [h.memory.id for h in hits] == [tagged.id]
    assert eng.recall(RecallQuery(cue, tags=frozenset({"x", "y"})))[0].memory.id == tagged.id
    assert eng.recall(RecallQuery(cue, tags=frozenset({"x", "z"}))) == []


@pytest.mark.parametrize("kind", ["memory", "sqlite-vec", "sqlite-fallback"])
def test_r9_store_nearest_tagged(kind: str, tmp_path: Path) -> None:
    store: InMemoryStore | SQLiteMemoryStore = (
        InMemoryStore()
        if kind == "memory"
        else SQLiteMemoryStore(tmp_path / "t.db", use_sqlite_vec=kind == "sqlite-vec")
    )
    store.register_channel(ChannelSpec("geo", dim=3))
    for i, tags in enumerate([set(), {"a"}, {"a", "b"}, {"b"}, {"a", "b", "c"}]):
        fp = SensoryFingerprint(
            embeddings={"geo": (1.0, float(i), 0.0)},
            context=SensoryContext(T, tags=frozenset(tags)),
        )
        store.add(Memory(id=f"mem_{i}", content="c", fingerprint=fp, created_at=T))
    q = (1.0, 0.0, 0.0)
    assert [i for i, _ in store.nearest_tagged("geo", q, 10, frozenset({"a", "b"}))] == [
        "mem_2",
        "mem_4",
    ]
    assert [i for i, _ in store.nearest_tagged("geo", q, 1, frozenset({"a"}))] == ["mem_1"]
    assert [i for i, _ in store.nearest_tagged("geo", q, 2, frozenset())] == ["mem_0", "mem_1"]
    assert store.nearest_tagged("geo", q, 5, frozenset({"zzz"})) == []
    assert store.nearest_tagged("audio", q, 5, frozenset({"a"})) == []
    store.close()


class _PlainStore:
    """A protocol-only store (no ``nearest_tagged``): delegates to an InMemoryStore."""

    def __init__(self) -> None:
        self._inner = InMemoryStore()

    def __getattr__(self, name: str) -> object:
        if name == "nearest_tagged":
            raise AttributeError(name)
        return getattr(self._inner, name)


def test_r9_tags_on_store_without_nearest_tagged(clock: FakeClock) -> None:
    eng = create_engine(
        MadeleineConfig(recall=RecallConfig(candidate_k=2)),
        clock=clock,
        registry=fake_registry(),
        store=_PlainStore(),  # type: ignore[arg-type]
        queue=InMemoryRecollectionQueue(),
    )
    for i in range(5):
        eng.remember(f"near {i}", context=SensoryContext(T, geohash=PARIS), embed_content=False)
    lat, lon = geohash.decode(PARIS)
    far = geohash.encode(lat + 0.009, lon, 7)
    tagged = eng.remember("t", context=SensoryContext(T, geohash=far, tags=frozenset({"x"})))
    cue = eng.build_cue(context=SensoryContext(T, geohash=PARIS), include_time=False)
    assert [h.memory.id for h in eng.recall(RecallQuery(cue, tags=frozenset({"x"})))] == [tagged.id]
