"""Regression tests for the pre-release review findings on the store and engine."""

from __future__ import annotations

import json
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
    Recollection,
    SensoryContext,
    SensoryFingerprint,
    StoreError,
)
from madeleine.engine import create_engine
from madeleine.store import (
    InMemoryRecollectionQueue,
    InMemoryStore,
    SQLiteDatabase,
    SQLiteMemoryStore,
    SQLiteRecollectionQueue,
    open_sqlite,
)
from madeleine.store import sqlite as sqlite_mod
from madeleine.store.codec import recollection_to_dict
from madeleine.testing import FakeClock

from .conftest import EngineFactory
from .fakes import DictRegistry, fake_registry, text_channel

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


# -- R30: recall / sense never write ---------------------------------------------------
def test_r30_reads_do_not_register_channels(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "ro.db"
    eng = create_engine(MadeleineConfig(db_path=str(path)), clock=clock, registry=fake_registry())
    cue = eng.build_cue([RawSignal("text", "hello")], context=SensoryContext(T, geohash=PARIS))
    assert eng.recall(RecallQuery(cue)) == []
    assert eng.sense(cue) == []
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT COUNT(*) FROM channels").fetchone()[0] == 0
    assert (
        conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'vec_ch%'").fetchone()[0]
        == 0
    )
    conn.close()
    m = eng.remember("hello there", context=SensoryContext(T, geohash=PARIS))
    assert eng.recall(RecallQuery(cue))[0].memory.id == m.id
    eng.close()


def test_r30_in_memory_reads_do_not_register(clock: FakeClock) -> None:
    store = InMemoryStore()
    eng = create_engine(
        MadeleineConfig(),
        clock=clock,
        registry=fake_registry(),
        store=store,
        queue=InMemoryRecollectionQueue(),
    )
    cue = eng.build_cue([RawSignal("text", "hello")], context=SensoryContext(T, geohash=PARIS))
    assert eng.recall(RecallQuery(cue)) == []
    assert eng.sense(cue) == []
    assert not store.check_channel(ChannelSpec("text", dim=4))


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_r30_check_channel_is_read_only_register(kind: str, tmp_path: Path) -> None:
    store: InMemoryStore | SQLiteMemoryStore = (
        InMemoryStore() if kind == "memory" else SQLiteMemoryStore(tmp_path / "c.db")
    )
    spec = ChannelSpec("geo", dim=3, model_id="m1")
    assert not store.check_channel(spec)  # unknown -> empty, nothing written
    store.register_channel(spec)
    assert store.check_channel(spec)
    assert not store.check_channel(ChannelSpec("geo", dim=3, model_id="m2"))  # no vectors yet
    fp = SensoryFingerprint(embeddings={"geo": (1.0, 0.0, 0.0)}, context=SensoryContext(T))
    store.add(Memory(id="mem_a", content="a", fingerprint=fp, created_at=T))
    with pytest.raises(StoreError, match="geo"):
        store.check_channel(ChannelSpec("geo", dim=3, model_id="m2"))
    with pytest.raises(StoreError):
        store.check_channel(ChannelSpec("geo", dim=4, model_id="m1"))
    assert store.check_channel(spec)
    store.close()


def test_r30_sense_still_refuses_model_change(tmp_path: Path, clock: FakeClock) -> None:
    cfg = MadeleineConfig(db_path=str(tmp_path / "m.db"))
    eng = create_engine(cfg, clock=clock, registry=DictRegistry([text_channel("model-a")]))
    eng.remember("hello")
    eng.close()
    eng2 = create_engine(cfg, clock=clock, registry=DictRegistry([text_channel("model-b")]))
    with pytest.raises(StoreError):
        eng2.sense(eng2.build_cue([RawSignal("text", "hello")]))
    eng2.close()


def test_r30_recall_with_stale_empty_registration(tmp_path: Path, clock: FakeClock) -> None:
    store = SQLiteMemoryStore(tmp_path / "s.db")
    store.register_channel(ChannelSpec("text", dim=7, model_id="old"))  # no vectors
    eng = create_engine(
        MadeleineConfig(),
        clock=clock,
        registry=fake_registry(),
        store=store,
        queue=InMemoryRecollectionQueue(),
    )
    assert eng.recall(RecallQuery(eng.build_cue([RawSignal("text", "hello")]))) == []
    assert store.check_channel(ChannelSpec("text", dim=7, model_id="old"))  # untouched
    m = eng.remember("hello")  # the write path re-registers
    assert eng.recall(RecallQuery(eng.build_cue([RawSignal("text", "hello")])))[0].memory == m
    store.close()


# -- R20: slim queue payloads ----------------------------------------------------------
_SLIM_KEYS = {
    "v",
    "id",
    "memory_id",
    "score",
    "channel_scores",
    "salience_factor",
    "decay_factor",
    "cue_id",
    "cue_channels",
    "triggered_at",
    "mode",
    "cue_source",
}


def _payloads(path: Path) -> list[dict[str, object]]:
    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT payload_json FROM queue ORDER BY seq").fetchall()
    conn.close()
    return [json.loads(r[0]) for r in rows]


def test_r20_queue_rows_hold_only_id_and_scores(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "slim.db"
    eng = create_engine(MadeleineConfig(db_path=str(path)), clock=clock, registry=fake_registry())
    m = eng.remember("CONTENT-IN-QUEUE? no", context=SensoryContext(T, geohash=PARIS))
    fired = eng.sense(eng.build_cue(context=SensoryContext(T, geohash=PARIS)))
    (payload,) = _payloads(path)
    assert set(payload) == _SLIM_KEYS
    assert payload["memory_id"] == m.id
    assert "CONTENT-IN-QUEUE" not in json.dumps(payload)
    assert eng.pending_recollections(drain=False) == fired  # rehydrated from the store
    assert eng.pending_recollections() == fired
    eng.close()


def test_r20_entry_of_missing_memory_is_dropped(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "orphan.db"
    eng = create_engine(MadeleineConfig(db_path=str(path)), clock=clock, registry=fake_registry())
    a = eng.remember("a", context=SensoryContext(T, geohash=PARIS))
    b = eng.remember("b", context=SensoryContext(T, geohash=PARIS))
    fired = eng.sense(eng.build_cue(context=SensoryContext(T, geohash=PARIS)))
    assert {r.memory.id for r in fired} == {a.id, b.id}
    conn = sqlite3.connect(path)  # an orphan row, as an older process could leave behind
    conn.execute("DELETE FROM memories WHERE id = ?", (a.id,))
    conn.commit()
    conn.close()
    assert [r.memory.id for r in eng.pending_recollections()] == [b.id]
    assert eng.queue.pending_count() == 0
    seen: list[Recollection] = []
    eng.queue.subscribe(seen.append)
    eng.queue.put(fired[0] if fired[0].memory.id == a.id else fired[1])  # memory gone
    assert eng.queue.pending_count() == 0
    assert seen == []
    eng.close()


def _legacy_db(path: Path, clock: FakeClock) -> tuple[list[Recollection], str]:
    """A schema-1 file whose queue rows were written by the codec-v1 encoder."""
    eng = create_engine(MadeleineConfig(db_path=str(path)), clock=clock, registry=fake_registry())
    kept = eng.remember("kept", context=SensoryContext(T, geohash=PARIS))
    gone = eng.remember("LEGACY-GONE-MARKER", context=SensoryContext(T, geohash=PARIS))
    fired = eng.sense(eng.build_cue(context=SensoryContext(T, geohash=PARIS)))
    eng.close()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA secure_delete = ON")  # the marker may only survive in the queue
    conn.execute("DELETE FROM queue")
    for rec in sorted(fired, key=lambda r: r.memory.id != kept.id):
        conn.execute(
            "INSERT INTO queue(rec_id, memory_id, payload_json, enqueued_at) VALUES (?,?,?,?)",
            (rec.id, rec.memory.id, json.dumps(recollection_to_dict(rec)), "2026-01-01"),
        )
    conn.execute(
        "INSERT INTO queue(rec_id, memory_id, payload_json, enqueued_at) "
        "VALUES ('rec_bad', 'x', 'not json', 'z')"
    )
    conn.execute("DELETE FROM memories WHERE id = ?", (gone.id,))
    conn.execute("DELETE FROM schema_version")
    conn.execute("INSERT INTO schema_version VALUES (1, '2026-01-01')")
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
    conn.close()
    return [r for r in fired if r.memory.id == kept.id], gone.id


def test_r20_schema_1_queue_is_migrated(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "legacy.db"
    expected, _gone = _legacy_db(path, clock)
    conn = sqlite3.connect(path)
    raw_rows = conn.execute("SELECT payload_json FROM queue").fetchall()
    conn.close()
    assert sum('"fingerprint"' in r[0] for r in raw_rows) == 2  # really the old format
    store, queue = open_sqlite(path)
    conn = sqlite3.connect(path)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == sqlite_mod.SCHEMA_VERSION
    versions = conn.execute("SELECT version FROM schema_version ORDER BY version").fetchall()
    assert versions == [(1,), (sqlite_mod.SCHEMA_VERSION,)]
    conn.close()
    assert [set(p) for p in _payloads(path)] == [_SLIM_KEYS]
    assert queue.drain() == expected
    store.close()
    queue.close()
    assert b"LEGACY-GONE-MARKER" not in b"".join(p.read_bytes() for p in _db_files(path))


def test_r20_v1_row_written_after_migration_still_decodes(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "late.db"
    eng = create_engine(MadeleineConfig(db_path=str(path)), clock=clock, registry=fake_registry())
    eng.remember("x", context=SensoryContext(T, geohash=PARIS))
    (rec,) = eng.sense(eng.build_cue(context=SensoryContext(T, geohash=PARIS)))
    eng.pending_recollections()
    conn = sqlite3.connect(path)  # e.g. an older madeleine process still running
    conn.execute(
        "INSERT INTO queue(rec_id, memory_id, payload_json, enqueued_at) VALUES (?,?,?,?)",
        (rec.id, rec.memory.id, json.dumps(recollection_to_dict(rec)), "2026-01-01"),
    )
    conn.commit()
    conn.close()
    assert eng.pending_recollections() == [rec]
    eng.close()


def test_r20_injected_store_gets_a_matching_queue(tmp_path: Path, clock: FakeClock) -> None:
    cfg = MadeleineConfig(db_path=str(tmp_path / "config-path.db"))
    store = SQLiteMemoryStore(tmp_path / "elsewhere.db")
    eng = create_engine(cfg, clock=clock, registry=fake_registry(), store=store)
    assert isinstance(eng.queue, SQLiteRecollectionQueue)
    assert eng.queue.database is store.database
    eng.remember("x", context=SensoryContext(T, geohash=PARIS))
    fired = eng.sense(eng.build_cue(context=SensoryContext(T, geohash=PARIS)))
    assert eng.pending_recollections() == fired != []
    eng.close()
    assert store.count() == 1  # injected store stays open
    store.close()
    eng2 = create_engine(cfg, clock=clock, registry=fake_registry(), store=InMemoryStore())
    assert isinstance(eng2.queue, InMemoryRecollectionQueue)
    eng2.close()
