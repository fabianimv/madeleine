from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from madeleine.core.fusion import channel_weight, fuse
from madeleine.core.types import (
    ChannelNotAvailableError,
    MadeleineConfig,
    MemoryEngine,
    ObservationError,
    RawSignal,
    RecallConfig,
    RecallQuery,
    ScoredMemory,
    SensoryContext,
    StoreError,
)
from madeleine.engine import Madeleine, create_engine, rank
from madeleine.store import InMemoryRecollectionQueue, InMemoryStore
from madeleine.testing import FakeClock

from .conftest import EngineFactory, build_engine
from .fakes import DictRegistry, fake_registry, geo_channel, text_channel, time_channel

T = datetime(2026, 6, 1, 20, 0, tzinfo=UTC)
PARIS = "u09tunq"


def test_conforms_to_protocol(make_engine: EngineFactory) -> None:
    eng = make_engine()
    assert isinstance(eng, MemoryEngine)
    _static: MemoryEngine = eng


# -- build_cue -------------------------------------------------------------------------
def test_build_cue_derives_time_and_geo(make_engine: EngineFactory, clock: FakeClock) -> None:
    eng = make_engine()
    cue = eng.build_cue(context=SensoryContext(T, geohash=PARIS), source="t")
    assert cue.channels == {"time", "geo"}
    assert cue.observations["time"].timestamp == T
    assert cue.observations["geo"].text == PARIS
    assert cue.source == "t"
    assert cue.id.startswith("cue_")
    bare = eng.build_cue(include_time=False, include_geo=False)
    assert bare.channels == frozenset()
    assert bare.context.timestamp == clock.now()
    no_geo = eng.build_cue()
    assert no_geo.channels == {"time"}


def test_build_cue_explicit_signals_win(make_engine: EngineFactory) -> None:
    eng = make_engine()
    other = datetime(2020, 1, 1, 8, 0, tzinfo=UTC)
    cue = eng.build_cue(
        [RawSignal("time", other), RawSignal("geo", (35.0, 135.0))],
        context=SensoryContext(T, geohash=PARIS),
    )
    assert cue.observations["time"].timestamp == other
    assert cue.observations["geo"].text != PARIS
    assert cue.context.geohash == PARIS  # context geohash is kept when set
    filled = eng.build_cue([RawSignal("geo", "xn76urx")], context=SensoryContext(T))
    assert filled.context.geohash == "xn76urx"
    assert filled.observations["geo"].text == "xn76urx"


def test_build_cue_errors(make_engine: EngineFactory) -> None:
    eng = make_engine()
    with pytest.raises(ChannelNotAvailableError):
        eng.build_cue([RawSignal("audio", b"x")])
    with pytest.raises(ObservationError):
        eng.build_cue([RawSignal("text", "a"), RawSignal("text", "b")])
    with pytest.raises(ObservationError):
        eng.build_cue([RawSignal("text", "   ")])


def test_derived_channels_skipped_when_not_registered(clock: FakeClock) -> None:
    eng = create_engine(
        MadeleineConfig(db_path=":memory:"),
        clock=clock,
        registry=DictRegistry([text_channel()]),
        store=InMemoryStore(),
        queue=InMemoryRecollectionQueue(),
    )
    cue = eng.build_cue(context=SensoryContext(T, geohash=PARIS))
    assert cue.channels == frozenset()
    m = eng.remember("hello", context=SensoryContext(T, geohash=PARIS))
    assert m.fingerprint.channels == {"text"}
    with pytest.raises(ChannelNotAvailableError):
        eng.build_cue([RawSignal("geo", PARIS)])


# -- remember --------------------------------------------------------------------------
def test_remember_fingerprint(make_engine: EngineFactory, clock: FakeClock) -> None:
    eng = make_engine()
    m = eng.remember(
        "Dinner at the tower",
        context=SensoryContext(T, geohash=PARIS, tags=frozenset({"paris"}), salience=0.9),
        media_ref="file:///p.jpg",
        metadata={"who": "ana"},
    )
    assert m.fingerprint.channels == {"text", "time", "geo"}
    assert m.fingerprint.observations["text"].text == "Dinner at the tower"
    assert m.created_at == clock.now()
    assert eng.store.get(m.id) == m
    assert m.id.startswith("mem_")
    no_text = eng.remember("x", embed_content=False)
    assert no_text.fingerprint.channels == {"time"}
    explicit = eng.remember("content", [RawSignal("text", "other words")])
    assert explicit.fingerprint.observations["text"].text == "other words"
    filled = eng.remember("c", [RawSignal("geo", "xn76urx")], context=SensoryContext(T))
    assert filled.context.geohash == "xn76urx"
    with pytest.raises(ObservationError):
        eng.remember("  ")
    with pytest.raises(ObservationError):
        eng.remember("dup", [RawSignal("geo", PARIS), RawSignal("geo", PARIS)])


RAW = b"\x89PNG-ENGINE-RAW-SECRET-BYTES"


@pytest.mark.parametrize(
    ("config_raw", "call_raw", "kept"),
    [
        (False, None, False),
        (False, True, True),
        (True, None, True),
        (True, False, False),
    ],
)
def test_remember_store_raw(
    tmp_path: Path, clock: FakeClock, config_raw: bool, call_raw: bool | None, kept: bool
) -> None:
    db = tmp_path / "raw.db"
    eng = build_engine("sqlite", clock, store_raw=config_raw, db_path=str(db))
    m = eng.remember("photo", [RawSignal("image", RAW, mime_type="image/png")], store_raw=call_raw)
    assert (m.fingerprint.observations["image"].data == RAW) is kept
    assert (eng.store.get(m.id).fingerprint.observations["image"].data == RAW) is kept
    eng.close()
    blob = b"".join(p.read_bytes() for p in tmp_path.iterdir())
    assert (RAW in blob) is kept
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM raw_blobs").fetchone()[0] == int(kept)
    conn.close()


def test_model_mismatch_on_reopen(tmp_path: Path, clock: FakeClock) -> None:
    cfg = MadeleineConfig(db_path=str(tmp_path / "m.db"))
    eng = create_engine(cfg, clock=clock, registry=DictRegistry([text_channel("model-a")]))
    eng.remember("hello")
    eng.close()
    eng2 = create_engine(cfg, clock=clock, registry=DictRegistry([text_channel("model-b")]))
    with pytest.raises(StoreError):
        eng2.remember("again")
    with pytest.raises(StoreError):
        eng2.recall(RecallQuery(eng2.build_cue([RawSignal("text", "hello")])))
    eng2.close()


# -- recall ----------------------------------------------------------------------------
def test_recall_basics(make_engine: EngineFactory) -> None:
    eng = make_engine()
    a = eng.remember("rain on a tin roof", context=SensoryContext(T, tags=frozenset({"home"})))
    b = eng.remember("sunny beach in Nice", context=SensoryContext(T))
    hits = eng.recall(RecallQuery(eng.build_cue([RawSignal("text", "tin roof rain")])))
    assert hits[0].memory.id == a.id
    assert set(hits[0].channel_scores) <= {"text", "time"}
    tagged = eng.recall(
        RecallQuery(eng.build_cue([RawSignal("text", "sunny")]), tags=frozenset({"home"}))
    )
    assert [h.memory.id for h in tagged] == [a.id]
    only_text = eng.recall(
        RecallQuery(eng.build_cue([RawSignal("text", "beach")]), channels=frozenset({"text"}))
    )
    assert only_text[0].memory.id == b.id
    assert all(set(h.channel_scores) == {"text"} for h in only_text)
    assert (
        eng.recall(RecallQuery(eng.build_cue([RawSignal("text", "beach")]), min_score=1.01)) == []
    )
    assert eng.recall(RecallQuery(eng.build_cue(include_time=False))) == []
    assert len(eng.recall(RecallQuery(eng.build_cue([RawSignal("text", "beach")]), k=1))) == 1
    # recall has no side effects
    assert eng.queue.pending_count() == 0
    assert eng.store.last_triggered(a.id) is None


def test_channel_weight_override_and_zero(clock: FakeClock) -> None:
    rc = RecallConfig(channel_weights={"time": 0.0})
    eng = build_engine("memory", clock, recall=rc)
    eng.remember("x", context=SensoryContext(T))
    cue = eng.build_cue(context=SensoryContext(T))
    assert eng.recall(RecallQuery(cue)) == []  # time-only cue, time weight 0


def test_recall_tie_break_newer_first(make_engine: EngineFactory) -> None:
    eng = make_engine()
    old = eng.remember("same words", context=SensoryContext(T), embed_content=True)
    new = eng.remember("same words", context=SensoryContext(T + timedelta(seconds=1)))
    cue = eng.build_cue([RawSignal("text", "same words")], include_time=False)
    hits = eng.recall(RecallQuery(cue, k=2))
    # newer memory has slightly higher decay too, so it ranks first either way
    assert [h.memory.id for h in hits] == [new.id, old.id]
    s = hits[0]
    tie = ScoredMemory(memory=old, score=s.score, channel_scores=s.channel_scores)
    assert rank([tie, s])[0].memory.id == new.id


WORDS = ["rain", "tin", "roof", "beach", "sun", "tower", "dinner", "kyoto", "train", "cafe"]
GEOS = ["u09tunq", "u09tunr", "u09whp", "xn76urx", "dr5ru7", None]
geo_st = st.sampled_from(GEOS)
text_st = st.lists(st.sampled_from(WORDS), min_size=1, max_size=4).map(" ".join)
when_st = st.datetimes(
    min_value=datetime(2015, 1, 1),  # noqa: DTZ001 - hypothesis needs naive bounds
    max_value=datetime(2026, 6, 1),  # noqa: DTZ001
    timezones=st.just(UTC),
)


def oracle(eng: Madeleine, query: RecallQuery) -> list[ScoredMemory]:
    cfg = eng.config.recall
    cue = query.cue
    vectors, sims, weights = {}, {}, {}
    for name, obs in cue.observations.items():
        if query.channels is not None and name not in query.channels:
            continue
        ch = eng.channels.get(name)
        w = channel_weight(name, ch.spec.default_weight, cfg)
        if w <= 0:
            continue
        weights[name] = w
        sims[name] = ch.similarity
        vectors[name] = eng._embed(ch, obs)
    floor = query.min_score if query.min_score is not None else cfg.pull_threshold
    out = []
    for m in eng.store.iter_memories(tags=query.tags):
        s = fuse(vectors, m, sims, weights, cfg, cue.context.timestamp)
        if s.channel_scores and s.score >= floor:
            out.append(s)
    out.sort(
        key=lambda s: (
            s.score,
            s.memory.context.timestamp.timestamp(),
            s.memory.created_at.timestamp(),
            s.memory.id,
        ),
        reverse=True,
    )
    return out[: query.k]


@settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    kind=st.sampled_from(["memory", "sqlite"]),
    memories=st.lists(
        st.tuples(st.none() | text_st, geo_st, when_st, st.floats(0, 1)), min_size=1, max_size=25
    ),
    cue_text=st.none() | text_st,
    cue_geo=geo_st,
    cue_when=when_st,
    restrict=st.none() | st.sets(st.sampled_from(["text", "geo", "time"]), min_size=1),
    k=st.integers(1, 30),
    min_score=st.none() | st.floats(0, 1),
)
def test_recall_equals_brute_force_oracle(
    kind: str,
    memories: list[tuple[str | None, str | None, datetime, float]],
    cue_text: str | None,
    cue_geo: str | None,
    cue_when: datetime,
    restrict: set[str] | None,
    k: int,
    min_score: float | None,
) -> None:
    clock = FakeClock(datetime(2026, 6, 2, tzinfo=UTC))
    eng = build_engine(kind, clock, recall=RecallConfig(candidate_k=64))
    try:
        for text, geo, when, sal in memories:
            eng.remember(
                text or "untexted",
                context=SensoryContext(when, geohash=geo, salience=sal),
                embed_content=text is not None,
            )
        assert eng.config.recall.candidate_k >= eng.store.count()
        signals = [RawSignal("text", cue_text)] if cue_text else []
        cue = eng.build_cue(signals, context=SensoryContext(cue_when, geohash=cue_geo))
        query = RecallQuery(
            cue,
            k=k,
            channels=frozenset(restrict) if restrict else None,
            min_score=min_score,
        )
        assert eng.recall(query) == oracle(eng, query)
    finally:
        eng.close()


# -- composition -----------------------------------------------------------------------
def test_create_engine_sqlite_memory_shares_connection(clock: FakeClock) -> None:
    eng = create_engine(MadeleineConfig(db_path=":memory:"), clock=clock, registry=fake_registry())
    m = eng.remember("x", context=SensoryContext(T, geohash=PARIS))
    fired = eng.sense(eng.build_cue(context=SensoryContext(T, geohash=PARIS)))
    assert [r.memory.id for r in fired] == [m.id]
    assert eng.queue.pending_count() == 1
    eng.close()
    eng.close()  # idempotent


def test_create_engine_file_and_default_registry_missing(tmp_path: Path, clock: FakeClock) -> None:
    db = tmp_path / "sub" / "x.db"
    reg = DictRegistry([text_channel(), geo_channel(), time_channel()])
    eng = create_engine(MadeleineConfig(db_path=str(db)), clock=clock, registry=reg)
    eng.remember("persisted")
    eng.close()
    assert db.exists()
    eng2 = create_engine(MadeleineConfig(db_path=str(db)), clock=clock, registry=reg)
    assert eng2.store.count() == 1
    eng2.close()


def test_create_engine_does_not_close_injected_store(clock: FakeClock) -> None:
    store = InMemoryStore()
    eng = create_engine(
        MadeleineConfig(),
        clock=clock,
        registry=fake_registry(),
        store=store,
        queue=InMemoryRecollectionQueue(),
    )
    eng.close()
    assert store.count() == 0  # still open


def test_forget_removes_everything(make_engine: EngineFactory) -> None:
    eng = make_engine()
    m = eng.remember("x", context=SensoryContext(T, geohash=PARIS))
    eng.sense(eng.build_cue(context=SensoryContext(T, geohash=PARIS)))
    assert eng.queue.pending_count() == 1
    assert eng.forget(m.id)
    assert eng.queue.pending_count() == 0
    assert eng.store.count() == 0
    assert eng.store.last_triggered(m.id) is None
    assert not eng.forget(m.id)
