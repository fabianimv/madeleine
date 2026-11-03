from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from madeleine.core.types import (
    Cue,
    RecallConfig,
    RecallMode,
    Recollection,
    SensoryContext,
)
from madeleine.engine import Madeleine
from madeleine.testing import FakeClock

from .conftest import EngineFactory, build_engine

PARIS = "u09tunq"
KYOTO = "xn0m7rz"
NYC = "dr5ru7v"
T = datetime(2025, 6, 1, 21, 0, tzinfo=UTC)


def _geo_cue(eng: Madeleine, when: datetime, geohash: str = PARIS, source: str = "test") -> Cue:
    return eng.build_cue(context=SensoryContext(when, geohash=geohash), source=source)


@settings(max_examples=30, deadline=None)
@given(hour=st.integers(0, 23), minute=st.integers(0, 59), cue_minute=st.integers(0, 59))
def test_time_only_match_never_fires(hour: int, minute: int, cue_minute: int) -> None:
    eng = build_engine("memory", FakeClock(T), recall=RecallConfig(trigger_threshold=0.0))
    try:
        mem_at = T.replace(hour=hour, minute=minute)
        eng.remember("at this hour", context=SensoryContext(mem_at), embed_content=False)
        eng.remember("with text", context=SensoryContext(mem_at))
        cue = eng.build_cue(context=SensoryContext(mem_at.replace(minute=cue_minute)))
        assert cue.channels == {"time"}
        assert eng.sense(cue) == []
        assert eng.queue.pending_count() == 0
    finally:
        eng.close()


def test_min_shared_channels(make_engine: EngineFactory) -> None:
    eng = make_engine(recall=RecallConfig(min_shared_channels=2))
    eng.remember("geo only", context=SensoryContext(T, geohash=PARIS), embed_content=False)
    eng.sense(_geo_cue(eng, T))  # geo + time shared -> 2 channels
    assert eng.queue.pending_count() == 1
    cue = eng.build_cue(
        context=SensoryContext(T + timedelta(days=1), geohash=PARIS), include_time=False
    )
    assert eng.sense(cue) == []


@settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    kind=st.sampled_from(["memory", "sqlite"]),
    period=st.timedeltas(min_value=timedelta(0), max_value=timedelta(days=2)),
    delta=st.timedeltas(min_value=-timedelta(days=4), max_value=timedelta(days=4)),
)
def test_refractory_suppresses_then_reallows(
    kind: str, period: timedelta, delta: timedelta
) -> None:
    eng = build_engine(kind, FakeClock(T), recall=RecallConfig(refractory_period=period))
    try:
        mem = eng.remember("cafe", context=SensoryContext(T - timedelta(days=3), geohash=PARIS))
        first = eng.sense(_geo_cue(eng, T))
        assert [r.memory.id for r in first] == [mem.id]
        again = eng.sense(_geo_cue(eng, T + delta))
        refractory = period > timedelta(0) and abs(delta) < period
        assert len(again) == (0 if refractory else 1)
        assert eng.store.last_triggered(mem.id) == (T if refractory else max(T, T + delta))
    finally:
        eng.close()


def test_refractory_measured_from_latest_fire(make_engine: EngineFactory) -> None:
    eng = make_engine(recall=RecallConfig(refractory_period=timedelta(hours=6)))
    eng.remember("cafe", context=SensoryContext(T - timedelta(days=1), geohash=PARIS))
    assert len(eng.sense(_geo_cue(eng, T))) == 1
    for h in (1, 3, 5):
        assert eng.sense(_geo_cue(eng, T + timedelta(hours=h))) == []
    assert eng.sense(_geo_cue(eng, T - timedelta(minutes=10))) == []  # late webhook
    assert len(eng.sense(_geo_cue(eng, T + timedelta(hours=6)))) == 1
    assert eng.sense(_geo_cue(eng, T + timedelta(hours=11))) == []
    assert len(eng.sense(_geo_cue(eng, T - timedelta(days=30)))) == 1  # far past is not


def test_max_push_per_cue(make_engine: EngineFactory) -> None:
    eng = make_engine(recall=RecallConfig(max_push_per_cue=3))
    mems = [
        eng.remember(
            f"memory {i}", context=SensoryContext(T - timedelta(days=30 * i), geohash=PARIS)
        )
        for i in range(5)
    ]
    fired = eng.sense(_geo_cue(eng, T))
    assert len(fired) == 3
    # newest memories decay least, so they score highest
    assert [r.memory.id for r in fired] == [m.id for m in mems[:3]]
    assert [r.score for r in fired] == sorted((r.score for r in fired), reverse=True)
    # the two left over fire on the next cue (the first three are refractory)
    nxt = eng.sense(_geo_cue(eng, T + timedelta(minutes=1)))
    assert {r.memory.id for r in nxt} == {m.id for m in mems[3:]}


def test_recollection_fields_and_listener(make_engine: EngineFactory) -> None:
    eng = make_engine()
    seen: list[Recollection] = []
    eng.queue.subscribe(seen.append)
    mem = eng.remember("x", context=SensoryContext(T, geohash=PARIS))
    cue = _geo_cue(eng, T + timedelta(hours=1), source="webhook:home_assistant")
    fired = eng.sense(cue)
    assert seen == fired
    (r,) = fired
    assert r.mode is RecallMode.PUSH
    assert r.cue_id == cue.id
    assert r.cue_source == "webhook:home_assistant"
    assert r.cue_channels == {"geo", "time"}
    assert r.triggered_at == cue.context.timestamp
    assert r.memory == mem
    assert r.id.startswith("rec_")
    assert eng.pending_recollections(drain=False) == fired
    assert eng.pending_recollections() == fired
    assert eng.pending_recollections() == []


def _hammer(engines: list[Madeleine], when: datetime, threads: int = 16) -> list[Recollection]:
    fired: list[Recollection] = []
    lock = threading.Lock()
    barrier = threading.Barrier(threads)

    def worker(i: int) -> None:
        eng = engines[i % len(engines)]
        cue = _geo_cue(eng, when)
        barrier.wait()
        for _ in range(3):
            got = eng.sense(cue)
            with lock:
                fired.extend(got)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return fired


def test_concurrent_sense_never_double_fires(make_engine: EngineFactory) -> None:
    eng = make_engine()
    mems = [
        eng.remember(f"m{i}", context=SensoryContext(T - timedelta(days=i), geohash=PARIS))
        for i in range(3)
    ]
    fired = _hammer([eng], T)
    assert sorted(r.memory.id for r in fired) == sorted(m.id for m in mems)
    assert eng.queue.pending_count() == 3


def test_concurrent_sense_across_engines_sharing_a_file(tmp_path: Path) -> None:
    db = str(tmp_path / "shared.db")
    clock = FakeClock(T)
    engines = [build_engine("sqlite", clock, db_path=db) for _ in range(2)]
    mem = engines[0].remember("shared cafe", context=SensoryContext(T, geohash=PARIS))
    fired = _hammer(engines, T + timedelta(minutes=5))
    assert [r.memory.id for r in fired] == [mem.id]
    assert engines[1].pending_recollections() == fired
    for e in engines:
        e.close()


def test_end_to_end_demo(tmp_path: Path) -> None:
    """Three memories with different fingerprints; a cue at the same place and a similar hour
    queues exactly the right one; a repeat within the refractory period queues nothing."""
    cest = timezone(timedelta(hours=2))
    clock = FakeClock(datetime(2026, 6, 1, 21, 40, tzinfo=cest))
    db = str(tmp_path / "demo.db")
    eng = build_engine("sqlite", clock, db_path=db)
    proposal = eng.remember(
        "Ana proposed at the Eiffel Tower",
        context=SensoryContext(
            datetime(2024, 6, 1, 21, 30, tzinfo=cest), geohash=PARIS, salience=0.95
        ),
    )
    eng.remember(
        "Rainy morning walk to Fushimi Inari",
        context=SensoryContext(
            datetime(2025, 3, 10, 7, 15, tzinfo=timezone(timedelta(hours=9))), geohash=KYOTO
        ),
    )
    eng.remember(
        "Lunch meeting about the Q3 budget",
        context=SensoryContext(
            datetime(2026, 2, 3, 12, 30, tzinfo=timezone(timedelta(hours=-5))), geohash=NYC
        ),
    )
    cue = eng.build_cue(context=SensoryContext(clock.now(), geohash=PARIS), source="demo")
    fired = eng.sense(cue)
    assert [r.memory.id for r in fired] == [proposal.id]
    assert fired[0].score >= eng.config.recall.trigger_threshold
    assert set(fired[0].match.channel_scores) == {"geo", "time"}
    eng.close()

    # another process (e.g. the Claude Code hook) drains the same file
    hook = build_engine("sqlite", clock, db_path=db)
    pending = hook.pending_recollections()
    assert [r.memory.id for r in pending] == [proposal.id]
    assert pending[0].to_prompt_line().startswith("[madeleine ")
    assert "Ana proposed at the Eiffel Tower" in pending[0].to_prompt_line()

    clock.advance(timedelta(minutes=20))
    again = hook.build_cue(context=SensoryContext(clock.now(), geohash=PARIS), source="demo")
    assert hook.sense(again) == []
    assert hook.pending_recollections() == []
    hook.close()
