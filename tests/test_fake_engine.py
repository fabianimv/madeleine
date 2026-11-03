from __future__ import annotations

from datetime import timedelta

from madeleine.core.types import RawSignal, RecallQuery, SensoryContext
from madeleine.testing import FakeClock, FakeEngine


def test_fake_engine_push_pull_refractory_forget(clock: FakeClock) -> None:
    eng = FakeEngine(clock=clock)
    seen = []
    eng.queue.subscribe(seen.append)
    paris = eng.remember(
        "Dinner at the tower", context=SensoryContext(clock.now(), geohash="u09tunq")
    )
    eng.remember("Rainy walk in Kyoto", context=SensoryContext(clock.now(), geohash="xn0m7"))

    cue = eng.build_cue([RawSignal("geo", "u09tunq")], source="test")
    fired = eng.sense(cue)
    assert [r.memory.id for r in fired] == [paris.id]
    assert seen == fired
    assert eng.sense(eng.build_cue([RawSignal("geo", "u09tunq")])) == []  # refractory
    clock.advance(timedelta(hours=2))
    assert len(eng.sense(eng.build_cue([RawSignal("geo", "u09tunq")]))) == 1

    hits = eng.recall(
        RecallQuery(eng.build_cue([RawSignal("text", "kyoto rain")], include_time=False))
    )
    assert hits[0].memory.content == "Rainy walk in Kyoto"

    assert len(eng.pending_recollections(drain=False)) == 2
    assert eng.forget(paris.id)
    assert eng.pending_recollections() == []
    assert not eng.forget(paris.id)
