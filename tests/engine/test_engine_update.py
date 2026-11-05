"""get_memory / update_memory (ADR 0003) on both stores."""

from __future__ import annotations

from pathlib import Path

import pytest

from madeleine.core.types import (
    TEXT,
    MemoryNotFoundError,
    ObservationError,
    RawSignal,
    RecallQuery,
    SensoryContext,
)
from madeleine.testing import FakeClock

from .conftest import EngineFactory, build_engine


def test_get_memory(make_engine: EngineFactory, clock: FakeClock) -> None:
    eng = make_engine()
    mem = eng.remember("tea at the tower", context=SensoryContext(clock.now(), geohash="u09tunq"))
    assert eng.get_memory(mem.id) == mem
    with pytest.raises(MemoryNotFoundError):
        eng.get_memory("mem_missing")


def test_update_annotations_keeps_sensory_context(
    make_engine: EngineFactory, clock: FakeClock
) -> None:
    eng = make_engine()
    mem = eng.remember(
        "tea at the tower",
        context=SensoryContext(clock.now(), geohash="u09tunq", tags=frozenset({"a"})),
        metadata={"k": "v"},
    )
    new = eng.update_memory(
        mem.id, tags=frozenset({"paris", " "}), salience=0.9, metadata={"x": "y"}, media_ref="m"
    )
    assert new.context.tags == frozenset({"paris"})
    assert new.context.salience == 0.9
    assert dict(new.metadata) == {"x": "y"}
    assert new.media_ref == "m"
    # sensory fingerprint is unchanged
    assert new.context.timestamp == mem.context.timestamp
    assert new.context.geohash == mem.context.geohash
    assert new.fingerprint.embeddings == mem.fingerprint.embeddings
    assert eng.get_memory(mem.id) == new
    # None leaves fields unchanged
    same = eng.update_memory(mem.id)
    assert same == new


def test_content_edit_re_embeds_derived_text(make_engine: EngineFactory, clock: FakeClock) -> None:
    eng = make_engine()
    mem = eng.remember("tea at the tower", context=SensoryContext(clock.now()))
    eng.remember("standup about the deploy pipeline", context=SensoryContext(clock.now()))
    new = eng.update_memory(mem.id, content="espresso on the riverbank")
    assert new.fingerprint.observations[TEXT].text == "espresso on the riverbank"
    assert new.fingerprint.embeddings[TEXT] != mem.fingerprint.embeddings[TEXT]
    cue = eng.build_cue([RawSignal(TEXT, "espresso riverbank")], include_time=False)
    assert eng.recall(RecallQuery(cue, k=1))[0].memory.id == mem.id
    cue_old = eng.build_cue([RawSignal(TEXT, "tea tower")], include_time=False)
    assert all(h.memory.id != mem.id or h.score < 0.5 for h in eng.recall(RecallQuery(cue_old)))


def test_content_edit_keeps_explicitly_sensed_text(
    make_engine: EngineFactory, clock: FakeClock
) -> None:
    eng = make_engine()
    mem = eng.remember(
        "note about the cafe",
        [RawSignal(TEXT, "song playing: la vie en rose")],
        context=SensoryContext(clock.now()),
    )
    new = eng.update_memory(mem.id, content="corrected note about the cafe")
    assert new.content == "corrected note about the cafe"
    assert new.fingerprint.observations[TEXT].text == "song playing: la vie en rose"
    assert new.fingerprint.embeddings[TEXT] == mem.fingerprint.embeddings[TEXT]


def test_update_validation(make_engine: EngineFactory, clock: FakeClock) -> None:
    eng = make_engine()
    mem = eng.remember("x", context=SensoryContext(clock.now()))
    with pytest.raises(ObservationError, match="empty"):
        eng.update_memory(mem.id, content="  ")
    with pytest.raises(ObservationError, match="salience"):
        eng.update_memory(mem.id, salience=1.5)
    with pytest.raises(MemoryNotFoundError):
        eng.update_memory("mem_missing", content="y")


def test_update_persists_across_reopen(tmp_path: Path, clock: FakeClock) -> None:
    db = str(tmp_path / "u.db")
    eng = build_engine("sqlite", clock, db_path=db)
    mem = eng.remember("tea at the tower", context=SensoryContext(clock.now()))
    eng.update_memory(mem.id, content="espresso on the riverbank", salience=0.8)
    eng.close()
    eng2 = build_engine("sqlite", clock, db_path=db)
    try:
        got = eng2.get_memory(mem.id)
        assert got.content == "espresso on the riverbank"
        assert got.context.salience == 0.8
        cue = eng2.build_cue([RawSignal(TEXT, "espresso riverbank")], include_time=False)
        assert eng2.recall(RecallQuery(cue, k=1))[0].memory.id == mem.id
    finally:
        eng2.close()
