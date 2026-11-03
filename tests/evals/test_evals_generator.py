from __future__ import annotations

import io
import struct
import wave
import zlib

import pytest

from madeleine.core import geohash as gh
from madeleine.core.types import AUDIO, IMAGE, TEXT
from madeleine.evals.generator import (
    ALL_CATEGORIES,
    CONTEXT_HOUR_WINDOW,
    NEGATIVE_CATEGORIES,
    OLD_MEMORY_MIN_AGE,
    POSITIVE_CATEGORIES,
    SyntheticDataset,
    circular_hour_diff,
    generate,
    local_hour,
)


@pytest.fixture(scope="module")
def ds() -> SyntheticDataset:
    return generate(memories=200, cues=180, seed=7)


def test_same_seed_is_identical() -> None:
    assert generate(memories=60, cues=40, seed=3) == generate(memories=60, cues=40, seed=3)


def test_different_seed_differs() -> None:
    a, b = generate(memories=60, cues=40, seed=1), generate(memories=60, cues=40, seed=2)
    assert [m.content for m in a.memories] != [m.content for m in b.memories]
    assert a.cues != b.cues


def test_every_category_present(ds: SyntheticDataset) -> None:
    counts = ds.category_counts()
    assert sum(counts.values()) == 180
    assert all(counts[c] > 0 for c in ALL_CATEGORIES)
    small = generate(memories=10, cues=len(ALL_CATEGORIES), seed=0)
    assert all(v == 1 for v in small.category_counts().values())


def test_category_mix_and_validation() -> None:
    only = generate(memories=20, cues=10, seed=0, category_mix={"context": 1.0})
    assert {c.category for c in only.cues} == {"context"}
    with pytest.raises(ValueError, match="unknown"):
        generate(memories=20, cues=5, category_mix={"nope": 1.0})
    with pytest.raises(ValueError, match="memories"):
        generate(memories=0)


def test_memories_are_well_formed(ds: SyntheticDataset) -> None:
    assert [m.index for m in ds.memories] == list(range(len(ds.memories)))
    for m in ds.memories:
        assert m.timestamp.utcoffset() is not None
        assert m.timestamp < ds.now
        assert (ds.now - m.timestamp).days <= 3 * 365
        ctx = m.context
        assert ctx.geohash == m.geohash
        assert gh.is_valid(ctx.geohash)
        assert 0 <= ctx.salience <= 1
        place = ds.places[m.place]
        assert gh.haversine_km((m.lat, m.lon), (place.lat, place.lon)) <= 0.07
        assert m.topic in m.tags
        assert place.kind in m.tags
        assert m.content.strip()
    assert any(m.image for m in ds.memories)
    assert any(m.audio for m in ds.memories)
    assert any(ds.now - m.timestamp >= OLD_MEMORY_MIN_AGE for m in ds.memories)


def test_places_are_distinct_clusters(ds: SyntheticDataset) -> None:
    for a in ds.places:
        for b in ds.places:
            if a.id < b.id:
                assert gh.haversine_km((a.lat, a.lon), (b.lat, b.lon)) > 1.0


def test_synthetic_media_is_valid() -> None:
    d = generate(memories=60, cues=0, seed=5)
    png = next(m.image for m in d.memories if m.image)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    (length,) = struct.unpack(">I", png[33:37])
    assert png[37:41] == b"IDAT"
    raw = zlib.decompress(png[41 : 41 + length])
    assert len(raw) == 16 * (1 + 16 * 3)
    wav = next(m.audio for m in d.memories if m.audio)
    with wave.open(io.BytesIO(wav)) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 8000
        assert w.getnframes() > 0
    sig = next(m for m in d.memories if m.image).signals[0]
    assert sig.channel == IMAGE
    assert sig.mime_type == "image/png"


def test_negatives_have_empty_relevant_sets(ds: SyntheticDataset) -> None:
    for c in ds.cues:
        if c.category in NEGATIVE_CATEGORIES:
            assert c.relevant == frozenset()
            assert not c.should_fire
            assert c.target is None
        else:
            assert c.category in POSITIVE_CATEGORIES
            assert c.should_fire
            assert c.target is not None
            assert c.target in c.relevant


def test_context_cue_ground_truth(ds: SyntheticDataset) -> None:
    cues = [c for c in ds.cues if c.category == "context"]
    assert cues
    for c in cues:
        assert c.signals == ()
        assert c.include_geo
        assert c.include_time
        assert c.context.geohash is not None
        here = gh.decode(c.context.geohash)
        hour = local_hour(c.context.timestamp)
        assert c.context.timestamp > ds.now
        for i in c.relevant:
            m = ds.memories[i]
            assert 0.0 < gh.haversine_km(here, (m.lat, m.lon)) <= 0.4
            assert circular_hour_diff(local_hour(m.timestamp), hour) <= CONTEXT_HOUR_WINDOW
        # Completeness: every memory at that place and hour is relevant.
        place = ds.memories[c.target or 0].place
        expected = {
            m.index
            for m in ds.memories
            if m.place == place
            and circular_hour_diff(local_hour(m.timestamp), hour) <= CONTEXT_HOUR_WINDOW
        }
        assert set(c.relevant) == expected


def test_cue_shapes(ds: SyntheticDataset) -> None:
    for c in ds.cues:
        channels = {s.channel for s in c.signals}
        target = ds.memories[c.target] if c.target is not None else None
        if c.category == "text_paraphrase":
            assert channels == {TEXT}
            assert not c.include_time
            assert c.context.geohash is None
            assert target is not None
            words = set(target.content.lower().replace(";", " ").replace(".", " ").split())
            cue_words = set(str(c.signals[0].value).lower().replace(",", " ").split())
            assert len(words & cue_words) >= 3
        elif c.category in ("exact_media", "perturbed_media"):
            assert target is not None
            assert len(c.signals) == 1
            sig = c.signals[0]
            assert sig.channel in (IMAGE, AUDIO)
            assert not c.include_time
            original = target.image if sig.channel == IMAGE else target.audio
            assert (sig.value == original) == (c.category == "exact_media")
        elif c.category in ("multi", "old_memory"):
            assert channels == {TEXT}
            assert c.context.geohash is not None
            assert target is not None
            assert gh.haversine_km(gh.decode(c.context.geohash), (target.lat, target.lon)) < 0.2
            hour_gap = circular_hour_diff(
                local_hour(c.context.timestamp), local_hour(target.timestamp)
            )
            assert hour_gap <= 0.5 + 1e-9
            if c.category == "old_memory":
                assert ds.now - target.timestamp >= OLD_MEMORY_MIN_AGE
        elif c.category == "time_only":
            assert c.signals == ()
            assert c.context.geohash is None
            assert c.include_time
        elif c.category == "distractor":
            assert c.context.geohash is not None
            far = gh.decode(c.context.geohash)
            assert all(gh.haversine_km(far, (m.lat, m.lon)) > 500 for m in ds.memories)
        elif c.category == "hard_negative":
            assert c.context.geohash is not None
            assert channels == {TEXT}
            here = gh.decode(c.context.geohash)
            near = [m for m in ds.memories if gh.haversine_km(here, (m.lat, m.lon)) < 0.3]
            assert near, "hard negatives are at a place with memories"
            text = str(c.signals[0].value).lower()
            assert not any(w in text for m in near for w in m.content.lower().split() if len(w) > 6)


def test_circular_hour_diff() -> None:
    assert circular_hour_diff(23.5, 0.5) == pytest.approx(1.0)
    assert circular_hour_diff(6, 18) == pytest.approx(12.0)
    assert circular_hour_diff(3, 3) == 0
