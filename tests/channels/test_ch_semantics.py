"""Semantic expectations of the deterministic channels and the hash text embedder."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone

import pytest

from madeleine.channels.biometric import make_biometric_channel
from madeleine.channels.geo import geo_distance_km, make_geo_channel
from madeleine.channels.hashing import HashEmbedder
from madeleine.channels.similarity import cosine01
from madeleine.channels.time import make_time_channel
from madeleine.core import geohash as gh
from madeleine.core.types import BIOMETRIC, GEO, TEXT, TIME, Channel, Observation, RawSignal

TZ2 = timezone(timedelta(hours=2))


def _tsim(ch: Channel, a: datetime, b: datetime) -> float:
    va = ch.embedder.embed(Observation(TIME, timestamp=a))
    vb = ch.embedder.embed(Observation(TIME, timestamp=b))
    return ch.similarity(va, vb)


def test_time_hour_of_day() -> None:
    ch = make_time_channel()
    t21 = datetime(2026, 6, 2, 21, 0, tzinfo=TZ2)  # a Tuesday
    assert _tsim(ch, t21, t21) == pytest.approx(1.0)
    assert _tsim(ch, t21, t21.replace(minute=30)) > 0.99
    assert _tsim(ch, t21, t21.replace(hour=9)) < 0.45
    # Same hour a week later is still very close (season moves a little).
    assert _tsim(ch, t21, t21 + timedelta(days=7)) > 0.99
    # Same hour, other side of the year: hour and weekday match, season does not.
    six_months = _tsim(ch, t21, datetime(2026, 12, 1, 21, 0, tzinfo=TZ2))
    assert 0.7 < six_months < 0.8


def test_time_uses_local_offset() -> None:
    ch = make_time_channel()
    local = datetime(2026, 6, 2, 21, 0, tzinfo=TZ2)
    same_instant_utc = local.astimezone(UTC)  # 19:00 UTC
    other_city_same_local = datetime(2026, 6, 2, 21, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert _tsim(ch, local, other_city_same_local) > 0.99
    assert _tsim(ch, local, same_instant_utc) < _tsim(ch, local, other_city_same_local) - 0.03


def test_time_vector_is_unit_and_l2_matches_similarity() -> None:
    ch = make_time_channel()
    a = ch.embedder.embed(Observation(TIME, timestamp=datetime(2026, 3, 1, 8, tzinfo=UTC)))
    b = ch.embedder.embed(Observation(TIME, timestamp=datetime(2026, 8, 5, 17, tzinfo=UTC)))
    assert math.fsum(x * x for x in a) == pytest.approx(1.0)
    l2sq = math.fsum((x - y) ** 2 for x, y in zip(a, b, strict=True))
    assert ch.similarity(a, b) == pytest.approx(1 - l2sq / 4)


def test_time_custom_weights() -> None:
    ch = make_time_channel({"hour": 1.0, "weekday": 0.0, "season": 0.0})
    t = datetime(2026, 6, 2, 21, 0, tzinfo=UTC)
    assert _tsim(ch, t, t + timedelta(days=100)) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown time cycles"):
        make_time_channel({"minute": 1.0})


def _gvec(ch: Channel, value: object) -> tuple[float, ...]:
    return ch.embedder.embed(ch.adapter.normalize(RawSignal(GEO, value)))


def test_geo_distances() -> None:
    ch = make_geo_channel()
    here = (48.8584, 2.2945)
    assert ch.similarity(_gvec(ch, "u09tunq"), _gvec(ch, "u09tunq")) == pytest.approx(1.0)
    dlat = 0.5 / (math.pi * gh.EARTH_RADIUS_KM / 180)  # 500 m north
    a, b = _gvec(ch, here), _gvec(ch, (here[0] + dlat, here[1]))
    assert geo_distance_km(a, b) == pytest.approx(0.5, rel=1e-3)
    assert ch.similarity(a, b) == pytest.approx(math.exp(-0.5), rel=1e-3)
    far = _gvec(ch, (here[0] + 20 * dlat, here[1]))  # 10 km
    assert ch.similarity(a, far) < 1e-4
    # Same geohash cell as text vs. exact coordinates inside it: very close.
    assert ch.similarity(_gvec(ch, gh.encode(*here, 9)), a) > 0.99


def test_geo_scale_and_float32_roundtrip() -> None:
    ch = make_geo_channel(scale_km=10.0)
    assert ch.spec.metric == "exp(-great_circle_km / 10)"
    a = _gvec(ch, (40.0, -3.7))
    import struct

    a32 = struct.unpack("3f", struct.pack("3f", *a))
    assert ch.similarity(a, a32) > 0.9999


def test_hash_text_semantics() -> None:
    emb = HashEmbedder(TEXT)

    def v(text: str) -> tuple[float, ...]:
        return emb.embed(Observation(TEXT, text=text))

    rain = v("rain on a tin roof")
    assert cosine01(rain, v("tin roof in the rain")) > cosine01(rain, v("stock market report"))
    assert cosine01(rain, v("stock market report")) < 0.2


def _bvec(ch: Channel, reading: dict[str, float]) -> tuple[float, ...]:
    return ch.embedder.embed(ch.adapter.normalize(RawSignal(BIOMETRIC, reading)))


def test_biometric_missing_features_do_not_penalise() -> None:
    ch = make_biometric_channel()
    cue = _bvec(ch, {"heart_rate": 120})
    memory = _bvec(ch, {"heart_rate": 120, "eda": 15.0, "skin_temp": 30.0})
    assert ch.similarity(cue, memory) == pytest.approx(1.0)
    calm = _bvec(ch, {"heart_rate": 60})
    assert ch.similarity(cue, calm) < 0.1
    near = _bvec(ch, {"heart_rate": 125})
    assert ch.similarity(cue, near) > 0.9
    disjoint = _bvec(ch, {"steps_per_min": 100})
    assert ch.similarity(cue, disjoint) == 0.0
    empty = (0.0,) * ch.spec.dim
    assert ch.similarity(empty, empty) == 0.0
    assert ch.similarity(cue, empty) == 0.0
