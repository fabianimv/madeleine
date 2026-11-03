from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta, timezone

import pytest
from hypothesis import given
from hypothesis import strategies as st

from madeleine.core import geohash as gh
from madeleine.core.serde import context_from_fields, parse_timestamp, signal_from_json
from madeleine.core.types import ObservationError

NOW = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)


def test_known_geohash() -> None:
    assert gh.encode(48.8584, 2.2945, 7) == "u09tunq"  # Eiffel Tower
    lat, lon = gh.decode("u09tunq")
    assert gh.haversine_km((lat, lon), (48.8584, 2.2945)) < 0.2


@given(st.floats(-89.9, 89.9), st.floats(-179.9, 179.9), st.integers(1, 12))
def test_round_trip_within_cell(lat: float, lon: float, precision: int) -> None:
    code = gh.encode(lat, lon, precision)
    lat_lo, lat_hi, lon_lo, lon_hi = gh.bounds(code)
    assert lat_lo <= lat <= lat_hi
    assert lon_lo <= lon <= lon_hi
    assert gh.is_valid(code)


def test_parse_timestamp() -> None:
    assert parse_timestamp(None, default=NOW) == NOW
    assert parse_timestamp("2026-06-01T21:30:00Z", default=NOW).tzinfo is not None
    naive = parse_timestamp("2026-06-01T21:30:00", default=NOW)
    assert naive.utcoffset() == timedelta(0)
    off = parse_timestamp("2026-06-01T21:30:00+02:00", default=NOW)
    assert off.utcoffset() == timezone(timedelta(hours=2)).utcoffset(None)
    with pytest.raises(ObservationError):
        parse_timestamp("yesterday", default=NOW)


def test_context_from_fields() -> None:
    ctx = context_from_fields(now=NOW, lat=48.8584, lon=2.2945, tags=["paris", " "])
    assert ctx.geohash is not None
    assert ctx.geohash.startswith("u09tunq")
    assert ctx.tags == frozenset({"paris"})
    assert context_from_fields(now=NOW, geohash="U09TUNQ", lat=0, lon=0).geohash == "u09tunq"
    with pytest.raises(ObservationError, match="together"):
        context_from_fields(now=NOW, lat=1.0)
    with pytest.raises(ObservationError, match="geohash"):
        context_from_fields(now=NOW, geohash="aaa")
    with pytest.raises(ObservationError, match="salience"):
        context_from_fields(now=NOW, salience=3)


def test_signal_from_json() -> None:
    s = signal_from_json({"channel": "geo", "value": [48.8, 2.29]})
    assert s.value == (48.8, 2.29)
    img = signal_from_json(
        {"channel": "image", "data_b64": base64.b64encode(b"abc").decode(), "mime_type": "x/y"}
    )
    assert img.value == b"abc"
    with pytest.raises(ObservationError):
        signal_from_json({"value": "x"})
    with pytest.raises(ObservationError):
        signal_from_json({"channel": "image", "data_b64": "!!!"})
    with pytest.raises(ObservationError):
        signal_from_json({"channel": "Bad Name", "value": 1})
