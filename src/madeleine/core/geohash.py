"""Tiny, dependency-free geohash encode/decode (public-domain algorithm by G. Niemeyer)."""

from __future__ import annotations

import math
from typing import Final

BASE32: Final = "0123456789bcdefghjkmnpqrstuvwxyz"
_DECODE: Final = {c: i for i, c in enumerate(BASE32)}
EARTH_RADIUS_KM: Final = 6371.0088


def encode(lat: float, lon: float, precision: int = 9) -> str:
    """Encode a WGS84 coordinate. ``precision`` 9 is ~4.8 m x 4.8 m."""
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise ValueError(f"coordinate out of range: ({lat}, {lon})")
    if not 1 <= precision <= 12:
        raise ValueError("precision must be in 1..12")
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    out: list[str] = []
    bits = 0
    nbits = 0
    even = True
    while len(out) < precision:
        if even:
            mid = (lon_lo + lon_hi) / 2
            if lon >= mid:
                bits = (bits << 1) | 1
                lon_lo = mid
            else:
                bits <<= 1
                lon_hi = mid
        else:
            mid = (lat_lo + lat_hi) / 2
            if lat >= mid:
                bits = (bits << 1) | 1
                lat_lo = mid
            else:
                bits <<= 1
                lat_hi = mid
        even = not even
        nbits += 1
        if nbits == 5:
            out.append(BASE32[bits])
            bits = 0
            nbits = 0
    return "".join(out)


def bounds(geohash: str) -> tuple[float, float, float, float]:
    """Return ``(lat_min, lat_max, lon_min, lon_max)`` of the cell."""
    lat_lo, lat_hi, lon_lo, lon_hi = -90.0, 90.0, -180.0, 180.0
    even = True
    for ch in geohash.lower():
        try:
            val = _DECODE[ch]
        except KeyError:
            raise ValueError(f"invalid geohash {geohash!r}") from None
        for shift in range(4, -1, -1):
            bit = (val >> shift) & 1
            if even:
                mid = (lon_lo + lon_hi) / 2
                if bit:
                    lon_lo = mid
                else:
                    lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2
                if bit:
                    lat_lo = mid
                else:
                    lat_hi = mid
            even = not even
    if not geohash:
        raise ValueError("empty geohash")
    return lat_lo, lat_hi, lon_lo, lon_hi


def decode(geohash: str) -> tuple[float, float]:
    """Return the ``(lat, lon)`` centre of the cell."""
    lat_lo, lat_hi, lon_lo, lon_hi = bounds(geohash)
    return (lat_lo + lat_hi) / 2, (lon_lo + lon_hi) / 2


def is_valid(geohash: str) -> bool:
    return 1 <= len(geohash) <= 12 and all(c in _DECODE for c in geohash.lower())


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two ``(lat, lon)`` points in kilometres."""
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))
