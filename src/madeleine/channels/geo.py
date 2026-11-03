"""The ``geo`` channel: place as a point on the unit sphere.

* **Adapter** - accepts a geohash string, a ``(lat, lon)`` tuple/list, or a mapping with
  ``"lat"`` and ``"lon"``. Produces ``Observation(text=geohash, numbers={"lat", "lon"})``:
  lat/lon inputs are encoded at precision 9 (~5 m) and keep their exact coordinates;
  geohash inputs keep their given precision (no ``numbers``).
* **Embedder** - exact lat/lon if present, else the geohash cell centre, mapped to
  ``(cos lat cos lon, cos lat sin lon, sin lat)`` (dim 3, unit norm).
* **Similarity** - ``exp(-d_km / scale_km)`` with ``d`` the great-circle distance recovered
  from the two xyz vectors (``2 R asin(|a - b| / 2)``); ``scale_km`` defaults to 1 km, so
  500 m apart scores ~0.61 and 10 km ~0.00005.

Chord length ``|a - b|`` is monotone with great-circle distance, so L2 nearest-neighbour
search over the stored vectors (sqlite-vec) is an exact pre-filter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final

from madeleine.core import geohash as gh
from madeleine.core.serde import DEFAULT_GEOHASH_PRECISION
from madeleine.core.types import (
    GEO,
    Channel,
    ChannelName,
    ChannelSpec,
    Observation,
    ObservationError,
    RawSignal,
    SimilarityFn,
    Vector,
)

__all__ = [
    "DEFAULT_GEO_SCALE_KM",
    "GEO_DIM",
    "GeoAdapter",
    "GeoEmbedder",
    "geo_distance_km",
    "geo_similarity",
    "geo_spec",
    "make_geo_channel",
    "make_geo_similarity",
]

GEO_DIM: Final = 3
DEFAULT_GEO_SCALE_KM: Final = 1.0


def geo_spec(scale_km: float = DEFAULT_GEO_SCALE_KM) -> ChannelSpec:
    return ChannelSpec(
        name=GEO,
        dim=GEO_DIM,
        default_weight=1.0,
        solo_trigger=True,
        metric=f"exp(-great_circle_km / {scale_km:g})",
        model_id=None,
        extra=None,
    )


def _coord(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ObservationError(f"geo {what} must be a number, got {type(value).__name__}")
    try:
        out = float(value)
    except ValueError as exc:
        raise ObservationError(f"geo {what} {value!r} is not a number") from exc
    if not math.isfinite(out):
        raise ObservationError(f"geo {what} must be finite")
    return out


def _from_lat_lon(lat_raw: object, lon_raw: object) -> Observation:
    lat, lon = _coord(lat_raw, "lat"), _coord(lon_raw, "lon")
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        raise ObservationError(f"geo coordinate out of range: ({lat}, {lon})")
    return Observation(
        GEO,
        text=gh.encode(lat, lon, DEFAULT_GEOHASH_PRECISION),
        numbers={"lat": lat, "lon": lon},
    )


class GeoAdapter:
    """Geohash string, ``(lat, lon)`` sequence or ``{"lat": .., "lon": ..}`` -> observation."""

    @property
    def channel(self) -> ChannelName:
        return GEO

    def normalize(self, signal: RawSignal) -> Observation:
        if signal.channel != GEO:
            raise ObservationError(f"GeoAdapter got a signal for {signal.channel!r}")
        value = signal.value
        if isinstance(value, str):
            text = value.strip().lower()
            if not gh.is_valid(text):
                raise ObservationError(f"invalid geohash {value!r}")
            return Observation(GEO, text=text)
        if isinstance(value, Mapping):
            if "lat" not in value or "lon" not in value:
                raise ObservationError("geo mapping needs both 'lat' and 'lon'")
            return _from_lat_lon(value["lat"], value["lon"])
        if isinstance(value, tuple | list):
            if len(value) != 2:
                raise ObservationError("geo (lat, lon) needs exactly two numbers")
            return _from_lat_lon(value[0], value[1])
        raise ObservationError(
            "geo signal must be a geohash string, a (lat, lon) pair or {'lat', 'lon'}, "
            f"got {type(value).__name__}"
        )


def _lat_lon_of(observation: Observation) -> tuple[float, float]:
    nums = observation.numbers
    if "lat" in nums and "lon" in nums:
        return float(nums["lat"]), float(nums["lon"])
    if observation.text and gh.is_valid(observation.text):
        return gh.decode(observation.text)
    raise ObservationError("geo observation needs a geohash (text) or numbers lat/lon")


class GeoEmbedder:
    """Lat/lon (or geohash cell centre) -> unit-sphere xyz. Deterministic, no model."""

    @property
    def channel(self) -> ChannelName:
        return GEO

    @property
    def dim(self) -> int:
        return GEO_DIM

    @property
    def model_id(self) -> str | None:
        return None

    def embed(self, observation: Observation) -> Vector:
        if observation.channel != GEO:
            raise ObservationError(f"GeoEmbedder got an observation for {observation.channel!r}")
        lat, lon = _lat_lon_of(observation)
        phi, lam = math.radians(lat), math.radians(lon)
        return (math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi))

    def embed_many(self, observations: Sequence[Observation]) -> list[Vector]:
        return [self.embed(o) for o in observations]


def geo_distance_km(a: Vector, b: Vector) -> float:
    """Great-circle distance (km) between two geo vectors (normalised defensively)."""
    if len(a) != GEO_DIM or len(b) != GEO_DIM:
        raise ValueError(f"geo vectors must have length {GEO_DIM}")
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return math.inf
    chord = math.sqrt(math.fsum((x / na - y / nb) ** 2 for x, y in zip(a, b, strict=True)))
    return 2.0 * gh.EARTH_RADIUS_KM * math.asin(min(1.0, chord / 2.0))


def make_geo_similarity(scale_km: float = DEFAULT_GEO_SCALE_KM) -> SimilarityFn:
    """``exp(-d_km / scale_km)``."""
    if not scale_km > 0 or not math.isfinite(scale_km):
        raise ValueError("scale_km must be a positive finite number")

    def similarity(a: Vector, b: Vector) -> float:
        d = geo_distance_km(a, b)
        return 0.0 if math.isinf(d) else max(0.0, min(1.0, math.exp(-d / scale_km)))

    similarity.__name__ = f"geo_similarity_{scale_km:g}km"
    return similarity


geo_similarity: Final = make_geo_similarity()
"""Similarity with the default 1 km scale."""


def make_geo_channel(scale_km: float = DEFAULT_GEO_SCALE_KM) -> Channel:
    """The ``geo`` channel; ``scale_km`` is the distance at which similarity is ``1/e``."""
    return Channel(
        spec=geo_spec(scale_km),
        adapter=GeoAdapter(),
        embedder=GeoEmbedder(),
        similarity=geo_similarity
        if scale_km == DEFAULT_GEO_SCALE_KM
        else make_geo_similarity(scale_km),
    )
