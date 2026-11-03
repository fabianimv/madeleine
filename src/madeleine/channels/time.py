"""The ``time`` channel: cyclical encoding of hour-of-day, weekday and season.

Encoding (dim 6), computed in the timestamp's **own** UTC offset (local time)::

    theta_hour    = 2*pi * (hour + minute/60 + second/3600) / 24
    theta_weekday = 2*pi * (weekday + day_fraction) / 7          # Monday = 0, continuous
    theta_season  = 2*pi * (day_of_year - 1 + day_fraction) / days_in_year
    vector = (sqrt(w_h)*sin, sqrt(w_h)*cos, sqrt(w_w)*sin, sqrt(w_w)*cos,
              sqrt(w_s)*sin, sqrt(w_s)*cos)

with weights ``w_h = 0.6``, ``w_w = 0.15``, ``w_s = 0.25`` (sum 1, so the vector is
unit-norm). The weekday angle is continuous so that Sunday 23:59 is close to Monday 00:01.

Similarity: for each cycle the angle difference is recovered from its (sin, cos) pair and
scored ``(1 + cos(delta))/2``; the result is the weighted mean of the three. Because each
pair is scaled by ``sqrt(w)``, this equals ``1 - |a - b|^2 / 4``, so L2 nearest-neighbour
search (sqlite-vec) ranks exactly like the similarity.

The channel is *context only*: ``solo_trigger=False`` and ``default_weight=0.5`` (every
memory made at 9 am must not fire at every 9 am).
"""

from __future__ import annotations

import calendar
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Final

from madeleine.core.serde import parse_timestamp
from madeleine.core.types import (
    TIME,
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
    "DEFAULT_TIME_WEIGHTS",
    "TIME_DIM",
    "TIME_SPEC",
    "TimeAdapter",
    "TimeEncoder",
    "make_time_channel",
    "make_time_similarity",
    "time_similarity",
]

TIME_DIM: Final = 6
DEFAULT_TIME_WEIGHTS: Final[Mapping[str, float]] = {"hour": 0.6, "weekday": 0.15, "season": 0.25}
_CYCLES: Final = ("hour", "weekday", "season")

TIME_SPEC: Final = ChannelSpec(
    name=TIME,
    dim=TIME_DIM,
    default_weight=0.5,
    solo_trigger=False,
    metric="weighted mean of (1+cos dθ)/2 over hour 0.6, weekday 0.15, season 0.25",
    model_id=None,
    extra=None,
)


def _normalized_weights(weights: Mapping[str, float] | None) -> tuple[float, float, float]:
    w = dict(DEFAULT_TIME_WEIGHTS)
    if weights is not None:
        unknown = set(weights) - set(_CYCLES)
        if unknown:
            raise ValueError(f"unknown time cycles {sorted(unknown)}; expected {list(_CYCLES)}")
        w.update(weights)
    values = tuple(float(w[c]) for c in _CYCLES)
    if any(v < 0 or not math.isfinite(v) for v in values) or sum(values) <= 0:
        raise ValueError("time weights must be finite, >= 0 and not all zero")
    total = sum(values)
    return values[0] / total, values[1] / total, values[2] / total


class TimeAdapter:
    """Accepts a ``datetime`` (naive = UTC), an ISO-8601 string, or epoch seconds.

    ``RawSignal("time", None, observed_at=dt)`` uses ``observed_at``. The timestamp keeps
    its own UTC offset: send local-offset timestamps so "evening" means local evening.
    """

    @property
    def channel(self) -> ChannelName:
        return TIME

    def normalize(self, signal: RawSignal) -> Observation:
        if signal.channel != TIME:
            raise ObservationError(f"TimeAdapter got a signal for {signal.channel!r}")
        value = signal.value
        if value is None:
            if signal.observed_at is None:
                raise ObservationError(
                    "time signal needs a datetime, an ISO-8601 string or epoch seconds"
                )
            return Observation(TIME, timestamp=signal.observed_at)
        if isinstance(value, bool):
            raise ObservationError("time signal must not be a boolean")
        if isinstance(value, int | float):
            if not math.isfinite(value):
                raise ObservationError("time signal epoch seconds must be finite")
            try:
                ts = datetime.fromtimestamp(float(value), tz=UTC)
            except (OverflowError, OSError, ValueError) as exc:
                raise ObservationError(f"time signal epoch {value!r} out of range") from exc
            return Observation(TIME, timestamp=ts)
        if isinstance(value, str) and not value.strip():
            raise ObservationError("time signal is an empty string")
        if isinstance(value, str | datetime):
            ts = parse_timestamp(value, default=datetime.now(UTC))
            return Observation(TIME, timestamp=ts)
        raise ObservationError(
            f"time signal must be a datetime, ISO-8601 string or epoch seconds, "
            f"got {type(value).__name__}"
        )


def time_angles(ts: datetime) -> tuple[float, float, float]:
    """``(theta_hour, theta_weekday, theta_season)`` in radians, in ``ts``'s own offset."""
    if ts.tzinfo is None or ts.utcoffset() is None:
        raise ObservationError("time observation needs a timezone-aware timestamp")
    day_fraction = (ts.hour + ts.minute / 60 + (ts.second + ts.microsecond / 1e6) / 3600) / 24
    days_in_year = 366 if calendar.isleap(ts.year) else 365
    yday = ts.timetuple().tm_yday
    tau = 2 * math.pi
    return (
        tau * day_fraction,
        tau * (ts.weekday() + day_fraction) / 7,
        tau * (yday - 1 + day_fraction) / days_in_year,
    )


class TimeEncoder:
    """Deterministic cyclical encoder (no model). Reads ``Observation.timestamp``."""

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        self._weights = _normalized_weights(weights)

    @property
    def channel(self) -> ChannelName:
        return TIME

    @property
    def dim(self) -> int:
        return TIME_DIM

    @property
    def model_id(self) -> str | None:
        return None

    @property
    def weights(self) -> tuple[float, float, float]:
        return self._weights

    def embed(self, observation: Observation) -> Vector:
        if observation.channel != TIME:
            raise ObservationError(f"TimeEncoder got an observation for {observation.channel!r}")
        if observation.timestamp is None:
            raise ObservationError("time observation needs a timestamp")
        out: list[float] = []
        for theta, w in zip(time_angles(observation.timestamp), self._weights, strict=True):
            r = math.sqrt(w)
            out.extend((r * math.sin(theta), r * math.cos(theta)))
        return tuple(out)

    def embed_many(self, observations: Sequence[Observation]) -> list[Vector]:
        return [self.embed(o) for o in observations]


def make_time_similarity(weights: Mapping[str, float] | None = None) -> SimilarityFn:
    """Weighted mean of per-cycle ``(1 + cos dθ)/2``; angles are decoded from each pair."""
    w = _normalized_weights(weights)

    def similarity(a: Vector, b: Vector) -> float:
        if len(a) != TIME_DIM or len(b) != TIME_DIM:
            raise ValueError(f"time vectors must have length {TIME_DIM}")
        num = 0.0
        den = 0.0
        for i, wi in enumerate(w):
            sa, ca, sb, cb = a[2 * i], a[2 * i + 1], b[2 * i], b[2 * i + 1]
            na = math.hypot(sa, ca)
            nb = math.hypot(sb, cb)
            if wi == 0.0 or na == 0.0 or nb == 0.0:
                continue
            cos_delta = max(-1.0, min(1.0, (sa * sb + ca * cb) / (na * nb)))
            num += wi * (1.0 + cos_delta) / 2.0
            den += wi
        return max(0.0, min(1.0, num / den)) if den > 0 else 0.0

    similarity.__name__ = "time_similarity"
    return similarity


time_similarity: Final = make_time_similarity()
"""Similarity with the default weights (hour 0.6, weekday 0.15, season 0.25)."""


def make_time_channel(weights: Mapping[str, float] | None = None) -> Channel:
    """The ``time`` channel. ``weights`` overrides ``{"hour", "weekday", "season"}``."""
    return Channel(
        spec=TIME_SPEC,
        adapter=TimeAdapter(),
        embedder=TimeEncoder(weights),
        similarity=make_time_similarity(weights) if weights is not None else time_similarity,
    )
