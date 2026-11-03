"""The ``biometric`` channel: bodily state from wearables.

Features and fixed reference ranges (values are clamped into the range, then scaled to
``[0, 1]``):

=====================  ==========  ===========
feature                unit        range
=====================  ==========  ===========
``heart_rate``         bpm         40 - 180
``hrv``                ms (RMSSD)  0 - 200
``skin_temp``          deg C       28 - 38
``eda``                uS          0 - 20
``steps_per_min``      steps/min   0 - 200
``respiration_rate``   breaths/m   6 - 40
=====================  ==========  ===========

Vector (dim 12, *not* L2-normalised): for each feature in the order above,
``[value_norm * present, present]`` with ``present`` in ``{0, 1}``.

Similarity: a Gaussian kernel ``exp(-(d/sigma)^2 / 2)`` (``sigma = 0.15`` in normalised
units, e.g. ~21 bpm of heart rate) averaged over the features present in **both** vectors,
so a missing feature never penalises a match. No feature in common -> ``0.0``. The empty
vector (no feature present) is **not a valid embedding**: the adapter and embedder reject
empty readings, and ``sim(empty, anything)`` is defined as ``0.0`` so it can never cause
a match.

Context only by default: ``solo_trigger=False``, ``default_weight=0.5``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

from madeleine.core.types import (
    BIOMETRIC,
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
    "BIOMETRIC_DIM",
    "BIOMETRIC_FEATURES",
    "BIOMETRIC_RANGES",
    "BIOMETRIC_SPEC",
    "DEFAULT_BIOMETRIC_SIGMA",
    "BiometricAdapter",
    "BiometricEncoder",
    "biometric_similarity",
    "make_biometric_channel",
    "make_biometric_similarity",
]

BIOMETRIC_RANGES: Final[Mapping[str, tuple[float, float]]] = {
    "heart_rate": (40.0, 180.0),
    "hrv": (0.0, 200.0),
    "skin_temp": (28.0, 38.0),
    "eda": (0.0, 20.0),
    "steps_per_min": (0.0, 200.0),
    "respiration_rate": (6.0, 40.0),
}
BIOMETRIC_FEATURES: Final = tuple(BIOMETRIC_RANGES)
BIOMETRIC_DIM: Final = 2 * len(BIOMETRIC_FEATURES)
DEFAULT_BIOMETRIC_SIGMA: Final = 0.15

BIOMETRIC_SPEC: Final = ChannelSpec(
    name=BIOMETRIC,
    dim=BIOMETRIC_DIM,
    default_weight=0.5,
    solo_trigger=False,
    metric="mean Gaussian kernel (sigma 0.15) over features present in both",
    model_id=None,
    extra=None,
)


def _validated_numbers(raw: Mapping[Any, object]) -> dict[str, float]:
    if not raw:
        raise ObservationError(
            f"biometric reading is empty; give at least one of {list(BIOMETRIC_FEATURES)}"
        )
    out: dict[str, float] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if name not in BIOMETRIC_RANGES:
            raise ObservationError(
                f"unknown biometric feature {name!r}; allowed: {', '.join(BIOMETRIC_FEATURES)}"
            )
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ObservationError(f"biometric {name!r} must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise ObservationError(f"biometric {name!r} must be finite")
        out[name] = number
    return out


class BiometricAdapter:
    """``{"heart_rate": 92, "eda": 3.1, ...}`` -> ``Observation(numbers=...)``.

    Unknown feature names raise :class:`ObservationError` listing the allowed names.
    """

    @property
    def channel(self) -> ChannelName:
        return BIOMETRIC

    def normalize(self, signal: RawSignal) -> Observation:
        if signal.channel != BIOMETRIC:
            raise ObservationError(f"BiometricAdapter got a signal for {signal.channel!r}")
        if not isinstance(signal.value, Mapping):
            raise ObservationError(
                "biometric signal must be a mapping of feature -> number, e.g. {'heart_rate': 92}"
            )
        return Observation(BIOMETRIC, numbers=_validated_numbers(signal.value))


def normalise_feature(name: str, value: float) -> float:
    lo, hi = BIOMETRIC_RANGES[name]
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


class BiometricEncoder:
    """Fixed-range normalisation with a presence mask (dim 12). Deterministic, no model."""

    @property
    def channel(self) -> ChannelName:
        return BIOMETRIC

    @property
    def dim(self) -> int:
        return BIOMETRIC_DIM

    @property
    def model_id(self) -> str | None:
        return None

    def embed(self, observation: Observation) -> Vector:
        if observation.channel != BIOMETRIC:
            raise ObservationError(
                f"BiometricEncoder got an observation for {observation.channel!r}"
            )
        numbers = _validated_numbers(dict(observation.numbers))
        out: list[float] = []
        for name in BIOMETRIC_FEATURES:
            if name in numbers:
                out.extend((normalise_feature(name, numbers[name]), 1.0))
            else:
                out.extend((0.0, 0.0))
        return tuple(out)

    def embed_many(self, observations: Sequence[Observation]) -> list[Vector]:
        return [self.embed(o) for o in observations]


def make_biometric_similarity(sigma: float = DEFAULT_BIOMETRIC_SIGMA) -> SimilarityFn:
    """Mean Gaussian kernel over features present in both vectors (0.0 if none)."""
    if not sigma > 0 or not math.isfinite(sigma):
        raise ValueError("sigma must be a positive finite number")

    def similarity(a: Vector, b: Vector) -> float:
        if len(a) != BIOMETRIC_DIM or len(b) != BIOMETRIC_DIM:
            raise ValueError(f"biometric vectors must have length {BIOMETRIC_DIM}")
        total = 0.0
        shared = 0
        for i in range(len(BIOMETRIC_FEATURES)):
            if a[2 * i + 1] > 0.5 and b[2 * i + 1] > 0.5:
                d = (a[2 * i] - b[2 * i]) / sigma
                total += math.exp(-0.5 * d * d)
                shared += 1
        return max(0.0, min(1.0, total / shared)) if shared else 0.0

    similarity.__name__ = "biometric_similarity"
    return similarity


biometric_similarity: Final = make_biometric_similarity()
"""Similarity with the default ``sigma = 0.15``."""


def make_biometric_channel(sigma: float = DEFAULT_BIOMETRIC_SIGMA) -> Channel:
    """The ``biometric`` channel; ``sigma`` is the kernel width in normalised units."""
    return Channel(
        spec=BIOMETRIC_SPEC,
        adapter=BiometricAdapter(),
        embedder=BiometricEncoder(),
        similarity=biometric_similarity
        if sigma == DEFAULT_BIOMETRIC_SIGMA
        else make_biometric_similarity(sigma),
    )
