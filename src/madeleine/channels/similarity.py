"""Similarity helpers shared by channels.

Every :data:`~madeleine.core.types.SimilarityFn` must return a value in ``[0, 1]``, be
symmetric and give ``sim(v, v) == 1`` (within 1e-6) for any valid vector. The helpers here
are pure Python (no numpy) so importing them is free and results are bit-for-bit symmetric.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Iterable, Sequence

from madeleine.core.types import SimilarityFn, Vector

__all__ = [
    "cosine",
    "cosine01",
    "describe_similarity",
    "dot",
    "l2_norm",
    "l2_normalize",
    "rescaled_cosine",
]


def _check_same_length(a: Sequence[float], b: Sequence[float]) -> None:
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} != {len(b)}")


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product of two equal-length vectors."""
    _check_same_length(a, b)
    return math.fsum(map(operator.mul, a, b))


def l2_norm(v: Sequence[float]) -> float:
    """Euclidean norm."""
    return math.sqrt(math.fsum(x * x for x in v))


def l2_normalize(values: Iterable[float]) -> Vector:
    """Return ``values`` scaled to unit L2 norm as a :data:`Vector` of Python floats.

    A zero (or all-``nan``) input is returned as zeros rather than dividing by zero;
    embedders must avoid producing it (a zero vector is not a valid cosine embedding).
    """
    vec = [float(x) for x in values]
    n = l2_norm(vec)
    if not math.isfinite(n) or n == 0.0:
        return tuple(0.0 for _ in vec)
    return tuple(x / n for x in vec)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in ``[-1, 1]``; ``0.0`` if either vector has zero norm."""
    _check_same_length(a, b)
    na = l2_norm(a)
    nb = l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    c = math.fsum(map(operator.mul, a, b)) / (na * nb)
    return max(-1.0, min(1.0, c))


def cosine01(a: Vector, b: Vector) -> float:
    """``max(0, cos(a, b))``: the default similarity of embedding channels.

    Zero vectors are safe (similarity ``0.0``) but are not valid embeddings.
    """
    return max(0.0, cosine(a, b))


def rescaled_cosine(lo: float, hi: float) -> SimilarityFn:
    """Return a similarity mapping cosine ``lo -> 0`` and ``hi -> 1`` linearly (clamped).

    For CLIP/CLAP-style spaces, where unrelated items still have a high cosine (often
    0.3-0.6), plain ``max(0, cos)`` would make everything look similar. ``hi`` must be
    ``<= 1`` so that ``sim(v, v) == 1``.
    """
    if not -1.0 <= lo < hi <= 1.0:
        raise ValueError(f"need -1 <= lo < hi <= 1, got lo={lo}, hi={hi}")
    span = hi - lo

    def similarity(a: Vector, b: Vector) -> float:
        return max(0.0, min(1.0, (cosine(a, b) - lo) / span))

    similarity.__name__ = f"rescaled_cosine_{lo:g}_{hi:g}"
    similarity.__doc__ = f"cosine rescaled from [{lo:g}, {hi:g}] to [0, 1]"
    return similarity


def describe_similarity(fn: SimilarityFn) -> str:
    """Short human-readable description of a similarity function (for ``ChannelSpec.metric``)."""
    if fn is cosine01:
        return "max(0, cosine)"
    doc = (getattr(fn, "__doc__", None) or "").strip().splitlines()
    return doc[0] if doc else getattr(fn, "__name__", "custom similarity")
