"""Pure metric functions for the benchmark (no engine, no I/O).

Conventions:

* Memories are referred to by their integer index in the synthetic dataset.
* A metric that is undefined for its input (no positive cues, nothing fired...) returns
  ``None`` rather than a misleading 0 or 1.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from madeleine.core.types import JSONValue

__all__ = [
    "RECALL_KS",
    "CategoryMetrics",
    "CueOutcome",
    "LatencyStats",
    "by_category",
    "false_trigger_rate",
    "latency_stats",
    "mean_recall_at_k",
    "percentile",
    "recall_at_k",
    "summarize",
    "trigger_hit_rate",
    "trigger_precision",
]

RECALL_KS: tuple[int, ...] = (1, 5, 10)


@dataclass(frozen=True, slots=True)
class CueOutcome:
    """What happened for one cue.

    * ``ranked`` - memory indices returned by PULL ``recall``, best first.
    * ``fired`` - memory indices that fired involuntarily (PUSH ``sense``).
    """

    category: str
    relevant: frozenset[int]
    should_fire: bool
    ranked: tuple[int, ...] = ()
    fired: tuple[int, ...] = ()


def recall_at_k(ranked: Sequence[int], relevant: frozenset[int], k: int) -> float | None:
    """Capped recall@k = ``|relevant ∩ top-k| / min(k, |relevant|)``.

    Capping by ``k`` keeps the metric in ``[0, 1]`` and reachable (a ``context`` cue may
    have 30 relevant memories; returning 10 of them in the top 10 is perfect).
    ``None`` when ``relevant`` is empty (undefined).
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if not relevant:
        return None
    hits = len(relevant.intersection(ranked[:k]))
    return hits / min(k, len(relevant))


def mean_recall_at_k(outcomes: Iterable[CueOutcome], k: int) -> float | None:
    """Mean capped recall@k over cues that have relevant memories (``None`` if none)."""
    values = [v for o in outcomes if (v := recall_at_k(o.ranked, o.relevant, k)) is not None]
    return sum(values) / len(values) if values else None


def trigger_precision(outcomes: Iterable[CueOutcome]) -> float | None:
    """Micro precision of involuntary triggers: relevant fires / all fires.

    Fires on negative cues count as irrelevant. ``None`` when nothing fired.
    """
    fired = relevant = 0
    for o in outcomes:
        fired += len(o.fired)
        relevant += sum(1 for i in o.fired if i in o.relevant)
    return relevant / fired if fired else None


def trigger_hit_rate(outcomes: Iterable[CueOutcome]) -> float | None:
    """Share of positive cues (``should_fire``) where at least one relevant memory fired."""
    positives = [o for o in outcomes if o.should_fire]
    if not positives:
        return None
    return sum(1 for o in positives if o.relevant.intersection(o.fired)) / len(positives)


def false_trigger_rate(outcomes: Iterable[CueOutcome]) -> float | None:
    """Share of negative cues (``not should_fire``) that fired anything at all."""
    negatives = [o for o in outcomes if not o.should_fire]
    if not negatives:
        return None
    return sum(1 for o in negatives if o.fired) / len(negatives)


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolated percentile (numpy's default method); ``None`` for no values."""
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be in [0, 100]")
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Latency summary for one operation, in milliseconds."""

    n: int
    p50_ms: float | None
    p95_ms: float | None
    mean_ms: float | None
    max_ms: float | None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "n": self.n,
            "p50_ms": _round(self.p50_ms),
            "p95_ms": _round(self.p95_ms),
            "mean_ms": _round(self.mean_ms),
            "max_ms": _round(self.max_ms),
        }


def latency_stats(values_ms: Sequence[float]) -> LatencyStats:
    if not values_ms:
        return LatencyStats(0, None, None, None, None)
    return LatencyStats(
        n=len(values_ms),
        p50_ms=percentile(values_ms, 50),
        p95_ms=percentile(values_ms, 95),
        mean_ms=sum(values_ms) / len(values_ms),
        max_ms=max(values_ms),
    )


@dataclass(frozen=True, slots=True)
class CategoryMetrics:
    """All metrics for one category (or ``"overall"``)."""

    category: str
    n_cues: int
    n_positive: int
    n_negative: int
    n_fired: int
    recall_at: Mapping[int, float | None] = field(default_factory=dict)
    precision: float | None = None
    hit_rate: float | None = None
    false_trigger_rate: float | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "category": self.category,
            "n_cues": self.n_cues,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_fired": self.n_fired,
            **{f"recall@{k}": _round(v) for k, v in sorted(self.recall_at.items())},
            "precision": _round(self.precision),
            "hit_rate": _round(self.hit_rate),
            "false_trigger_rate": _round(self.false_trigger_rate),
        }


def summarize(
    outcomes: Sequence[CueOutcome], category: str = "overall", ks: Sequence[int] = RECALL_KS
) -> CategoryMetrics:
    """Compute every metric over ``outcomes``."""
    return CategoryMetrics(
        category=category,
        n_cues=len(outcomes),
        n_positive=sum(1 for o in outcomes if o.should_fire),
        n_negative=sum(1 for o in outcomes if not o.should_fire),
        n_fired=sum(len(o.fired) for o in outcomes),
        recall_at={k: mean_recall_at_k(outcomes, k) for k in ks},
        precision=trigger_precision(outcomes),
        hit_rate=trigger_hit_rate(outcomes),
        false_trigger_rate=false_trigger_rate(outcomes),
    )


def by_category(
    outcomes: Sequence[CueOutcome],
    order: Sequence[str] = (),
    ks: Sequence[int] = RECALL_KS,
) -> dict[str, CategoryMetrics]:
    """Per-category metrics; categories listed in ``order`` first, then the rest sorted."""
    cats = [c for c in order if any(o.category == c for o in outcomes)]
    cats += sorted({o.category for o in outcomes} - set(cats))
    return {c: summarize([o for o in outcomes if o.category == c], c, ks) for c in cats}


def _round(v: float | None, digits: int = 4) -> float | None:
    return None if v is None else round(v, digits)
