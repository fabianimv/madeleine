"""Recall fusion: pure functions implementing the normative formula of ADR 0001.

For a cue *q* and a memory *m* (see :class:`~madeleine.core.types.RecallConfig`):

* ``S`` = channels present in both with weight ``w_c > 0``; ``C`` = cue channels with ``w_c > 0``.
* ``base = sum(w_c * s_c, S) / sum(w_c, S)`` (0 if ``S`` is empty)
* ``coverage = (sum(w_c, S) / sum(w_c, C)) ** coverage_exponent``
* ``salience_factor = 1 + salience_weight * (m.salience - 0.5)``
* ``decay_factor = decay_floor + (1 - decay_floor) * 0.5 ** (age / half_life)``
* ``score = clamp(base * coverage * salience_factor * decay_factor, 0, 1)``

Everything here is pure (no I/O, no state) so it can be property-tested in isolation and
re-used by evals / oracles. Refractory *state* lives in the store; this module only holds
the rules.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from madeleine.core.types import (
    ChannelName,
    Memory,
    RecallConfig,
    ScoredMemory,
    SimilarityFn,
    Vector,
)

__all__ = [
    "FusionResult",
    "channel_weight",
    "clamp01",
    "combine",
    "coverage",
    "decay_factor",
    "fuse",
    "is_refractory",
    "push_eligible",
    "salience_factor",
]


def clamp01(value: float) -> float:
    """Clamp to ``[0, 1]``; non-finite values (NaN, +-inf) map to ``0``."""
    if not math.isfinite(value):
        return 0.0
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def channel_weight(name: ChannelName, spec_default: float, config: RecallConfig) -> float:
    """Effective fusion weight: ``config.channel_weights[name]`` overrides the spec default."""
    return float(config.channel_weights.get(name, spec_default))


def coverage(shared_weight: float, cue_weight: float, config: RecallConfig) -> float:
    """``(sum_S w / sum_C w) ** coverage_exponent`` (0 when nothing is shared)."""
    if shared_weight <= 0.0 or cue_weight <= 0.0:
        return 0.0
    ratio = min(1.0, shared_weight / cue_weight)
    return float(ratio**config.coverage_exponent)


def salience_factor(salience: float, config: RecallConfig) -> float:
    """``1 + salience_weight * (salience - 0.5)``: a bounded multiplicative boost."""
    return 1.0 + config.salience_weight * (salience - 0.5)


def decay_factor(age: timedelta, config: RecallConfig) -> float:
    """``decay_floor + (1 - decay_floor) * 0.5 ** (age / half_life)``; ``age <= 0`` gives 1."""
    if age <= timedelta(0):
        return 1.0
    halvings = age / config.half_life
    return float(config.decay_floor + (1.0 - config.decay_floor) * 0.5**halvings)


@dataclass(frozen=True, slots=True)
class FusionResult:
    """The arithmetic outcome of fusion (before being attached to a memory)."""

    score: float
    base: float
    coverage: float
    salience_factor: float
    decay_factor: float


def combine(
    channel_scores: Mapping[ChannelName, float],
    cue_weights: Mapping[ChannelName, float],
    *,
    salience: float,
    age: timedelta,
    config: RecallConfig,
) -> FusionResult:
    """Fuse per-channel similarities into a score.

    ``cue_weights`` holds the weight of every cue channel considered (``C`` plus any
    zero-weight channels, which are ignored). ``channel_scores`` holds the similarity of each
    channel shared with the memory; keys without a positive cue weight are ignored.
    Similarities are clamped to ``[0, 1]``.
    """
    cue_total = 0.0
    for w in cue_weights.values():
        if w > 0.0:
            cue_total += w
    shared_total = 0.0
    weighted = 0.0
    for name, sim in channel_scores.items():
        w = cue_weights.get(name, 0.0)
        if w > 0.0:
            shared_total += w
            weighted += w * clamp01(sim)
    base = weighted / shared_total if shared_total > 0.0 else 0.0
    cov = coverage(shared_total, cue_total, config)
    sal = salience_factor(salience, config)
    dec = decay_factor(age, config)
    return FusionResult(
        score=clamp01(base * cov * sal * dec),
        base=base,
        coverage=cov,
        salience_factor=sal,
        decay_factor=dec,
    )


def fuse(
    cue_vectors: Mapping[ChannelName, Vector],
    memory: Memory,
    similarities: Mapping[ChannelName, SimilarityFn],
    weights: Mapping[ChannelName, float],
    config: RecallConfig,
    now: datetime,
    channels: Collection[ChannelName] | None = None,
) -> ScoredMemory:
    """Score ``memory`` against a cue given as per-channel vectors.

    * ``similarities`` / ``weights`` must cover every channel of ``cue_vectors`` that should
      take part; a channel missing from ``weights`` (or with weight 0) is ignored entirely.
    * ``channels`` restricts fusion (both ``S`` and ``C``) to those channels.
    * ``now`` is the cue timestamp (age is measured from ``memory.context.timestamp``).

    ``channel_scores`` of the result lists exactly the channels of ``S``.
    """
    cue_weights: dict[ChannelName, float] = {}
    for name in cue_vectors:
        if channels is not None and name not in channels:
            continue
        w = weights.get(name, 0.0)
        if w > 0.0:
            cue_weights[name] = w
    mem_vectors = memory.fingerprint.embeddings
    scores: dict[ChannelName, float] = {}
    for name in cue_weights:
        mv = mem_vectors.get(name)
        if mv is not None:
            scores[name] = clamp01(float(similarities[name](cue_vectors[name], mv)))
    result = combine(
        scores,
        cue_weights,
        salience=memory.context.salience,
        age=now - memory.context.timestamp,
        config=config,
    )
    return ScoredMemory(
        memory=memory,
        score=result.score,
        channel_scores=scores,
        salience_factor=result.salience_factor,
        decay_factor=result.decay_factor,
    )


def push_eligible(
    scored: ScoredMemory, solo_channels: Collection[ChannelName], config: RecallConfig
) -> bool:
    """PUSH rule minus refractory state.

    True iff ``score >= trigger_threshold``, ``|S| >= min_shared_channels`` and at least one
    channel of ``S`` (``scored.channel_scores``) is in ``solo_channels`` (channels whose
    spec has ``solo_trigger=True``).
    """
    shared = scored.channel_scores
    return (
        scored.score >= config.trigger_threshold
        and len(shared) >= config.min_shared_channels
        and any(name in solo_channels for name in shared)
    )


def is_refractory(last: datetime | None, at: datetime, period: timedelta) -> bool:
    """Whether a memory that last fired at ``last`` must stay silent for a cue at ``at``.

    Refractory iff it fired before (``last`` set), ``period > 0`` and ``at - last < period``
    (a cue older than the last trigger is therefore also refractory).
    """
    if last is None or period <= timedelta(0):
        return False
    return at - last < period
