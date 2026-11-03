from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from madeleine.evals.metrics import (
    CueOutcome,
    by_category,
    false_trigger_rate,
    latency_stats,
    mean_recall_at_k,
    percentile,
    recall_at_k,
    summarize,
    trigger_hit_rate,
    trigger_precision,
)

indices = st.integers(min_value=0, max_value=30)
rankings = st.lists(indices, max_size=15, unique=True).map(tuple)
relevant_sets = st.frozensets(indices, max_size=10)


@st.composite
def outcomes(draw: st.DrawFn) -> CueOutcome:
    rel = draw(relevant_sets)
    return CueOutcome(
        category=draw(st.sampled_from(["a", "b", "c"])),
        relevant=rel,
        should_fire=bool(rel),
        ranked=draw(rankings),
        fired=draw(st.lists(indices, max_size=3, unique=True).map(tuple)),
    )


def test_recall_at_k_examples() -> None:
    assert recall_at_k((1, 2, 3), frozenset({1}), 1) == 1.0
    assert recall_at_k((2, 1, 3), frozenset({1}), 1) == 0.0
    assert recall_at_k((2, 1, 3), frozenset({1}), 5) == 1.0
    # Capped: 2 of 2 possible slots in top-2, although 4 memories are relevant.
    assert recall_at_k((1, 2, 9), frozenset({1, 2, 3, 4}), 2) == 1.0
    assert recall_at_k((1, 9, 2), frozenset({1, 2, 3, 4}), 2) == 0.5
    assert recall_at_k((), frozenset({1}), 10) == 0.0
    assert recall_at_k((1, 2), frozenset(), 10) is None
    with pytest.raises(ValueError, match="k must"):
        recall_at_k((1,), frozenset({1}), 0)


@given(rankings, relevant_sets, st.integers(min_value=1, max_value=12))
def test_recall_bounds_and_monotonic(ranked: tuple[int, ...], rel: frozenset[int], k: int) -> None:
    v = recall_at_k(ranked, rel, k)
    if not rel:
        assert v is None
        return
    assert v is not None
    assert 0.0 <= v <= 1.0
    perfect = tuple(sorted(rel)) + tuple(i for i in ranked if i not in rel)
    assert recall_at_k(perfect, rel, k) == 1.0
    assert recall_at_k(tuple(i for i in ranked if i not in rel), rel, k) == 0.0


@given(rankings, relevant_sets, st.integers(min_value=1, max_value=12))
def test_recall_hits_never_decrease_with_k(
    ranked: tuple[int, ...], rel: frozenset[int], k: int
) -> None:
    hits = len(rel.intersection(ranked[:k]))
    assert hits <= len(rel.intersection(ranked[: k + 1]))


def test_trigger_metrics_examples() -> None:
    pos_hit = CueOutcome("p", frozenset({1}), True, fired=(1, 2))
    pos_miss = CueOutcome("p", frozenset({3}), True, fired=())
    neg_quiet = CueOutcome("n", frozenset(), False, fired=())
    neg_fired = CueOutcome("n", frozenset(), False, fired=(4,))
    all_ = [pos_hit, pos_miss, neg_quiet, neg_fired]
    assert trigger_precision(all_) == pytest.approx(1 / 3)
    assert trigger_hit_rate(all_) == 0.5
    assert false_trigger_rate(all_) == 0.5
    # Undefined cases are None, not 0/1.
    assert trigger_precision([pos_miss, neg_quiet]) is None
    assert trigger_hit_rate([neg_quiet]) is None
    assert false_trigger_rate([pos_hit]) is None
    assert trigger_precision([]) is None
    assert trigger_hit_rate([]) is None
    assert mean_recall_at_k([neg_quiet], 5) is None


@given(st.lists(outcomes(), max_size=25))
def test_trigger_metrics_bounded(outs: list[CueOutcome]) -> None:
    for v in (trigger_precision(outs), trigger_hit_rate(outs), false_trigger_rate(outs)):
        assert v is None or 0.0 <= v <= 1.0
    m = summarize(outs)
    assert m.n_cues == len(outs) == m.n_positive + m.n_negative
    assert m.n_fired == sum(len(o.fired) for o in outs)
    cats = by_category(outs, order=["c", "a"])
    assert sum(c.n_cues for c in cats.values()) == len(outs)
    assert list(cats)[: len([c for c in ("c", "a") if c in cats])] == [
        c for c in ("c", "a") if c in cats
    ]


@given(st.lists(outcomes(), min_size=1, max_size=20))
def test_perfect_and_zero_engines(outs: list[CueOutcome]) -> None:
    perfect = [
        CueOutcome(
            o.category,
            o.relevant,
            o.should_fire,
            tuple(sorted(o.relevant)),
            tuple(sorted(o.relevant))[:3],
        )
        for o in outs
    ]
    silent = [CueOutcome(o.category, o.relevant, o.should_fire) for o in outs]
    p, z = summarize(perfect), summarize(silent)
    if any(o.relevant for o in outs):
        assert all(p.recall_at[k] == 1.0 for k in (1, 5, 10))
        assert all(z.recall_at[k] == 0.0 for k in (1, 5, 10))
        assert p.precision == 1.0
        assert p.hit_rate == 1.0
        assert z.hit_rate == 0.0
    assert z.precision is None
    if any(not o.should_fire for o in outs):
        assert p.false_trigger_rate == 0.0
        assert z.false_trigger_rate == 0.0


def test_percentile_examples() -> None:
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 95) == pytest.approx(4.8)
    with pytest.raises(ValueError, match="q must"):
        percentile([1.0], 101)
    empty = latency_stats([])
    assert empty.n == 0
    assert empty.p50_ms is None
    assert latency_stats([1.0, 3.0]).to_dict()["mean_ms"] == 2.0


@given(st.lists(st.floats(min_value=0, max_value=1e6), min_size=1, max_size=50), st.floats(0, 100))
def test_percentile_within_range(values: list[float], q: float) -> None:
    v = percentile(values, q)
    assert v is not None
    assert min(values) - 1e-6 <= v <= max(values) + 1e-6
    assert percentile(values, 0) == min(values)
    assert percentile(values, 100) == max(values)
