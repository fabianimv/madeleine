from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from madeleine.core.fusion import (
    channel_weight,
    combine,
    coverage,
    decay_factor,
    fuse,
    is_refractory,
    push_eligible,
    salience_factor,
)
from madeleine.core.types import (
    Memory,
    RecallConfig,
    ScoredMemory,
    SensoryContext,
    SensoryFingerprint,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
NAMES = ("text", "geo", "time", "image", "audio")

sims = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
weights = st.floats(min_value=0.0, max_value=5.0, allow_nan=False)
ages = st.timedeltas(min_value=timedelta(0), max_value=timedelta(days=365 * 80))


@st.composite
def configs(draw: st.DrawFn) -> RecallConfig:
    return RecallConfig(
        coverage_exponent=draw(st.floats(0.0, 3.0)),
        salience_weight=draw(st.floats(0.0, 2.0)),
        half_life=draw(st.timedeltas(min_value=timedelta(hours=1), max_value=timedelta(days=5000))),
        decay_floor=draw(st.floats(0.0, 1.0)),
    )


@st.composite
def fusion_inputs(draw: st.DrawFn) -> tuple[dict[str, float], dict[str, float]]:
    cue = draw(st.lists(st.sampled_from(NAMES), min_size=1, unique=True))
    cue_weights = {c: draw(weights) for c in cue}
    shared = draw(st.lists(st.sampled_from(cue), unique=True))
    scores = {c: draw(sims) for c in shared}
    return scores, cue_weights


@settings(max_examples=300)
@given(fusion_inputs(), st.floats(0.0, 1.0), ages, configs())
def test_score_always_in_unit_interval(
    inputs: tuple[dict[str, float], dict[str, float]],
    salience: float,
    age: timedelta,
    config: RecallConfig,
) -> None:
    scores, cue_weights = inputs
    r = combine(scores, cue_weights, salience=salience, age=age, config=config)
    assert 0.0 <= r.score <= 1.0
    assert 0.0 <= r.base <= 1.0
    assert 0.0 <= r.coverage <= 1.0


@settings(max_examples=300)
@given(fusion_inputs(), st.data(), st.floats(0.0, 1.0), ages, configs())
def test_monotone_in_single_channel_similarity(
    inputs: tuple[dict[str, float], dict[str, float]],
    data: st.DataObject,
    salience: float,
    age: timedelta,
    config: RecallConfig,
) -> None:
    scores, cue_weights = inputs
    if not scores:
        return
    name = data.draw(st.sampled_from(sorted(scores)))
    lo, hi = sorted((scores[name], data.draw(sims)))
    a = combine({**scores, name: lo}, cue_weights, salience=salience, age=age, config=config)
    b = combine({**scores, name: hi}, cue_weights, salience=salience, age=age, config=config)
    assert a.score <= b.score


@given(fusion_inputs(), st.floats(0.0, 1.0), st.floats(0.0, 1.0), ages, configs())
def test_monotone_in_salience(
    inputs: tuple[dict[str, float], dict[str, float]],
    s1: float,
    s2: float,
    age: timedelta,
    config: RecallConfig,
) -> None:
    scores, cue_weights = inputs
    lo, hi = sorted((s1, s2))
    a = combine(scores, cue_weights, salience=lo, age=age, config=config)
    b = combine(scores, cue_weights, salience=hi, age=age, config=config)
    assert a.score <= b.score
    assert salience_factor(lo, config) <= salience_factor(hi, config)


@given(ages, ages, configs())
def test_decay_non_increasing_and_bounded(
    a1: timedelta, a2: timedelta, config: RecallConfig
) -> None:
    lo, hi = sorted((a1, a2))
    assert decay_factor(lo, config) >= decay_factor(hi, config)
    assert config.decay_floor <= decay_factor(hi, config) <= 1.0


@given(configs())
def test_decay_is_one_at_zero_and_tends_to_floor(config: RecallConfig) -> None:
    assert decay_factor(timedelta(0), config) == 1.0
    assert decay_factor(-timedelta(days=3), config) == 1.0
    far = decay_factor(config.half_life * 200, config)
    assert math.isclose(far, config.decay_floor, abs_tol=1e-9)


def test_decay_half_life_default() -> None:
    cfg = RecallConfig()
    assert decay_factor(cfg.half_life, cfg) == pytest.approx(0.7 + 0.3 * 0.5)


def test_salience_factor_defaults() -> None:
    cfg = RecallConfig()
    assert salience_factor(0.5, cfg) == 1.0
    assert salience_factor(0.0, cfg) == pytest.approx(0.8)
    assert salience_factor(1.0, cfg) == pytest.approx(1.2)


@given(fusion_inputs(), st.sampled_from(NAMES), weights)
def test_coverage_penalises_missing_cue_channels(
    inputs: tuple[dict[str, float], dict[str, float]], extra: str, w: float
) -> None:
    scores, cue_weights = inputs
    if extra in cue_weights or w == 0:
        return
    cfg = RecallConfig()
    a = combine(scores, cue_weights, salience=0.5, age=timedelta(0), config=cfg)
    b = combine(scores, {**cue_weights, extra: w}, salience=0.5, age=timedelta(0), config=cfg)
    assert b.coverage <= a.coverage
    assert b.score <= a.score


def test_coverage_formula() -> None:
    cfg = RecallConfig(coverage_exponent=0.5)
    assert coverage(1.0, 4.0, cfg) == pytest.approx(0.5)
    assert coverage(0.0, 4.0, cfg) == 0.0
    assert coverage(2.0, 2.0, cfg) == 1.0
    cfg0 = RecallConfig(coverage_exponent=0.0)
    assert coverage(1.0, 100.0, cfg0) == 1.0


def test_exact_formula_example() -> None:
    cfg = RecallConfig()
    r = combine(
        {"geo": 1.0, "time": 0.5},
        {"geo": 1.0, "time": 0.5, "text": 1.0},
        salience=0.75,
        age=cfg.half_life,
        config=cfg,
    )
    base = (1.0 * 1.0 + 0.5 * 0.5) / 1.5
    cov = (1.5 / 2.5) ** 0.5
    expected = base * cov * (1 + 0.4 * 0.25) * (0.7 + 0.3 * 0.5)
    assert r.score == pytest.approx(expected)


@given(fusion_inputs(), sims)
def test_zero_weight_channel_is_ignored(
    inputs: tuple[dict[str, float], dict[str, float]], sim: float
) -> None:
    scores, cue_weights = inputs
    cfg = RecallConfig()
    free = [n for n in NAMES if n not in cue_weights]
    if not free:
        return
    a = combine(scores, cue_weights, salience=0.5, age=timedelta(0), config=cfg)
    b = combine(
        {**scores, free[0]: sim},
        {**cue_weights, free[0]: 0.0},
        salience=0.5,
        age=timedelta(0),
        config=cfg,
    )
    assert a == b


def _memory(embeddings: dict[str, tuple[float, ...]], *, salience: float = 0.5) -> Memory:
    ctx = SensoryContext(timestamp=T0, salience=salience)
    return Memory(
        id="mem_1",
        content="x",
        fingerprint=SensoryFingerprint(embeddings=embeddings, context=ctx),
        created_at=T0,
    )


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return max(0.0, sum(x * y for x, y in zip(a, b, strict=True)))


def test_fuse_channels_shared_weights_and_restriction() -> None:
    mem = _memory({"text": (1.0, 0.0), "geo": (0.0, 1.0)})
    cue = {"text": (1.0, 0.0), "geo": (1.0, 0.0), "image": (1.0, 0.0)}
    simf = dict.fromkeys(cue, _dot)
    cfg = RecallConfig()
    w = {"text": 1.0, "geo": 1.0, "image": 1.0}
    s = fuse(cue, mem, simf, w, cfg, T0)
    assert dict(s.channel_scores) == {"text": 1.0, "geo": 0.0}
    assert s.score == pytest.approx(0.5 * (2 / 3) ** 0.5)
    # weight 0 removes geo from S and C
    s0 = fuse(cue, mem, simf, {**w, "geo": 0.0}, cfg, T0)
    assert dict(s0.channel_scores) == {"text": 1.0}
    assert s0.score == pytest.approx((1 / 2) ** 0.5)
    # restriction to text only: full coverage
    sr = fuse(cue, mem, simf, w, cfg, T0, channels={"text"})
    assert dict(sr.channel_scores) == {"text": 1.0}
    assert sr.score == 1.0
    # nothing shared
    sn = fuse({"image": (1.0, 0.0)}, mem, simf, w, cfg, T0)
    assert sn.score == 0.0
    assert dict(sn.channel_scores) == {}


def test_fuse_clamps_bad_similarities() -> None:
    mem = _memory({"text": (1.0,)})
    cfg = RecallConfig()
    s = fuse({"text": (1.0,)}, mem, {"text": lambda a, b: float("nan")}, {"text": 1.0}, cfg, T0)
    assert s.score == 0.0
    s = fuse({"text": (1.0,)}, mem, {"text": lambda a, b: 1.5}, {"text": 1.0}, cfg, T0)
    assert s.score == 1.0


def test_channel_weight_override() -> None:
    cfg = RecallConfig(channel_weights={"time": 2.0})
    assert channel_weight("time", 0.5, cfg) == 2.0
    assert channel_weight("geo", 1.0, cfg) == 1.0


def _scored(channels: dict[str, float], score: float) -> ScoredMemory:
    return ScoredMemory(memory=_memory({}), score=score, channel_scores=channels)


def test_push_eligible_rules() -> None:
    cfg = RecallConfig(trigger_threshold=0.6, min_shared_channels=2)
    solo = {"geo", "text"}
    assert push_eligible(_scored({"geo": 1, "time": 1}, 0.9), solo, cfg)
    assert not push_eligible(_scored({"geo": 1, "time": 1}, 0.5), solo, cfg)
    assert not push_eligible(_scored({"geo": 1}, 0.9), solo, cfg)
    assert not push_eligible(_scored({"time": 1, "biometric": 1}, 1.0), solo, cfg)
    cfg1 = RecallConfig(trigger_threshold=0.6)
    assert not push_eligible(_scored({"time": 1.0}, 1.0), solo, cfg1)
    assert push_eligible(_scored({"text": 1.0}, 0.6), solo, cfg1)


@given(
    st.timedeltas(min_value=-timedelta(days=3), max_value=timedelta(days=3)),
    st.timedeltas(min_value=timedelta(0), max_value=timedelta(days=1)),
)
def test_is_refractory_symmetric(delta: timedelta, period: timedelta) -> None:
    expected = period > timedelta(0) and abs(delta) < period
    assert is_refractory(T0, T0 + delta, period) == expected
    assert is_refractory(T0 + delta, T0, period) == expected
    assert not is_refractory(None, T0 + delta, period)
    assert not is_refractory(T0, T0 + delta, timedelta(0))


def test_is_refractory_edges() -> None:
    period = timedelta(hours=6)
    assert is_refractory(T0, T0, period)
    assert is_refractory(T0, T0 - timedelta(minutes=5), period)  # late webhook
    assert not is_refractory(T0, T0 - period, period)
    assert not is_refractory(T0, T0 + period, period)
    assert not is_refractory(T0, T0 - timedelta(days=365), period)
    assert not is_refractory(T0, T0, timedelta(0))
