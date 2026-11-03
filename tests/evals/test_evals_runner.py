from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

import pytest

from madeleine.core.types import MemoryEngine, RecallConfig
from madeleine.evals import DEFAULT_SWEEP, BenchReport, run_bench
from madeleine.evals.generator import ALL_CATEGORIES
from madeleine.testing import FakeEngine


class Factory:
    """FakeEngine-based factory that records the configs it was asked for."""

    def __init__(self) -> None:
        self.configs: list[RecallConfig] = []
        self.engines: list[FakeEngine] = []

    def __call__(self, rc: RecallConfig) -> MemoryEngine:
        self.configs.append(rc)
        eng = FakeEngine(trigger_threshold=rc.trigger_threshold, refractory=rc.refractory_period)
        self.engines.append(eng)
        return eng


@pytest.fixture(scope="module")
def report() -> BenchReport:
    return run_bench(memories=40, cues=30, seed=0, engine_factory=Factory())


def test_end_to_end_report(report: BenchReport) -> None:
    assert report.overall.n_cues == 30
    assert set(report.per_category) == set(ALL_CATEGORIES)
    assert sum(m.n_cues for m in report.per_category.values()) == 30
    assert report.latency["remember"].n == 40
    assert report.latency["recall"].n == report.latency["sense"].n == 30
    assert report.config["engine"] == "madeleine.testing.FakeEngine"
    # Negatives have no recall@k; the FakeEngine should never fire on time-only cues.
    assert report.per_category["distractor"].recall_at[10] is None
    assert report.per_category["time_only"].n_fired == 0
    assert report.per_category["text_paraphrase"].recall_at[10] is not None


def test_markdown_and_json_render(report: BenchReport) -> None:
    md = report.to_markdown()
    for section in (
        "## Configuration",
        "## Overall",
        "## Per category",
        "## Latency",
        "## Habituation",
    ):
        assert section in md
    for cat in ALL_CATEGORIES:
        assert f"| {cat} |" in md
    data: dict[str, Any] = json.loads(json.dumps(report.to_dict()))
    assert data["overall"]["n_cues"] == 30
    assert set(data["per_category"]) == set(ALL_CATEGORIES)
    assert data["habituation"]["repeats"] == 5
    assert data["sweep"] == []


def test_fresh_engines_and_configs() -> None:
    f = Factory()
    rc = RecallConfig(trigger_threshold=0.55, refractory_period=timedelta(hours=2))
    rep = run_bench(memories=40, cues=20, engine_factory=f, recall_config=rc)
    per_cue, hab = f.configs
    assert per_cue.refractory_period == timedelta(0)
    assert per_cue.trigger_threshold == 0.55
    assert hab.refractory_period == timedelta(hours=2)
    assert all(e.closed for e in f.engines)
    assert all(e.queue.pending_count() == 0 for e in f.engines)
    assert rep.config["recall"]["trigger_threshold"] == 0.55  # type: ignore[index]


def test_habituation_fires_once(report: BenchReport) -> None:
    h = report.habituation
    assert h is not None
    assert h.target is not None
    assert h.spacing_s * (h.repeats - 1) < h.refractory_s
    assert h.max_fires_per_memory <= 1
    assert h.passed == (h.target_fires == 1 and h.target_fires_after_refractory == 1)
    assert h.passed  # FakeEngine honours its refractory period


def test_sweep_matches_direct_runs() -> None:
    """The sweep derives every threshold from one low-threshold pass; check it against
    independent runs at each threshold."""
    rep = run_bench(
        memories=40, cues=30, engine_factory=Factory(), thresholds=[0.5, 0.7], habituation=False
    )
    assert [r.threshold for r in rep.sweep] == [0.5, 0.7]
    assert rep.config["measured_at_threshold"] == 0.5
    for row in rep.sweep:
        direct = run_bench(
            memories=40,
            cues=30,
            engine_factory=Factory(),
            recall_config=RecallConfig(trigger_threshold=row.threshold),
            habituation=False,
        )
        assert direct.overall.precision == row.precision
        assert direct.overall.hit_rate == row.hit_rate
        assert direct.overall.false_trigger_rate == row.false_trigger_rate
        assert direct.overall.n_fired == row.n_fired
    # The configured threshold (0.62) is also derived, and matches a plain run.
    plain = run_bench(memories=40, cues=30, engine_factory=Factory(), habituation=False)
    assert rep.overall == plain.overall
    md = rep.to_markdown()
    assert "## Trigger threshold sweep" in md
    assert "| 0.70 |" in md


def test_deterministic_metrics() -> None:
    a = run_bench(memories=40, cues=30, seed=4, engine_factory=Factory(), habituation=False)
    b = run_bench(memories=40, cues=30, seed=4, engine_factory=Factory(), habituation=False)
    assert a.overall == b.overall
    assert a.per_category == b.per_category


def test_invalid_sweep() -> None:
    with pytest.raises(ValueError, match="sweep"):
        run_bench(memories=10, cues=5, engine_factory=Factory(), thresholds=[1.5])
    assert len(DEFAULT_SWEEP) == 9
    assert DEFAULT_SWEEP[0] == 0.4
    assert DEFAULT_SWEEP[-1] == 0.8


def test_default_factory_real_engine() -> None:
    pytest.importorskip("madeleine.engine")
    pytest.importorskip("madeleine.channels.registry")
    rep = run_bench(memories=60, cues=45, seed=0, thresholds=DEFAULT_SWEEP)
    assert rep.overall.n_cues == 45
    assert rep.per_category["time_only"].n_fired == 0  # solo_trigger rule
    assert rep.per_category["exact_media"].recall_at[1] == 1.0
    assert rep.habituation is not None
    assert rep.habituation.max_fires_per_memory <= 1
    json.dumps(rep.to_dict())
