"""Run the synthetic benchmark against a :class:`~madeleine.core.types.MemoryEngine`.

Methodology (see ``docs/benchmarks.md``):

1. Generate a deterministic dataset (:func:`madeleine.evals.generator.generate`).
2. Create a fresh engine through ``engine_factory(recall_config)`` with
   ``refractory_period = 0`` so every cue is measured independently.
3. Insert every memory with ``engine.remember``; then for each cue, in chronological
   order: ``build_cue`` ->
   ``recall(RecallQuery(cue, k=10))`` (PULL) -> ``sense(cue)`` (PUSH) -> drain the queue.
4. Habituation: a second fresh engine with the configured (non-zero) refractory period;
   one positive cue is repeated several times inside that period (expected: the target
   fires exactly once), then once more whole weeks later, after the period, at the same
   local time (expected: fires again).
5. Optional threshold sweep. Under the ADR 0001 push rule the set that fires at threshold
   ``t`` is exactly the set that fires at a lower threshold ``t0 <= t`` filtered by
   ``score >= t`` (eligibility does not depend on the threshold, and the top
   ``max_push_per_cue`` of a score-sorted prefix is a prefix of the top
   ``max_push_per_cue``). So the per-cue run uses ``min(sweep + [configured])`` and every
   threshold, including the configured one, is derived from a single pass.
"""

from __future__ import annotations

import importlib
import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any, cast

from madeleine.core.types import (
    ChannelNotAvailableError,
    JSONValue,
    MadeleineConfig,
    MemoryEngine,
    RawSignal,
    RecallConfig,
    RecallQuery,
)
from madeleine.evals.generator import (
    ALL_CATEGORIES,
    CONTEXT,
    MULTI,
    OLD_MEMORY,
    SyntheticCue,
    SyntheticDataset,
    SyntheticMemory,
    generate,
)
from madeleine.evals.metrics import (
    RECALL_KS,
    CategoryMetrics,
    CueOutcome,
    LatencyStats,
    by_category,
    false_trigger_rate,
    latency_stats,
    summarize,
    trigger_hit_rate,
    trigger_precision,
)

__all__ = [
    "DEFAULT_SWEEP",
    "BenchReport",
    "EngineFactory",
    "HabituationResult",
    "SweepRow",
    "default_engine_factory",
    "run_bench",
]

_log = logging.getLogger(__name__)

EngineFactory = Callable[[RecallConfig], MemoryEngine]
"""``engine_factory(recall_config) -> MemoryEngine``; called once per fresh engine."""

DEFAULT_SWEEP: tuple[float, ...] = tuple(round(0.40 + 0.05 * i, 2) for i in range(9))
"""0.40, 0.45, ..., 0.80."""

PULL_K = 10
HABITUATION_REPEATS = 5
CUE_SOURCE = "bench"


def default_engine_factory(recall_config: RecallConfig) -> MemoryEngine:
    """In-memory real engine with every ML channel backed by the offline hash embedder.

    Imports ``madeleine.engine`` and ``madeleine.channels.registry`` lazily, so this module
    imports fine without them.
    """
    engine_mod: Any = importlib.import_module("madeleine.engine")
    registry_mod: Any = importlib.import_module("madeleine.channels.registry")
    config = MadeleineConfig(db_path=":memory:", text_backend="hash", recall=recall_config)
    engine = engine_mod.create_engine(config, registry=registry_mod.hash_registry())
    return cast(MemoryEngine, engine)


# -- report dataclasses -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SweepRow:
    threshold: float
    precision: float | None
    hit_rate: float | None
    false_trigger_rate: float | None
    n_fired: int

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "threshold": self.threshold,
            "precision": _r(self.precision),
            "hit_rate": _r(self.hit_rate),
            "false_trigger_rate": _r(self.false_trigger_rate),
            "n_fired": self.n_fired,
        }


@dataclass(frozen=True, slots=True)
class HabituationResult:
    """Same cue repeated inside the refractory period."""

    cue_index: int | None
    category: str | None
    target: int | None
    repeats: int
    spacing_s: float
    refractory_s: float
    fires_per_repeat: tuple[int, ...]
    target_fires: int
    max_fires_per_memory: int
    target_fires_after_refractory: int
    passed: bool
    note: str = ""

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "cue_index": self.cue_index,
            "category": self.category,
            "target": self.target,
            "repeats": self.repeats,
            "spacing_s": self.spacing_s,
            "refractory_s": self.refractory_s,
            "fires_per_repeat": list(self.fires_per_repeat),
            "target_fires": self.target_fires,
            "max_fires_per_memory": self.max_fires_per_memory,
            "target_fires_after_refractory": self.target_fires_after_refractory,
            "passed": self.passed,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class BenchReport:
    """Result of :func:`run_bench`. Render with :meth:`to_markdown` or :meth:`to_dict`."""

    config: Mapping[str, JSONValue]
    overall: CategoryMetrics
    per_category: Mapping[str, CategoryMetrics]
    latency: Mapping[str, LatencyStats]
    habituation: HabituationResult | None = None
    sweep: tuple[SweepRow, ...] = ()
    channels: Mapping[str, Mapping[str, JSONValue]] = field(default_factory=dict)
    skipped: Mapping[str, int] = field(default_factory=dict)
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, JSONValue]:
        """JSON-serialisable view (``json.dumps(report.to_dict())`` works)."""
        return {
            "config": dict(self.config),
            "channels": {k: dict(v) for k, v in self.channels.items()},
            "overall": self.overall.to_dict(),
            "per_category": {k: v.to_dict() for k, v in self.per_category.items()},
            "latency": {k: v.to_dict() for k, v in self.latency.items()},
            "habituation": self.habituation.to_dict() if self.habituation else None,
            "sweep": [row.to_dict() for row in self.sweep],
            "skipped": dict(self.skipped),
            "duration_s": round(self.duration_s, 3),
        }

    def to_markdown(self) -> str:
        """Human-readable report (GitHub-flavoured markdown tables)."""
        lines: list[str] = ["# madeleine benchmark", ""]
        lines += ["## Configuration", ""]
        lines += [f"- **{k}**: `{_md_value(v)}`" for k, v in self.config.items()]
        lines += [f"- **duration**: {self.duration_s:.2f} s", ""]
        if self.channels:
            lines += ["### Channels", "", "| channel | backend | dim | weight | solo trigger |"]
            lines += ["|---|---|---:|---:|:---:|"]
            for name, info in self.channels.items():
                if not info.get("available", True):
                    lines.append(f"| {name} | unavailable | n/a | n/a | n/a |")
                    continue
                lines.append(
                    f"| {name} | {info.get('model_id') or 'deterministic'} | {info.get('dim')} "
                    f"| {info.get('weight')} | {'yes' if info.get('solo_trigger') else 'no'} |"
                )
            lines.append("")
        lines += ["## Overall", ""]
        lines += _category_table([self.overall])
        lines += ["", "## Per category", ""]
        lines += _category_table(list(self.per_category.values()))
        lines += [
            "",
            "R@k = capped recall@k of PULL recall; precision = relevant fires / all fires; "
            "hit = positive cues where a relevant memory fired; false = negative cues that "
            "fired anything. `n/a` = undefined (e.g. no relevant memories).",
            "",
            "## Latency (ms)",
            "",
            "| operation | n | p50 | p95 | mean | max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for op, st in self.latency.items():
            lines.append(
                f"| {op} | {st.n} | {_f(st.p50_ms, 2)} | {_f(st.p95_ms, 2)} "
                f"| {_f(st.mean_ms, 2)} | {_f(st.max_ms, 2)} |"
            )
        lines += ["", "## Habituation", ""]
        h = self.habituation
        if h is None:
            lines.append("Not run.")
        else:
            verdict = "PASS" if h.passed else "FAIL"
            lines.append(
                f"**{verdict}** - cue #{h.cue_index} ({h.category}) repeated {h.repeats}x every "
                f"{h.spacing_s:.0f} s inside a {h.refractory_s / 3600:.1f} h refractory period: "
                f"fires per repeat {list(h.fires_per_repeat)}, target fired {h.target_fires}x "
                f"(expected 1), max fires of any memory {h.max_fires_per_memory}; after the "
                f"period the target fired {h.target_fires_after_refractory}x (expected 1)."
            )
            if h.note:
                lines.append(f"\nNote: {h.note}")
        if self.sweep:
            lines += ["", "## Trigger threshold sweep", ""]
            lines += ["| threshold | precision | hit rate | false-trigger rate | fires |"]
            lines += ["|---:|---:|---:|---:|---:|"]
            for row in self.sweep:
                lines.append(
                    f"| {row.threshold:.2f} | {_f(row.precision)} | {_f(row.hit_rate)} "
                    f"| {_f(row.false_trigger_rate)} | {row.n_fired} |"
                )
        if any(self.skipped.values()):
            lines += ["", "## Skipped", ""]
            lines += [f"- {k}: {v}" for k, v in self.skipped.items() if v]
        return "\n".join(lines) + "\n"


def _f(v: float | None, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _r(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


def _md_value(v: JSONValue) -> str:
    if isinstance(v, dict):
        return ", ".join(f"{k}={_md_value(x)}" for k, x in v.items())
    if isinstance(v, list):
        return ", ".join(_md_value(x) for x in v)
    return str(v)


def _category_table(rows: Sequence[CategoryMetrics]) -> list[str]:
    ks = sorted({k for r in rows for k in r.recall_at}) or list(RECALL_KS)
    head = "| category | cues | +/- | " + " | ".join(f"R@{k}" for k in ks)
    head += " | precision | hit rate | false-trigger | fires |"
    sep = "|---|---:|:---:|" + "---:|" * len(ks) + "---:|---:|---:|---:|"
    out = [head, sep]
    for r in rows:
        sign = "+" if r.n_negative == 0 else "-" if r.n_positive == 0 else "±"
        recalls = " | ".join(_f(r.recall_at.get(k)) for k in ks)
        out.append(
            f"| {r.category} | {r.n_cues} | {sign} | {recalls} | {_f(r.precision)} "
            f"| {_f(r.hit_rate)} | {_f(r.false_trigger_rate)} | {r.n_fired} |"
        )
    return out


# -- running ------------------------------------------------------------------------------


@dataclass(slots=True)
class _CueRun:
    cue: SyntheticCue
    ranked: tuple[int, ...]
    fired: tuple[tuple[int, float], ...]  # (memory index, score)


@dataclass(slots=True)
class _Timings:
    remember: list[float] = field(default_factory=list)
    build_cue: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    sense: list[float] = field(default_factory=list)


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _insert(
    engine: MemoryEngine,
    memories: Sequence[SyntheticMemory],
    timings: _Timings | None,
    skipped: dict[str, int],
) -> dict[str, int]:
    """Insert memories; returns ``memory id -> dataset index``."""
    index_of: dict[str, int] = {}
    for m in memories:
        signals: Sequence[RawSignal] = m.signals
        t0 = time.perf_counter()
        try:
            mem = engine.remember(m.content, signals, context=m.context)
        except ChannelNotAvailableError:
            # The engine lacks the image/audio channel: keep the memory without media.
            skipped["media_signals_dropped"] = skipped.get("media_signals_dropped", 0) + 1
            t0 = time.perf_counter()
            mem = engine.remember(m.content, (), context=m.context)
        if timings is not None:
            timings.remember.append(_ms(t0))
        index_of[mem.id] = m.index
    return index_of


def _run_cues(
    engine: MemoryEngine,
    cues: Sequence[SyntheticCue],
    index_of: Mapping[str, int],
    timings: _Timings,
    skipped: dict[str, int],
) -> list[_CueRun]:
    runs: list[_CueRun] = []
    for sc in cues:
        t0 = time.perf_counter()
        try:
            cue = engine.build_cue(
                sc.signals,
                context=sc.context,
                include_time=sc.include_time,
                include_geo=sc.include_geo,
                source=CUE_SOURCE,
            )
        except ChannelNotAvailableError:
            key = f"cues_unavailable_channel:{sc.category}"
            skipped[key] = skipped.get(key, 0) + 1
            continue
        timings.build_cue.append(_ms(t0))
        t0 = time.perf_counter()
        ranked = engine.recall(RecallQuery(cue, k=PULL_K))
        timings.recall.append(_ms(t0))
        t0 = time.perf_counter()
        fired = engine.sense(cue)
        timings.sense.append(_ms(t0))
        engine.pending_recollections(drain=True)
        runs.append(
            _CueRun(
                cue=sc,
                ranked=tuple(index_of[s.memory.id] for s in ranked if s.memory.id in index_of),
                fired=tuple(
                    (index_of[r.memory.id], r.score) for r in fired if r.memory.id in index_of
                ),
            )
        )
    return runs


def _outcomes(runs: Sequence[_CueRun], threshold: float) -> list[CueOutcome]:
    return [
        CueOutcome(
            category=r.cue.category,
            relevant=r.cue.relevant,
            should_fire=r.cue.should_fire,
            ranked=r.ranked,
            fired=tuple(i for i, score in r.fired if score >= threshold - 1e-12),
        )
        for r in runs
    ]


def _pick_habituation_cue(
    outcomes: Sequence[CueOutcome], cues: Sequence[SyntheticCue]
) -> tuple[SyntheticCue | None, str]:
    paired = list(zip(outcomes, cues, strict=True))  # same order: both come from the runs
    for wanted in (MULTI, CONTEXT, OLD_MEMORY):
        for o, c in paired:
            if c.category == wanted and c.target is not None and c.target in o.fired:
                return c, ""
    for c in cues:
        if c.category in (MULTI, CONTEXT) and c.target is not None:
            return c, "no positive cue fired its target in the main run; repeated anyway"
    return None, "no suitable positive cue in the dataset"


def _habituation(
    factory: EngineFactory,
    base: RecallConfig,
    dataset: SyntheticDataset,
    cue: SyntheticCue | None,
    note: str,
) -> HabituationResult:
    refractory = base.refractory_period
    if refractory <= timedelta(0):
        refractory = RecallConfig().refractory_period
        note = (note + "; " if note else "") + "refractory_period was 0, used the default"
    spacing = min(timedelta(minutes=10), refractory / (HABITUATION_REPEATS + 1))
    if cue is None or cue.target is None:
        return HabituationResult(
            None, None, None, 0, spacing.total_seconds(), refractory.total_seconds(),
            (), 0, 0, 0, False, note,
        )  # fmt: skip
    engine = factory(replace(base, refractory_period=refractory))
    try:
        index_of = _insert(engine, dataset.memories, None, {})
        counts: dict[int, int] = {}
        per_repeat: list[int] = []

        def fire(offset: timedelta) -> list[int]:
            ctx = replace(cue.context, timestamp=cue.context.timestamp + offset)
            built = engine.build_cue(
                cue.signals,
                context=ctx,
                include_time=cue.include_time,
                include_geo=cue.include_geo,
                source=CUE_SOURCE,
            )
            fired = [index_of[r.memory.id] for r in engine.sense(built) if r.memory.id in index_of]
            engine.pending_recollections(drain=True)
            return fired

        for i in range(HABITUATION_REPEATS):
            fired = fire(spacing * i)
            per_repeat.append(len(fired))
            for idx in fired:
                counts[idx] = counts.get(idx, 0) + 1
        # Re-check once the period is over, at the same local time and weekday (whole weeks
        # later) so that only the refractory state differs, not the time-of-day match.
        weeks = max(1, math.ceil(refractory / timedelta(days=7)))
        after = fire(timedelta(days=7 * weeks))
    finally:
        engine.close()
    target_fires = counts.get(cue.target, 0)
    max_fires = max(counts.values(), default=0)
    after_target = after.count(cue.target)
    return HabituationResult(
        cue_index=cue.index,
        category=cue.category,
        target=cue.target,
        repeats=HABITUATION_REPEATS,
        spacing_s=spacing.total_seconds(),
        refractory_s=refractory.total_seconds(),
        fires_per_repeat=tuple(per_repeat),
        target_fires=target_fires,
        max_fires_per_memory=max_fires,
        target_fires_after_refractory=after_target,
        passed=target_fires == 1 and max_fires <= 1 and after_target == 1,
        note=note,
    )


def _describe_channels(
    engine: MemoryEngine, rc: RecallConfig
) -> dict[str, Mapping[str, JSONValue]]:
    out: dict[str, Mapping[str, JSONValue]] = {}
    for name in engine.channels.names():
        try:
            ch = engine.channels.get(name)
        except Exception:  # fakes or partial registries: report as unavailable
            out[name] = {"available": False}
            continue
        out[name] = {
            "available": True,
            "model_id": ch.spec.model_id or ch.embedder.model_id,
            "dim": ch.spec.dim,
            "weight": float(rc.channel_weights.get(name, ch.spec.default_weight)),
            "solo_trigger": ch.spec.solo_trigger,
        }
    return out


def _recall_config_dict(rc: RecallConfig) -> dict[str, JSONValue]:
    return {
        "trigger_threshold": rc.trigger_threshold,
        "pull_threshold": rc.pull_threshold,
        "min_shared_channels": rc.min_shared_channels,
        "coverage_exponent": rc.coverage_exponent,
        "salience_weight": rc.salience_weight,
        "half_life_days": rc.half_life.total_seconds() / 86400,
        "decay_floor": rc.decay_floor,
        "refractory_period_h": rc.refractory_period.total_seconds() / 3600,
        "max_push_per_cue": rc.max_push_per_cue,
        "candidate_k": rc.candidate_k,
        "channel_weights": {k: float(v) for k, v in sorted(rc.channel_weights.items())},
    }


def run_bench(
    *,
    memories: int = 500,
    cues: int = 300,
    seed: int = 0,
    engine_factory: EngineFactory | None = None,
    recall_config: RecallConfig | None = None,
    thresholds: Sequence[float] | None = None,
    habituation: bool = True,
) -> BenchReport:
    """Run the synthetic benchmark and return a :class:`BenchReport`.

    * ``engine_factory(recall_config) -> MemoryEngine`` builds each fresh engine; default
      :func:`default_engine_factory` (in-memory SQLite + hash embedders).
    * ``recall_config`` - fusion parameters under test (default :class:`RecallConfig`).
      Per-cue measurements override ``refractory_period`` with 0.
    * ``thresholds`` - optional trigger thresholds to sweep (e.g. :data:`DEFAULT_SWEEP`).
    * ``habituation`` - run the refractory scenario (needs a second engine).
    """
    started = time.perf_counter()
    factory = engine_factory or default_engine_factory
    base = recall_config or RecallConfig()
    sweep = sorted({round(float(t), 6) for t in thresholds or ()})
    for t in sweep:
        if not 0.0 <= t <= 1.0:
            raise ValueError(f"sweep thresholds must be in [0, 1], got {t}")
    measure_threshold = min([*sweep, base.trigger_threshold])
    per_cue_config = replace(
        base, refractory_period=timedelta(0), trigger_threshold=measure_threshold
    )

    dataset = generate(memories=memories, cues=cues, seed=seed)
    timings = _Timings()
    skipped: dict[str, int] = {}
    engine = factory(per_cue_config)
    try:
        engine_name = type(engine).__module__ + "." + type(engine).__qualname__
        text_backend = engine.config.text_backend
        channels = _describe_channels(engine, base)
        index_of = _insert(engine, dataset.memories, timings, skipped)
        # Chronological order: time only moves forward, so a memory's last trigger is never
        # after the cue being evaluated (a negative "since last fired" would read as
        # refractory in some engines even with a zero refractory period).
        ordered = sorted(dataset.cues, key=lambda c: (c.context.timestamp, c.index))
        runs = _run_cues(engine, ordered, index_of, timings, skipped)
    finally:
        engine.close()

    outcomes = _outcomes(runs, base.trigger_threshold)
    overall = summarize(outcomes)
    per_category = by_category(outcomes, order=ALL_CATEGORIES)
    sweep_rows = []
    for t in sweep:
        o = _outcomes(runs, t)
        sweep_rows.append(
            SweepRow(
                threshold=t,
                precision=trigger_precision(o),
                hit_rate=trigger_hit_rate(o),
                false_trigger_rate=false_trigger_rate(o),
                n_fired=sum(len(x.fired) for x in o),
            )
        )

    hab: HabituationResult | None = None
    if habituation:
        hab_cue, note = _pick_habituation_cue(outcomes, [r.cue for r in runs])
        hab = _habituation(factory, base, dataset, hab_cue, note)

    config: dict[str, JSONValue] = {
        "memories": memories,
        "cues": len(dataset.cues),
        "seed": seed,
        "engine": engine_name,
        "text_backend": text_backend,
        "pull_k": PULL_K,
        "per_cue_refractory": "0 (cues measured independently)",
        "measured_at_threshold": measure_threshold,
        "recall": _recall_config_dict(base),
        "category_counts": dict(dataset.category_counts()),
    }
    return BenchReport(
        config=config,
        overall=overall,
        per_category=per_category,
        latency={
            "remember": latency_stats(timings.remember),
            "build_cue": latency_stats(timings.build_cue),
            "recall": latency_stats(timings.recall),
            "sense": latency_stats(timings.sense),
        },
        habituation=hab,
        sweep=tuple(sweep_rows),
        channels=channels,
        skipped=skipped,
        duration_s=time.perf_counter() - started,
    )
