"""Synthetic, offline benchmark of cue-triggered recall (``madeleine bench``).

>>> from madeleine.evals import run_bench
>>> report = run_bench(memories=500, cues=300, seed=0)   # doctest: +SKIP
>>> print(report.to_markdown())                          # doctest: +SKIP
"""

from __future__ import annotations

from madeleine.evals.runner import DEFAULT_SWEEP, BenchReport, EngineFactory, run_bench

__all__ = ["DEFAULT_SWEEP", "BenchReport", "EngineFactory", "run_bench"]
