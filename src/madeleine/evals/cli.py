"""``madeleine bench`` argument parsing: ``main(argv) -> exit code``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from madeleine.core.types import ChannelNotAvailableError
from madeleine.evals.runner import DEFAULT_SWEEP, run_bench, semantic_engine_factory


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="madeleine bench",
        description="Offline synthetic benchmark of cue-triggered recall.",
    )
    p.add_argument("--memories", type=int, default=500, help="memories to insert (500)")
    p.add_argument("--cues", type=int, default=300, help="labelled cues to run (300)")
    p.add_argument("--seed", type=int, default=0, help="generator seed (0)")
    p.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    p.add_argument("--output", type=Path, default=None, help="write the report to PATH")
    p.add_argument(
        "--text-backend",
        choices=("hash", "sentence-transformers"),
        default="hash",
        help="text embedder: offline 'hash' (default) or 'sentence-transformers' "
        "(needs the [text] extra; downloads the model)",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help=f"also sweep trigger thresholds {DEFAULT_SWEEP[0]:.2f}..{DEFAULT_SWEEP[-1]:.2f}",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark; print (or write) the report. Returns a process exit code."""
    args = _parser().parse_args(None if argv is None else list(argv))
    if args.memories < 1 or args.cues < 0:
        print("error: --memories must be >= 1 and --cues >= 0", file=sys.stderr)
        return 2
    try:
        extra: dict[str, Any] = {}
        if args.text_backend != "hash":
            extra["engine_factory"] = semantic_engine_factory()
        report = run_bench(
            memories=args.memories,
            cues=args.cues,
            seed=args.seed,
            thresholds=DEFAULT_SWEEP if args.sweep else None,
            **extra,
        )
    except ImportError as exc:
        print(f"error: the recall engine is not available ({exc})", file=sys.stderr)
        return 1
    except ChannelNotAvailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    text = (
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n"
        if args.json
        else report.to_markdown()
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0
