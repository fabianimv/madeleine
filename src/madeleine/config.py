"""Build a :class:`MadeleineConfig` from defaults, ``MADELEINE_*`` env vars and overrides."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from madeleine.core.types import MadeleineConfig, TextBackend

_TEXT_BACKENDS: frozenset[str] = frozenset({"auto", "sentence-transformers", "hash"})


def load_config(env: Mapping[str, str] | None = None, **overrides: Any) -> MadeleineConfig:
    """Return the effective configuration.

    Precedence: explicit ``overrides`` > environment > defaults. Recognised variables:

    * ``MADELEINE_DB`` - SQLite path (``:memory:`` allowed)
    * ``MADELEINE_TEXT_BACKEND`` - ``auto`` | ``sentence-transformers`` | ``hash``
    * ``MADELEINE_CHANNELS`` - comma-separated enabled channels
    * ``MADELEINE_STORE_RAW`` - ``1``/``true`` to persist raw media (default off)
    """
    env = os.environ if env is None else env
    cfg = MadeleineConfig()
    if db := env.get("MADELEINE_DB"):
        cfg = replace(cfg, db_path=db)
    if backend := env.get("MADELEINE_TEXT_BACKEND"):
        if backend not in _TEXT_BACKENDS:
            raise ValueError(f"MADELEINE_TEXT_BACKEND must be one of {sorted(_TEXT_BACKENDS)}")
        cfg = replace(cfg, text_backend=cast(TextBackend, backend))
    if channels := env.get("MADELEINE_CHANNELS"):
        cfg = replace(cfg, channels=tuple(c.strip() for c in channels.split(",") if c.strip()))
    if raw := env.get("MADELEINE_STORE_RAW"):
        cfg = replace(cfg, store_raw=raw.strip().lower() in {"1", "true", "yes", "on"})
    overrides = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, **overrides) if overrides else cfg


def resolve_db_path(db_path: str) -> str:
    """Expand ``~`` and create the parent directory (no-op for ``:memory:``)."""
    if db_path == ":memory:":
        return db_path
    path = Path(db_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)
