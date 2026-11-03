"""Private helpers for optional extras: availability checks and lazy imports.

``extra_available`` uses :func:`importlib.util.find_spec`, which locates a package without
importing it (so checking for ``torch`` never imports torch).
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType
from typing import Final

from madeleine.core.types import ChannelName, ChannelNotAvailableError

EXTRA_MODULES: Final[dict[str, tuple[str, ...]]] = {
    "text": ("sentence_transformers",),
    "image": ("open_clip", "torch", "PIL"),
    "audio": ("transformers", "torch", "soundfile"),
}
"""Top-level modules each pip extra must provide."""


def install_hint(channel: ChannelName, extra: str) -> str:
    return f"channel {channel!r} needs: pip install 'madeleine-mcp[{extra}]'"


def module_available(name: str) -> bool:
    """True if ``name`` can be imported (checked without importing it)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def missing_modules(extra: str) -> list[str]:
    """Modules of ``extra`` that are not installed."""
    return [m for m in EXTRA_MODULES[extra] if not module_available(m)]


def extra_available(extra: str) -> bool:
    return not missing_modules(extra)


def import_extra(module: str, *, channel: ChannelName, extra: str) -> ModuleType:
    """Import ``module`` or raise :class:`ChannelNotAvailableError` with the pip hint."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ChannelNotAvailableError(
            f"{install_hint(channel, extra)} (could not import {module!r}: {exc})"
        ) from exc
