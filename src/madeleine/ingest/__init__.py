"""HTTP ingestion: webhook cue adapters and the FastAPI app (``create_app``).

Importing this package is cheap: the adapters are pure Python and FastAPI is only imported
when :func:`create_app` (``madeleine.ingest.app``) is first accessed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from madeleine.core.types import Clock, CueAdapter
from madeleine.ingest.adapters import HomeAssistantAdapter, IOSShortcutsAdapter
from madeleine.ingest.webhook import GenericWebhookAdapter

if TYPE_CHECKING:
    from madeleine.ingest.app import create_app

__all__ = [
    "GenericWebhookAdapter",
    "HomeAssistantAdapter",
    "IOSShortcutsAdapter",
    "create_app",
    "default_adapters",
]


def default_adapters(*, clock: Clock | None = None) -> Sequence[CueAdapter]:
    """The built-in adapters: ``generic``, ``home_assistant``, ``ios_shortcuts``."""
    return (
        GenericWebhookAdapter(clock=clock),
        HomeAssistantAdapter(clock=clock),
        IOSShortcutsAdapter(clock=clock),
    )


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from madeleine.ingest.app import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
