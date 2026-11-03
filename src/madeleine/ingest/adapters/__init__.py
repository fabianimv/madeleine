"""Source-specific :class:`~madeleine.core.types.CueAdapter` implementations."""

from __future__ import annotations

from madeleine.ingest.adapters.home_assistant import HomeAssistantAdapter
from madeleine.ingest.adapters.ios_shortcuts import IOSShortcutsAdapter

__all__ = ["HomeAssistantAdapter", "IOSShortcutsAdapter"]
