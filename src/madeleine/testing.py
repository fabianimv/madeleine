"""Test helpers shared by madeleine's own tests and by downstream users."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from madeleine.core.types import _require_aware


class FakeClock:
    """A controllable :class:`~madeleine.core.types.Clock`."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
        _require_aware(self._now, "FakeClock start")

    def now(self) -> datetime:
        return self._now

    def advance(self, delta: timedelta) -> datetime:
        self._now += delta
        return self._now

    def set(self, when: datetime) -> None:
        _require_aware(when, "FakeClock.set")
        self._now = when
