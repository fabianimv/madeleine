"""Shared fixtures (owned by the lead)."""

from __future__ import annotations

import pytest

from madeleine.testing import FakeClock


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()
