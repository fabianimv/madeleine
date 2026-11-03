"""Fixtures for the ingestion tests (FakeEngine-backed, fully offline)."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from madeleine.ingest import create_app
from madeleine.testing import FakeClock, FakeEngine


@pytest.fixture(autouse=True)
def _no_env_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MADELEINE_API_TOKEN", raising=False)
    monkeypatch.delenv("MADELEINE_MAX_BODY_BYTES", raising=False)
    # TestClient sends "Host: testserver"; allow it on top of the loopback defaults (R1).
    monkeypatch.setenv("MADELEINE_ALLOWED_HOSTS", "testserver")


@pytest.fixture
def engine(clock: FakeClock) -> FakeEngine:
    return FakeEngine(clock=clock)


@pytest.fixture
def client(engine: FakeEngine, clock: FakeClock) -> Iterator[TestClient]:
    with TestClient(create_app(engine, clock=clock)) as c:
        yield c
