from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from madeleine.core.types import MadeleineConfig, RecallConfig
from madeleine.engine import Madeleine, create_engine
from madeleine.store import InMemoryRecollectionQueue, InMemoryStore
from madeleine.testing import FakeClock

from .fakes import fake_registry

EngineFactory = Callable[..., Madeleine]


def build_engine(
    kind: str,
    clock: FakeClock,
    *,
    recall: RecallConfig | None = None,
    store_raw: bool = False,
    db_path: str = ":memory:",
) -> Madeleine:
    config = MadeleineConfig(
        db_path=db_path, text_backend="hash", store_raw=store_raw, recall=recall or RecallConfig()
    )
    if kind == "memory":
        return create_engine(
            config,
            clock=clock,
            registry=fake_registry(),
            store=InMemoryStore(),
            queue=InMemoryRecollectionQueue(),
        )
    return create_engine(config, clock=clock, registry=fake_registry())


@pytest.fixture(params=["memory", "sqlite"])
def make_engine(request: pytest.FixtureRequest, clock: FakeClock) -> Iterator[EngineFactory]:
    made: list[Madeleine] = []

    def factory(**kwargs: object) -> Madeleine:
        eng = build_engine(request.param, clock, **kwargs)  # type: ignore[arg-type]
        made.append(eng)
        return eng

    yield factory
    for eng in made:
        eng.close()
