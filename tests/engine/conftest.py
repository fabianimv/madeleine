from __future__ import annotations

import os
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


def _sqlite_vec_loadable() -> bool:
    """Some Python builds (e.g. certain macOS ones) compile sqlite3 without extension loading."""
    import sqlite3

    try:
        import sqlite_vec

        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.close()
    except Exception:
        return False
    return True


# MADELEINE_TEST_NO_SQLITE_VEC=1 reproduces such a Python locally (skips only; the store
# itself still loads the extension).
SQLITE_VEC_LOADABLE = _sqlite_vec_loadable() and not os.environ.get("MADELEINE_TEST_NO_SQLITE_VEC")

# Tests that need the real sqlite-vec extension (the numpy fallback is covered separately).
_NEEDS_SQLITE_VEC = (
    "[sqlite-vec]",
    "test_sqlite_vec_and_fallback_give_same_nearest",
    "test_fallback_writes_are_reindexed_by_vec_connection",
    "test_raw_bytes_only_persisted_when_present[True]",
    "test_r7_forget_leaves_no_bytes_in_db_files[True]",
    "test_r14_vec_connection_sees_fallback_writes_without_reopen",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if SQLITE_VEC_LOADABLE:
        return
    skip = pytest.mark.skip(reason="this Python's sqlite3 cannot load extensions (sqlite-vec)")
    for item in items:
        if any(marker in item.nodeid for marker in _NEEDS_SQLITE_VEC):
            item.add_marker(skip)
