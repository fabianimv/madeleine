"""Regression tests for the pre-release review findings on the store and engine."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from madeleine.core.types import MadeleineConfig, RawSignal, SensoryContext
from madeleine.engine import create_engine
from madeleine.store import SQLiteDatabase, SQLiteMemoryStore, open_sqlite
from madeleine.testing import FakeClock

from .fakes import fake_registry

T = datetime(2026, 6, 1, 20, 0, tzinfo=UTC)
PARIS = "u09tunq"


def _db_files(path: Path) -> list[Path]:
    return [p for p in path.parent.iterdir() if p.name.startswith(path.name)]


# -- R7: forget leaves no recoverable bytes ---------------------------------------------
@pytest.mark.parametrize("use_vec", [True, False])
def test_r7_forget_leaves_no_bytes_in_db_files(
    tmp_path: Path, clock: FakeClock, use_vec: bool
) -> None:
    path = tmp_path / "secure.db"
    content = "CONTENT-MARKER-7f3a9c the secret we must forget"
    raw = b"RAWPIXELS-MARKER-" + bytes(range(256)) * 8
    store, queue = open_sqlite(path, use_sqlite_vec=use_vec)
    eng = create_engine(
        MadeleineConfig(db_path=str(path)),
        clock=clock,
        registry=fake_registry(),
        store=store,
        queue=queue,
    )
    keep = eng.remember("an unrelated memory that stays", context=SensoryContext(T))
    m = eng.remember(
        content,
        [RawSignal("image", raw, mime_type="image/png")],
        context=SensoryContext(T, geohash=PARIS),
        store_raw=True,
    )
    assert eng.sense(eng.build_cue(context=SensoryContext(T, geohash=PARIS)))
    assert eng.forget(m.id)
    eng.close()
    store.close()
    queue.close()
    blob = b"".join(p.read_bytes() for p in _db_files(path))
    assert b"CONTENT-MARKER-7f3a9c" not in blob
    assert b"RAWPIXELS-MARKER-" not in blob
    reopened = SQLiteMemoryStore(path, use_sqlite_vec=use_vec)
    assert [x.id for x in reopened.iter_memories()] == [keep.id]
    reopened.close()


def test_r7_secure_delete_pragma_on_every_connection(tmp_path: Path) -> None:
    db = SQLiteDatabase(tmp_path / "p.db")
    with db.read() as conn:
        assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
    db.acquire().release()
