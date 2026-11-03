from __future__ import annotations

from pathlib import Path

import pytest

from madeleine.config import load_config, resolve_db_path


def test_env_and_overrides() -> None:
    cfg = load_config(
        {"MADELEINE_DB": ":memory:", "MADELEINE_TEXT_BACKEND": "hash", "MADELEINE_STORE_RAW": "1"}
    )
    assert cfg.db_path == ":memory:"
    assert cfg.text_backend == "hash"
    assert cfg.store_raw is True
    assert load_config({"MADELEINE_CHANNELS": "text, geo"}).channels == ("text", "geo")
    assert load_config({"MADELEINE_DB": "a.db"}, db_path="b.db").db_path == "b.db"
    with pytest.raises(ValueError, match="TEXT_BACKEND"):
        load_config({"MADELEINE_TEXT_BACKEND": "nope"})


def test_resolve_db_path(tmp_path: Path) -> None:
    assert resolve_db_path(":memory:") == ":memory:"
    target = resolve_db_path(str(tmp_path / "nested" / "dir" / "m.db"))
    assert Path(target) == tmp_path / "nested" / "dir" / "m.db"
    assert Path(target).parent.is_dir()
