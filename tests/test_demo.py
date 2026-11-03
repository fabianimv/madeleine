"""The definition-of-done scenario: `madeleine demo` runs offline end to end."""

from __future__ import annotations

from pathlib import Path

import pytest

from madeleine.demo import run_demo


def test_demo_delivers_involuntary_recollection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_demo(db_path=str(tmp_path / "demo.db")) == 0
    out = capsys.readouterr().out
    assert "via geo+time] Sunday afternoons in grandmother's kitchen" in out
    assert "<- push recollection" in out
    assert "fired: 0" in out
    assert "Done: the memory came back without being asked for." in out
