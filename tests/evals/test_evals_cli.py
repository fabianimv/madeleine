from __future__ import annotations

import json
from pathlib import Path

import pytest

from madeleine.core.types import MemoryEngine, RecallConfig
from madeleine.evals import cli
from madeleine.evals.runner import BenchReport, run_bench
from madeleine.testing import FakeEngine


def _fake(rc: RecallConfig) -> MemoryEngine:
    return FakeEngine(trigger_threshold=rc.trigger_threshold, refractory=rc.refractory_period)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    seen: list[dict[str, object]] = []

    def fake_run_bench(**kwargs: object) -> BenchReport:
        seen.append(kwargs)
        return run_bench(engine_factory=_fake, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cli, "run_bench", fake_run_bench)
    return seen


def test_markdown_to_stdout(
    calls: list[dict[str, object]], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["--memories", "30", "--cues", "18", "--seed", "2"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# madeleine benchmark")
    assert calls == [{"memories": 30, "cues": 18, "seed": 2, "thresholds": None}]


def test_json_output_file_and_sweep(
    calls: list[dict[str, object]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "out" / "bench.json"
    code = cli.main(
        ["--memories", "30", "--cues", "18", "--json", "--sweep", "--output", str(path)]
    )
    assert code == 0
    assert capsys.readouterr().out == ""
    data = json.loads(path.read_text())
    assert len(data["sweep"]) == 9
    assert calls[0]["thresholds"] is not None


def test_bad_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--memories", "0"]) == 2
    assert "error" in capsys.readouterr().err


def test_missing_engine_reports_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom(**kwargs: object) -> BenchReport:
        raise ModuleNotFoundError("No module named 'madeleine.engine'")

    monkeypatch.setattr(cli, "run_bench", boom)
    assert cli.main([]) == 1
    assert "not available" in capsys.readouterr().err
