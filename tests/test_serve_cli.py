"""Lead-owned composition: server bind policy and CLI error handling."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from madeleine.cli import app
from madeleine.serve import InsecureBindError, check_bind

runner = CliRunner()


def test_loopback_bind_needs_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MADELEINE_API_TOKEN", raising=False)
    for host in ("127.0.0.1", "localhost", "::1"):
        check_bind(host)


def test_public_bind_requires_token_or_explicit_opt_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MADELEINE_API_TOKEN", raising=False)
    with pytest.raises(InsecureBindError, match="MADELEINE_API_TOKEN"):
        check_bind("0.0.0.0")
    check_bind("0.0.0.0", insecure_no_auth=True)
    monkeypatch.setenv("MADELEINE_API_TOKEN", "s3cret")
    check_bind("0.0.0.0")


def test_cli_refuses_public_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MADELEINE_API_TOKEN", raising=False)
    result = runner.invoke(app, ["serve", "--transport", "http", "--host", "0.0.0.0"])
    assert result.exit_code == 1
    assert "refusing to bind" in result.output


def test_cli_reports_bad_config_without_traceback(tmp_path: object) -> None:
    result = runner.invoke(
        app, ["remember", "x", "--db", f"{tmp_path}/m.db", "--text-backend", "bogus"]
    )
    assert result.exit_code == 1
    assert "text_backend must be one of" in result.output
    assert "Traceback" not in result.output


def test_cli_recall_lat_without_lon(tmp_path: object) -> None:
    result = runner.invoke(app, ["recall", "--lon", "3", "--db", f"{tmp_path}/m.db"])
    assert result.exit_code == 1
    assert "--lat and --lon must be given together" in result.output


def test_cli_round_trip(tmp_path: object) -> None:
    db = f"{tmp_path}/m.db"
    env = {"MADELEINE_TEXT_BACKEND": "hash"}
    out = runner.invoke(
        app,
        ["remember", "tea in grandmother's kitchen", "--geohash", "u09tunq", "--db", db],
        env=env,
    )
    assert out.exit_code == 0, out.output
    fired = runner.invoke(app, ["sense", "--geohash", "u09tunq", "--db", db], env=env)
    assert fired.exit_code == 0, fired.output
    assert "tea in grandmother's kitchen" in fired.output
    pending = runner.invoke(app, ["pending", "--db", db], env=env)
    assert "tea in grandmother's kitchen" in pending.output


def test_allow_file_uris_flag_reaches_engine_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import madeleine.engine
    import madeleine.mcp_server.server
    import madeleine.serve as serve_mod
    from madeleine.testing import FakeEngine

    seen: dict[str, object] = {}

    def fake_create_engine(config: object) -> FakeEngine:
        seen["config"] = config
        return FakeEngine()

    def fake_run_stdio(engine: object, **kwargs: object) -> None:
        seen["kwargs"] = kwargs

    monkeypatch.setattr(madeleine.engine, "create_engine", fake_create_engine)
    monkeypatch.setattr(madeleine.mcp_server.server, "run_stdio", fake_run_stdio)
    monkeypatch.delenv("MADELEINE_ALLOW_LOCAL_FILES", raising=False)

    serve_mod.serve(transport="stdio", db=":memory:")
    assert seen["config"].allow_local_files is False  # type: ignore[attr-defined]
    serve_mod.serve(transport="stdio", db=":memory:", allow_file_uris=True)
    assert seen["config"].allow_local_files is True  # type: ignore[attr-defined]
    assert seen["kwargs"] == {"allow_file_uris": True}


def test_cli_show_and_edit(tmp_path: object) -> None:
    import json as _json

    db = f"{tmp_path}/m.db"
    env = {"MADELEINE_TEXT_BACKEND": "hash"}
    created = runner.invoke(app, ["remember", "tea at the tower", "--db", db], env=env)
    mid = _json.loads(created.output)["id"]
    shown = runner.invoke(app, ["show", mid, "--db", db], env=env)
    assert shown.exit_code == 0
    assert _json.loads(shown.output)["content"] == "tea at the tower"
    edited = runner.invoke(
        app,
        ["edit", mid, "--content", "espresso", "--tag", "cafe", "--salience", "0.8", "--db", db],
        env=env,
    )
    assert edited.exit_code == 0, edited.output
    out = _json.loads(edited.output)
    assert out["content"] == "espresso"
    assert out["context"]["tags"] == ["cafe"]
    assert out["context"]["salience"] == 0.8
    missing = runner.invoke(app, ["show", "mem_nope", "--db", db], env=env)
    assert missing.exit_code == 1
    assert "no memory" in missing.output
    bad = runner.invoke(app, ["edit", mid, "--salience", "3", "--db", db], env=env)
    assert bad.exit_code == 1
    assert "salience" in bad.output
