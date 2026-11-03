"""MCP cannot make the server read or probe local files (R2/R3 regression tests).

Covers every bypass from the review: ``uri``, a string ``value`` on image/audio, and a
missing file giving the same answer as an existing one; plus the ``allow_file_uris``
default (the engine's ``allow_local_files``) and the adapters as the final authority.
"""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from madeleine.channels.registry import hash_registry
from madeleine.core.types import MadeleineConfig, ObservationError
from madeleine.engine import Madeleine, create_engine
from madeleine.mcp_server import create_server
from madeleine.mcp_server._inputs import LOCAL_FILES_DISABLED, parse_signals
from madeleine.testing import FakeEngine
from tests.mcp_server.mcp_helpers import call, call_error
from tests.mcp_server.test_mcp_inputs import NOW

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _engine(*, allow: bool) -> Madeleine:
    return create_engine(
        MadeleineConfig(db_path=":memory:", text_backend="hash", allow_local_files=allow),
        registry=hash_registry(allow_local_files=allow),
    )


@pytest.fixture
def secret(tmp_path: Path) -> Path:
    f = tmp_path / "secret.png"
    f.write_bytes(PNG)
    return f


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    real = Path.read_bytes

    def spy(self: Path) -> bytes:
        seen.append(str(self))
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", spy)
    return seen


def _bare(message: str) -> str:
    """Drop the SDK's "Error executing tool <name>: " and the adapter's "<channel>: "."""
    message = re.sub(r"^Error executing tool \w+: ", "", message)
    return re.sub(r"^(image|audio): ", "", message)


def _bypasses(secret: Path) -> list[tuple[str, dict[str, Any]]]:
    missing = secret.parent / "missing.png"
    out: list[tuple[str, dict[str, Any]]] = []
    for sig in (
        {"channel": "image", "uri": secret.as_uri()},
        {"channel": "image", "uri": missing.as_uri()},
        {"channel": "image", "value": str(secret)},
        {"channel": "image", "value": str(missing)},
        {"channel": "audio", "value": "/etc/hostname"},
        {"channel": "audio", "value": "/nonexistent/x.wav"},
        {"channel": "image", "data_b64": base64.b64encode(PNG).decode(), "uri": str(secret)},
    ):
        out.append(("remember", {"content": "x", "signals": [sig]}))
        out.append(("sense", {"signals": [sig]}))
    return out


@pytest.mark.parametrize("server_flag", [None, False])
async def test_no_local_file_bypass_over_mcp(
    secret: Path, reads: list[str], server_flag: bool | None
) -> None:
    engine = _engine(allow=False)
    try:
        async with Client(create_server(engine, allow_file_uris=server_flag)) as client:
            messages = {
                _bare(await call_error(client, tool, args)) for tool, args in _bypasses(secret)
            }
            assert messages == {LOCAL_FILES_DISABLED}
            assert engine.store.count() == 0
    finally:
        engine.close()
    assert reads == []


async def test_adapters_refuse_even_if_the_mcp_guard_is_opened(
    secret: Path, reads: list[str]
) -> None:
    """``allow_file_uris=True`` on the server cannot grant what the engine refuses."""
    engine = _engine(allow=False)
    try:
        async with Client(create_server(engine, allow_file_uris=True)) as client:
            messages = {
                _bare(await call_error(client, tool, args)) for tool, args in _bypasses(secret)
            }
            assert len(messages) == 1  # identical for existing and missing files
            assert "disabled" in messages.pop()
    finally:
        engine.close()
    assert reads == []


async def test_default_follows_engine_config(secret: Path) -> None:
    engine = _engine(allow=True)
    try:
        server = create_server(engine)
        assert server.allow_file_uris is True
        async with Client(server) as client:
            mem = await call(
                client,
                "remember",
                {"content": "x", "signals": [{"channel": "image", "value": str(secret)}]},
            )
            assert "image" in mem["channels"]
            missing = {"channel": "image", "uri": (secret.parent / "nope.png").as_uri()}
            assert "file not found" in await call_error(client, "sense", {"signals": [missing]})
    finally:
        engine.close()
    assert create_server(FakeEngine()).allow_file_uris is False


def test_parse_signals_rejects_media_string_values() -> None:
    for channel in ("image", "audio"):
        for value in ("/etc/passwd", "relative.png", "https://example.org/a.png"):
            with pytest.raises(ObservationError) as info:
                parse_signals(
                    [{"channel": channel, "value": value}], now=NOW, allow_file_uris=False
                )
            assert str(info.value) == LOCAL_FILES_DISABLED
    # Non-media string values are untouched; media values pass when files are allowed.
    assert parse_signals(
        [{"channel": "text", "value": "/etc/passwd"}], now=NOW, allow_file_uris=False
    )
    assert parse_signals([{"channel": "image", "value": "/a.png"}], now=NOW, allow_file_uris=True)


def test_refusal_message_does_not_advertise_remote_uris() -> None:
    """R19: models follow the wording; remote URIs never work."""
    assert "http(s)" not in LOCAL_FILES_DISABLED
    assert "data_b64" in LOCAL_FILES_DISABLED
    assert "--allow-file-uris" in LOCAL_FILES_DISABLED


async def test_uri_field_description(secret: Path) -> None:
    """R19: the schema the model reads says what actually works."""
    async with Client(create_server(FakeEngine())) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}
        schema = tools["remember"].input_schema
        defs = schema.get("$defs") or schema.get("definitions") or {}
        desc = defs["SignalInput"]["properties"]["uri"]["description"]
        assert "only if the server was started with --allow-file-uris" in desc
        assert "otherwise send the bytes in data_b64" in desc
        assert "(https://...)" not in desc.split("Remote")[0]
