"""Unit tests for the MCP input glue."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from madeleine.core.types import ObservationError
from madeleine.mcp_server._inputs import (
    SignalInput,
    check_unique_channels,
    is_local_uri,
    parse_signals,
)

NOW = datetime(2026, 6, 1, 12, tzinfo=UTC)


@pytest.mark.parametrize(
    ("uri", "local"),
    [
        ("file:///etc/passwd", True),
        ("FILE://host/share/x.png", True),
        ("/home/me/a.wav", True),
        ("relative/a.png", True),
        ("~/a.png", True),
        ("C:\\Users\\me\\a.png", True),
        ("https://example.org/a.png", False),
        ("http://example.org/a.png", False),
        ("s3://bucket/a.png", False),
        ("data:image/png;base64,AAAA", False),
    ],
)
def test_is_local_uri(uri: str, local: bool) -> None:
    assert is_local_uri(uri) is local


def test_parse_signals_uses_serde_shapes() -> None:
    signals = parse_signals(
        [
            SignalInput(channel="geo", value=[48.8584, 2.2945]),
            {"channel": "image", "data_b64": "aGk=", "mime_type": "image/png"},
            SignalInput(channel="time", value="2026-06-01T21:30:00+02:00", at="2026-06-01"),
        ],
        now=NOW,
        allow_file_uris=False,
    )
    assert signals[0].value == (48.8584, 2.2945)
    assert signals[1].value == b"hi"
    assert signals[1].mime_type == "image/png"
    assert signals[2].observed_at == datetime(2026, 6, 1, tzinfo=UTC)
    assert parse_signals(None, now=NOW, allow_file_uris=False) == []


def test_file_uri_policy() -> None:
    items = [{"channel": "image", "uri": "file:///tmp/a.png"}]
    with pytest.raises(ObservationError, match="allow-file-uris"):
        parse_signals(items, now=NOW, allow_file_uris=False)
    assert parse_signals(items, now=NOW, allow_file_uris=True)[0].uri == "file:///tmp/a.png"


def test_unique_channels() -> None:
    sigs = parse_signals(
        [{"channel": "text", "value": "a"}, {"channel": "text", "value": "b"}],
        now=NOW,
        allow_file_uris=False,
    )
    with pytest.raises(ObservationError, match="more than one"):
        check_unique_channels(sigs)
