"""Tool input models and the glue that hands MCP arguments to :mod:`madeleine.core.serde`.

All real parsing (timestamps, geohash / lat-lon, base64, signal shapes) lives in ``serde``;
this module only adds what is specific to the MCP surface: the JSON schema the model sees,
an early local-file check and the "one signal per channel" rule of a cue.

The local-file policy itself is enforced by the image/audio adapters
(``MadeleineConfig.allow_local_files``); :func:`parse_signals` repeats it as defence in
depth so a refused request never reaches the engine.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from madeleine.core import serde
from madeleine.core.types import AUDIO, IMAGE, ObservationError, RawSignal

__all__ = [
    "LOCAL_FILES_DISABLED",
    "MEDIA_CHANNELS",
    "SignalInput",
    "check_unique_channels",
    "is_local_uri",
    "parse_signals",
]


MEDIA_CHANNELS: frozenset[str] = frozenset({IMAGE, AUDIO})
"""Channels whose string ``value`` is read as a file path by their adapters."""

LOCAL_FILES_DISABLED = (
    "local file paths / file:// URIs are disabled on this server; send the media bytes "
    "base64-encoded in data_b64 (madeleine never downloads remote URLs). The server operator "
    "can enable local files with 'serve --allow-file-uris'"
)
"""Refusal for a local file reference; constant, so it reveals nothing about the file."""


class SignalInput(BaseModel):
    """One raw reading on one channel (the shape ``serde.signal_from_json`` accepts)."""

    model_config = ConfigDict(extra="forbid")

    channel: str = Field(
        description=(
            "Channel name: 'text', 'geo', 'time', 'biometric', 'image' or 'audio' "
            "(whatever the server has enabled)."
        )
    )
    value: Any = Field(
        default=None,
        description=(
            "The reading. text: a string. geo: a geohash string or [lat, lon]. "
            "time: an ISO-8601 timestamp. biometric: an object of numbers, e.g. "
            '{"heart_rate": 92} (bpm), hrv (ms), skin_temp (deg C), eda (microsiemens), '
            "steps_per_min, respiration_rate. image/audio: omit it and send data_b64."
        ),
    )
    data_b64: str | None = Field(
        default=None,
        description="Raw media bytes (image/audio), base64-encoded. Takes precedence over value.",
    )
    mime_type: str | None = Field(
        default=None, description="MIME type of the media, e.g. 'image/png' or 'audio/wav'."
    )
    uri: str | None = Field(
        default=None,
        description=(
            "image/audio only: local file path or file:// URI of the media, only if the "
            "server was started with --allow-file-uris; otherwise send the bytes in "
            "data_b64. Remote URLs (https://...) are never fetched."
        ),
    )
    at: str | None = Field(
        default=None, description="When this reading was sensed (ISO 8601). Default: now."
    )


def is_local_uri(uri: str) -> bool:
    """Whether ``uri`` points at the local filesystem (``file://``, a bare or Windows path)."""
    scheme = urlsplit(uri.strip()).scheme.lower()
    return scheme in ("", "file") or len(scheme) == 1


def _as_dict(item: SignalInput | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(item, SignalInput):
        return item.model_dump(exclude_none=True)
    return dict(item)


def parse_signals(
    items: Iterable[SignalInput | Mapping[str, Any]] | None,
    *,
    now: datetime,
    allow_file_uris: bool,
) -> list[RawSignal]:
    """Apply the local-file policy, then parse with :func:`serde.signals_from_json`.

    When ``allow_file_uris`` is False, a local ``uri`` and a string ``value`` on a media
    channel (which the adapters read as a path) are refused with
    :data:`LOCAL_FILES_DISABLED`, without touching the filesystem.
    """
    dicts = [_as_dict(item) for item in items or ()]
    if not allow_file_uris:
        for d in dicts:
            uri = d.get("uri")
            if isinstance(uri, str) and is_local_uri(uri):
                raise ObservationError(LOCAL_FILES_DISABLED)
            if d.get("channel") in MEDIA_CHANNELS and isinstance(d.get("value"), str):
                raise ObservationError(LOCAL_FILES_DISABLED)
    return serde.signals_from_json(dicts, now=now)


def check_unique_channels(signals: Sequence[RawSignal]) -> None:
    """A cue holds at most one observation per channel; refuse ambiguous input."""
    seen: set[str] = set()
    for s in signals:
        if s.channel in seen:
            raise ObservationError(f"more than one signal for channel {s.channel!r}")
        seen.add(s.channel)
