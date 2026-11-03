"""Tool input models and the glue that hands MCP arguments to :mod:`madeleine.core.serde`.

All real parsing (timestamps, geohash / lat-lon, base64, signal shapes) lives in ``serde``;
this module only adds what is specific to the MCP surface: the JSON schema the model sees,
the local-file URI policy and the "one signal per channel" rule of a cue.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from madeleine.core import serde
from madeleine.core.types import ObservationError, RawSignal

__all__ = [
    "SignalInput",
    "check_unique_channels",
    "is_local_uri",
    "parse_signals",
]


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
            "steps_per_min, respiration_rate. Omit for image/audio sent as data_b64 or uri."
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
            "Where the media lives (https://...). Local file paths / file:// URIs are "
            "rejected unless the server was started with --allow-file-uris."
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
    """Apply the URI policy, then parse with :func:`serde.signals_from_json`."""
    dicts = [_as_dict(item) for item in items or ()]
    if not allow_file_uris:
        for d in dicts:
            uri = d.get("uri")
            if isinstance(uri, str) and is_local_uri(uri):
                raise ObservationError(
                    f"local file URIs are disabled on this server ({uri!r}); send the bytes "
                    "as data_b64, use an http(s) URI, or start the server with --allow-file-uris"
                )
    return serde.signals_from_json(dicts, now=now)


def check_unique_channels(signals: Sequence[RawSignal]) -> None:
    """A cue holds at most one observation per channel; refuse ambiguous input."""
    seen: set[str] = set()
    for s in signals:
        if s.channel in seen:
            raise ObservationError(f"more than one signal for channel {s.channel!r}")
        seen.add(s.channel)
