"""iOS Shortcuts :class:`~madeleine.core.types.CueAdapter` (``POST /cues/ios_shortcuts``).

The payload is what a Shortcut builds with **Get Contents of URL** (Method ``POST``,
Request Body ``JSON``); every field is optional but the cue needs at least one signal or a
location (recipe: ``examples/ios_shortcuts/README.md``)::

    {
      "text": "Back at the old bakery",               # Text / Dictate Text
      "latitude": 48.8584, "longitude": 2.2945,       # Current Location (alias lat/lon)
      "date": "2026-06-01T21:30:00+02:00",            # Format Date -> ISO 8601 (alias at)
      "heart_rate": 72,                               # Find Health Samples (Heart Rate)
      "photo_b64": "...", "photo_mime_type": "image/jpeg",   # Base64 Encode of a photo
      "audio_b64": "...", "audio_mime_type": "audio/mp4",    # Base64 Encode of a recording
      "focus": "Work",                                # current Focus -> tag "focus:work"
      "battery": 15,                                  # 0-100 or 0-1; <= 20 % -> tag "battery:low"
      "tags": ["commute"],
      "biometric": {"hrv": 41}                        # any extra biometric features
    }

* Numbers may be sent as numbers or numeric strings (``"48,8584"`` is accepted). Empty
  strings mean "not provided" (unset Shortcut variables).
* ``date`` **must be ISO 8601**: in the Shortcut use *Format Date* with *Date Format: ISO
  8601* (and *Include ISO 8601 Time*). Locale formats such as ``"1 Jun 2026 at 21:30"`` are
  rejected with HTTP 422 because they carry no UTC offset and are ambiguous.
* ``photo_b64`` / ``audio_b64`` may contain line breaks (Base64 Encode's default) or be a
  ``data:<mime>;base64,`` URI. Raw bytes are only embedded, never stored (unless the server
  stores raw media). Convert HEIC photos to JPEG in the Shortcut ("Convert Image").
* Unknown keys are ignored.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from madeleine.core.types import (
    AUDIO,
    IMAGE,
    Clock,
    ObservationError,
    RawSignal,
    SensoryContext,
    SystemClock,
)
from madeleine.ingest import _fields as f
from madeleine.ingest.webhook import parse_generic_payload

__all__ = ["IOSShortcutsAdapter"]

_DEFAULT_MIME = {IMAGE: "image/jpeg", AUDIO: "audio/mp4"}
LOW_BATTERY = 0.2


def _media_signal(channel: str, b64: object, mime: object, field: str) -> dict[str, Any] | None:
    text = f.as_str(b64, field)
    if text is None:
        return None
    mime_type = f.as_str(mime, f"{field[:-4]}_mime_type")
    if text.startswith("data:"):
        header, sep, text = text.partition(",")
        if not sep or ";base64" not in header:
            raise ObservationError(f"'{field}' data URI must be base64-encoded")
        mime_type = mime_type or header[5:].split(";", 1)[0] or None
    cleaned = "".join(text.split())  # Base64 Encode inserts line breaks by default
    return {
        "channel": channel,
        "data_b64": cleaned,
        "mime_type": mime_type or _DEFAULT_MIME[channel],
    }


def _first(payload: Mapping[str, Any], *keys: str) -> object:
    for key in keys:
        if not f.is_missing(payload.get(key)):
            return payload.get(key)
    return None


class IOSShortcutsAdapter:
    """Adapter ``ios_shortcuts`` (payload documented in the module docstring)."""

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock: Clock = clock or SystemClock()

    @property
    def name(self) -> str:
        return "ios_shortcuts"

    def to_signals(
        self, payload: Mapping[str, Any]
    ) -> tuple[list[RawSignal], SensoryContext | None]:
        """Map a Shortcuts payload to signals and context."""
        signals: list[dict[str, Any]] = []
        for channel, key in ((IMAGE, "photo"), (AUDIO, "audio")):
            media = _media_signal(
                channel, payload.get(f"{key}_b64"), payload.get(f"{key}_mime_type"), f"{key}_b64"
            )
            if media is not None:
                signals.append(media)

        biometric = f.as_biometric(payload.get("biometric"))
        heart_rate = f.as_float(payload.get("heart_rate"), "heart_rate")
        if heart_rate is not None:
            biometric["heart_rate"] = heart_rate

        tags = f.as_tags(payload.get("tags"))
        focus = f.as_str(payload.get("focus"), "focus")
        if focus:
            tags.append(f"focus:{focus.lower()}")
        battery = f.as_float(payload.get("battery"), "battery")
        if battery is not None:
            level = battery / 100.0 if battery > 1.0 else battery
            if level <= LOW_BATTERY:
                tags.append("battery:low")

        at = _first(payload, "date", "at")
        if isinstance(at, str):
            at = at.strip()
            if not _looks_iso(at):
                raise ObservationError(
                    f"'date' must be ISO 8601 (got {at!r}): in the Shortcut use "
                    "'Format Date' with 'Date Format: ISO 8601'"
                )
        generic: dict[str, Any] = {
            "signals": signals,
            "text": payload.get("text"),
            "biometric": biometric or None,
            "at": at,
            "geohash": payload.get("geohash"),
            "lat": _first(payload, "latitude", "lat"),
            "lon": _first(payload, "longitude", "lon"),
            "tags": tags,
            "salience": payload.get("salience"),
        }
        return parse_generic_payload(generic, now=self._clock.now())


def _looks_iso(value: str) -> bool:
    """Cheap pre-check so locale dates get a Shortcut-specific error message."""
    return len(value) >= 10 and value[:4].isdigit() and value[4] == "-"
