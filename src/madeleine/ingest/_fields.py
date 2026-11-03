"""Lenient coercion of loose webhook fields (shared by every :class:`CueAdapter`).

Webhook senders (Home Assistant templates, iOS Shortcuts) often send numbers as strings,
``""`` for "unset", or ``"None"`` when a template variable was empty. These helpers turn such
values into clean Python values or raise :class:`ObservationError` with a readable message.
Timestamp, geohash, context and signal parsing itself is delegated to
:mod:`madeleine.core.serde` (never re-implemented here).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from typing import Any

from madeleine.core import serde
from madeleine.core.types import BIOMETRIC, TEXT, ObservationError, RawSignal, SensoryContext

__all__ = [
    "MISSING_STRINGS",
    "as_biometric",
    "as_float",
    "as_str",
    "as_tags",
    "build_context",
    "is_missing",
    "merge_convenience_signals",
    "require_mapping",
    "require_some_signal",
]

MISSING_STRINGS: frozenset[str] = frozenset({"", "none", "null", "unknown", "unavailable"})
"""String values treated as "not provided" (what empty template variables render to)."""


def is_missing(value: object) -> bool:
    """``None`` or a blank / placeholder string (see :data:`MISSING_STRINGS`)."""
    return value is None or (isinstance(value, str) and value.strip().lower() in MISSING_STRINGS)


def as_str(value: object, field: str) -> str | None:
    """A stripped, non-empty string, or ``None`` when missing."""
    if is_missing(value):
        return None
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ObservationError(f"'{field}' must be a string")
    return str(value).strip()


def as_float(value: object, field: str) -> float | None:
    """A finite float, accepting numeric strings (``"48.85"``, ``"48,85"``); ``None`` if missing."""
    if is_missing(value):
        return None
    if isinstance(value, bool):
        raise ObservationError(f"'{field}' must be a number")
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        text = value.strip()
        if "," in text and "." not in text:
            text = text.replace(",", ".")  # locale decimal comma (e.g. iOS in fr_FR)
        try:
            number = float(text)
        except ValueError:
            raise ObservationError(f"'{field}' must be a number, got {value!r}") from None
    else:
        raise ObservationError(f"'{field}' must be a number")
    if not math.isfinite(number):
        raise ObservationError(f"'{field}' must be finite")
    return number


def as_tags(value: object, field: str = "tags") -> list[str]:
    """A list of tags from a JSON list or a comma-separated string."""
    if is_missing(value):
        return []
    items: Iterable[object]
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, Sequence):
        items = value
    else:
        raise ObservationError(f"'{field}' must be a list of strings")
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise ObservationError(f"'{field}' must be a list of strings")
        if item.strip():
            out.append(item.strip())
    return out


def as_biometric(value: object, field: str = "biometric") -> dict[str, float]:
    """A mapping of biometric feature name -> finite float (missing values dropped)."""
    if is_missing(value):
        return {}
    if not isinstance(value, Mapping):
        raise ObservationError(f"'{field}' must be an object of numbers")
    out: dict[str, float] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise ObservationError(f"'{field}' keys must be non-empty strings")
        number = as_float(raw, f"{field}.{key}")
        if number is not None:
            out[key] = number
    return out


def require_mapping(value: object, field: str) -> Mapping[str, Any]:
    """``value`` as a mapping (``{}`` when missing)."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ObservationError(f"'{field}' must be an object")
    return value


def build_context(
    *,
    now: datetime,
    at: object = None,
    geohash: object = None,
    lat: object = None,
    lon: object = None,
    tags: Iterable[str] = (),
    salience: object = None,
) -> SensoryContext | None:
    """Build a context through :func:`madeleine.core.serde.context_from_fields`.

    Returns ``None`` when no context field was given, so that the engine applies its own
    clock ("now, no location") instead of the caller's.
    """
    at_s = as_str(at, "at")
    geohash_s = as_str(geohash, "geohash")
    lat_f = as_float(lat, "lat")
    lon_f = as_float(lon, "lon")
    salience_f = as_float(salience, "salience")
    tag_list = list(tags)
    given = (at_s, geohash_s, lat_f, lon_f, salience_f)
    if all(v is None for v in given) and not tag_list:
        return None
    return serde.context_from_fields(
        now=now,
        at=at_s,
        geohash=geohash_s,
        lat=lat_f,
        lon=lon_f,
        tags=tag_list,
        salience=salience_f,
    )


def merge_convenience_signals(
    signals: list[RawSignal], *, text: str | None, biometric: Mapping[str, float]
) -> list[RawSignal]:
    """Append ``text`` / ``biometric`` convenience fields, refusing duplicates of a channel."""
    present = {s.channel for s in signals}
    out = list(signals)
    if text is not None:
        if TEXT in present:
            raise ObservationError("text given both as 'text' and inside 'signals'")
        out.append(RawSignal(TEXT, text))
    if biometric:
        if BIOMETRIC in present:
            raise ObservationError("biometric given both as 'biometric' and inside 'signals'")
        out.append(RawSignal(BIOMETRIC, dict(biometric)))
    seen: set[str] = set()
    for s in out:
        if s.channel in seen:
            raise ObservationError(f"more than one signal on channel {s.channel!r}")
        seen.add(s.channel)
    return out


def require_some_signal(signals: Sequence[RawSignal], context: SensoryContext | None) -> None:
    """A cue needs at least one signal or a location (a bare "now" never triggers anything)."""
    if not signals and (context is None or context.geohash is None):
        raise ObservationError("the cue is empty: send text, signals, biometric data or a location")
