"""Shared parsing of external (JSON / CLI) inputs into contract types.

The MCP server, the HTTP API, webhook adapters and the CLI all accept the same field names
(``at``, ``geohash``, ``lat``/``lon``, ``tags``, ``salience``, ``signals``). They must
parse them through these functions so that behaviour is identical everywhere.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from madeleine.core import geohash as gh
from madeleine.core.types import ObservationError, RawSignal, SensoryContext

__all__ = [
    "DEFAULT_GEOHASH_PRECISION",
    "context_from_fields",
    "parse_timestamp",
    "signal_from_json",
    "signals_from_json",
]

DEFAULT_GEOHASH_PRECISION = 9
"""Precision used when converting lat/lon to a geohash (~5 m cells)."""


def parse_timestamp(value: str | datetime | None, *, default: datetime) -> datetime:
    """Parse an ISO-8601 timestamp.

    * ``None``/``""`` -> ``default``.
    * A trailing ``Z`` is accepted.
    * A **naive** value (no offset) is interpreted as UTC. Callers that care about
      "hour of day" should send their local offset (``2026-06-01T21:30:00+02:00``).
    """
    if value is None or value == "":
        return default
    if isinstance(value, datetime):
        dt = value
    else:
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ObservationError(f"invalid ISO-8601 timestamp {value!r}") from exc
    if dt.tzinfo is None or dt.utcoffset() is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def context_from_fields(
    *,
    now: datetime,
    at: str | datetime | None = None,
    geohash: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    tags: Iterable[str] | None = None,
    salience: float | None = None,
) -> SensoryContext:
    """Build a :class:`SensoryContext` from loose fields.

    ``geohash`` wins over ``lat``/``lon``; ``lat`` and ``lon`` must come together and are
    encoded at :data:`DEFAULT_GEOHASH_PRECISION`.
    """
    if (lat is None) != (lon is None):
        raise ObservationError("lat and lon must be given together")
    resolved: str | None = None
    if geohash:
        if not gh.is_valid(geohash):
            raise ObservationError(f"invalid geohash {geohash!r}")
        resolved = geohash.lower()
    elif lat is not None and lon is not None:
        try:
            resolved = gh.encode(float(lat), float(lon), DEFAULT_GEOHASH_PRECISION)
        except ValueError as exc:
            raise ObservationError(str(exc)) from exc
    try:
        return SensoryContext(
            timestamp=parse_timestamp(at, default=now),
            geohash=resolved,
            tags=frozenset(t.strip() for t in (tags or ()) if t.strip()),
            salience=0.5 if salience is None else float(salience),
        )
    except ValueError as exc:
        raise ObservationError(str(exc)) from exc


def signal_from_json(obj: Mapping[str, Any], *, now: datetime | None = None) -> RawSignal:
    """Parse one signal object.

    Shape::

        {"channel": "text", "value": "rain on a tin roof"}
        {"channel": "geo", "value": [48.8584, 2.2945]}        # or "u09tunq"
        {"channel": "biometric", "value": {"heart_rate": 92}}
        {"channel": "image", "data_b64": "...", "mime_type": "image/png"}
        {"channel": "audio", "uri": "https://...", "mime_type": "audio/wav"}
        {"channel": "time", "value": "2026-06-01T21:30:00+02:00"}

    ``data_b64`` is decoded to ``bytes`` and becomes the value. JSON lists become tuples.
    An optional ``at`` sets ``observed_at``.
    """
    channel = obj.get("channel")
    if not isinstance(channel, str) or not channel:
        raise ObservationError("signal needs a non-empty string 'channel'")
    value: object = obj.get("value")
    if "data_b64" in obj and obj["data_b64"] is not None:
        raw = obj["data_b64"]
        if not isinstance(raw, str):
            raise ObservationError("'data_b64' must be a base64 string")
        try:
            value = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ObservationError("'data_b64' is not valid base64") from exc
    elif isinstance(value, list):
        value = tuple(value)
    mime_type = obj.get("mime_type")
    uri = obj.get("uri")
    at = obj.get("at")
    observed_at = parse_timestamp(at, default=now) if at and now is not None else None
    if at and now is None:
        observed_at = parse_timestamp(at, default=datetime.now(UTC))
    try:
        return RawSignal(
            channel=channel,
            value=value,
            mime_type=mime_type if isinstance(mime_type, str) else None,
            uri=uri if isinstance(uri, str) else None,
            observed_at=observed_at,
        )
    except ValueError as exc:
        raise ObservationError(str(exc)) from exc


def signals_from_json(
    items: Sequence[Mapping[str, Any]] | None, *, now: datetime | None = None
) -> list[RawSignal]:
    """Parse a list of signal objects (``None`` -> ``[]``)."""
    return [signal_from_json(item, now=now) for item in (items or ())]
