"""The generic webhook :class:`~madeleine.core.types.CueAdapter` (``POST /cues/generic``).

Accepted JSON (every field optional, but the cue must contain at least one signal or a
location)::

    {
      "signals": [{"channel": "text", "value": "rain on a tin roof"},
                  {"channel": "image", "data_b64": "...", "mime_type": "image/png"}],
      "text": "smell of rain",                 # shortcut for a text signal
      "biometric": {"heart_rate": 92},         # shortcut for a biometric signal
      "at": "2026-06-01T21:30:00+02:00",       # ISO 8601; default: now
      "geohash": "u09tunq",                    # or "lat"/"lon"
      "lat": 48.8584, "lon": 2.2945,
      "tags": ["walk"],                        # list or comma-separated string
      "salience": 0.7
    }

This is exactly the body of ``POST /cues`` (whose ``source`` field is ignored here: cues from
this adapter have source ``"webhook:generic"``). Unknown keys are ignored so that third-party
senders can post their own payloads unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from madeleine.core import serde
from madeleine.core.types import Clock, ObservationError, RawSignal, SensoryContext, SystemClock
from madeleine.ingest import _fields as f

__all__ = ["GenericWebhookAdapter", "parse_generic_payload"]


def parse_generic_payload(
    payload: Mapping[str, Any], *, now: datetime
) -> tuple[list[RawSignal], SensoryContext | None]:
    """Parse the generic cue JSON into raw signals and a context (``None`` = engine "now").

    Raises :class:`ObservationError` for malformed fields or an empty cue.
    """
    raw_signals = payload.get("signals")
    if raw_signals is not None and not isinstance(raw_signals, list):
        raise ObservationError("'signals' must be a list of signal objects")
    items = raw_signals or []
    for item in items:
        if not isinstance(item, Mapping):
            raise ObservationError("each signal must be an object with a 'channel'")
    signals = serde.signals_from_json(items, now=now)
    signals = f.merge_convenience_signals(
        signals,
        text=f.as_str(payload.get("text"), "text"),
        biometric=f.as_biometric(payload.get("biometric")),
    )
    context = f.build_context(
        now=now,
        at=payload.get("at"),
        geohash=payload.get("geohash"),
        lat=payload.get("lat"),
        lon=payload.get("lon"),
        tags=f.as_tags(payload.get("tags")),
        salience=payload.get("salience"),
    )
    f.require_some_signal(signals, context)
    return signals, context


class GenericWebhookAdapter:
    """Adapter ``generic``: the ``POST /cues`` JSON shape, posted by any webhook sender.

    ``clock`` supplies "now" when the payload gives context (tags, location) but no ``at``.
    """

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock: Clock = clock or SystemClock()

    @property
    def name(self) -> str:
        return "generic"

    def to_signals(
        self, payload: Mapping[str, Any]
    ) -> tuple[list[RawSignal], SensoryContext | None]:
        """Map the payload (see module docstring)."""
        return parse_generic_payload(payload, now=self._clock.now())
