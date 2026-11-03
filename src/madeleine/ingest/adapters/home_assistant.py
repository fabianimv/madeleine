"""Home Assistant :class:`~madeleine.core.types.CueAdapter` (``POST /cues/home_assistant``).

Supported payload shapes
========================

1. **Flat state change** (what a ``rest_command`` built from an automation's ``trigger``
   sends; see ``examples/home_assistant/configuration.yaml``)::

       {"event": "state_changed",             # optional; "zone.enter"/"zone.leave" too
        "entity_id": "media_player.kitchen",
        "state": "playing",
        "old_state": "idle",                  # optional, a state string
        "attributes": {"friendly_name": "Kitchen speaker", "media_title": "Clair de lune",
                       "media_artist": "Debussy", "unit_of_measurement": "bpm", ...},
        "latitude": 48.85, "longitude": 2.29, # optional (also read from attributes)
        "zone": "Home",                       # optional zone name or "zone.home"
        "time_fired": "2026-06-01T21:30:00+02:00"}   # optional (alias "at")

2. **Native event** (``state_changed`` as forwarded by Node-RED / AppDaemon / the
   websocket API)::

       {"event_type": "state_changed", "time_fired": "...",
        "data": {"entity_id": "...", "new_state": {"state": "...", "attributes": {...},
                                                   "last_changed": "..."},
                 "old_state": {"state": "..."}}}

3. **Any of the generic fields** (``text``, ``signals``, ``biometric``, ``tags``, ``at``,
   ``geohash``, ``lat``/``lon``, ``salience``) may be added to either shape; an explicit
   ``text`` replaces the generated description.

Mapping
=======

* ``text`` signal: a sentence describing the change, by domain:
  ``media_player`` ("Kitchen speaker started playing Clair de lune by Debussy"),
  ``person``/``device_tracker`` ("Ana arrived home", "Ana left Work", "Ana arrived at Gym"),
  ``light``/``switch``/``binary_sensor``/``input_boolean``/``fan``/``cover``/``lock``
  ("Hallway light turned on"), ``weather`` ("The weather is rainy"), ``sun``
  ("The sun set"), ``sensor`` ("Living room temperature is 21.5 °C"), anything else
  ("<name> changed to <state>"), and non-state events ("Home Assistant event: <event>").
* **biometric** signal: a ``sensor`` whose entity id / device class / unit identifies a
  wearable reading - ``heart_rate`` (unit ``bpm``), ``hrv``, ``respiration_rate``,
  ``skin_temp`` (only *skin*/*body* temperature; °F converted to °C), ``eda``,
  ``steps_per_min``. Such sensors produce no text. Ambient temperature is *not* biometric.
  Note that ``time`` and ``biometric`` cannot trigger a recollection alone: send a location
  with wearable readings to make them useful.
* **geo**: from ``latitude``/``longitude`` (top level, else the entity's attributes).
* **time**: from ``time_fired`` (or ``at``; native events also ``new_state.last_changed``).
  Use a local-offset timestamp (``{{ now().isoformat() }}``): HA's own ``time_fired`` is in
  UTC, and "hour of day" is computed in the timestamp's offset.

Placeholder values that HA templates render for missing attributes (``"None"``,
``"unknown"``, ``"unavailable"``, ``""``) are treated as absent.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from madeleine.core.types import Clock, ObservationError, RawSignal, SensoryContext, SystemClock
from madeleine.ingest import _fields as f
from madeleine.ingest.webhook import parse_generic_payload

__all__ = ["HomeAssistantAdapter", "describe_state_change"]

_ON_OFF_DOMAINS = frozenset(
    {"light", "switch", "binary_sensor", "input_boolean", "fan", "cover", "lock", "climate"}
)
_STOPPED_MEDIA = frozenset({"idle", "off", "standby", "stopped"})


def _pretty(name: str) -> str:
    """``"kitchen_speaker"`` / ``"zone.work"`` -> ``"Kitchen speaker"`` / ``"Work"``."""
    base = name.split(".", 1)[1] if "." in name else name
    words = base.replace("_", " ").strip()
    return words[:1].upper() + words[1:] if words else name


def _attr(attributes: Mapping[str, Any], key: str) -> str | None:
    value = attributes.get(key)
    if f.is_missing(value) or isinstance(value, Mapping | list):
        return None
    return str(value).strip()


def _biometric_feature(entity_id: str, attributes: Mapping[str, Any]) -> str | None:
    """Which biometric feature a ``sensor`` reports, or ``None`` if it is not biometric."""
    hints = " ".join(
        x.lower()
        for x in (
            entity_id,
            _attr(attributes, "friendly_name") or "",
            _attr(attributes, "device_class") or "",
        )
    ).replace(" ", "_")
    unit = (_attr(attributes, "unit_of_measurement") or "").lower()
    if "hrv" in hints or "heart_rate_variability" in hints:
        return "hrv"
    if "heart_rate" in hints or "pulse" in hints or unit in {"bpm", "beats/min"}:
        return "heart_rate"
    if "respirat" in hints or unit in {"breaths/min", "brpm"}:
        return "respiration_rate"
    if ("skin" in hints or "body" in hints) and "temp" in hints:
        return "skin_temp"
    if "electrodermal" in hints or "_eda" in hints or hints.startswith("eda"):
        return "eda"
    if "steps_per_min" in hints or "cadence" in hints:
        return "steps_per_min"
    return None


def describe_state_change(
    entity_id: str,
    state: str | None,
    attributes: Mapping[str, Any],
    *,
    old_state: str | None = None,
    zone: str | None = None,
    event: str | None = None,
) -> str:
    """A short natural-language sentence for a Home Assistant state change."""
    domain = entity_id.split(".", 1)[0] if "." in entity_id else ""
    name = _attr(attributes, "friendly_name") or _pretty(entity_id)
    st = (state or "").strip()
    if domain == "media_player":
        title = _attr(attributes, "media_title")
        artist = _attr(attributes, "media_artist")
        what = title or _attr(attributes, "app_name")
        if what and artist and title:
            what = f"{what} by {artist}"
        if st == "playing":
            return f"{name} started playing {what}" if what else f"{name} started playing"
        if st == "paused":
            return f"{name} paused {what}" if what else f"{name} paused"
        if st in _STOPPED_MEDIA:
            return f"{name} stopped playing"
    if domain in {"person", "device_tracker"} or (zone and event):
        leaving = (event or "").endswith("leave") or st == "not_home"
        place = _pretty(zone) if zone else None
        if leaving:
            if place is None:
                place = _pretty(old_state) if old_state and old_state != "not_home" else "home"
            return f"{name} left home" if place.lower() == "home" else f"{name} left {place}"
        place = place or (_pretty(st) if st else None)
        if place is None:
            return f"{name} moved"
        return f"{name} arrived home" if place.lower() == "home" else f"{name} arrived at {place}"
    if domain in _ON_OFF_DOMAINS and st in {"on", "off", "open", "closed", "locked", "unlocked"}:
        return f"{name} turned {st}" if st in {"on", "off"} else f"{name} is {st}"
    if domain == "weather" and st:
        return f"The weather is {st.replace('_', ' ').replace('-', ' ')}"
    if domain == "sun" and st in {"above_horizon", "below_horizon"}:
        return "The sun rose" if st == "above_horizon" else "The sun set"
    if domain == "sensor" and st:
        unit = _attr(attributes, "unit_of_measurement")
        return f"{name} is {st} {unit}" if unit else f"{name} is {st}"
    if st:
        return f"{name} changed to {st.replace('_', ' ')}"
    return f"{name} changed"


def _flatten(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Turn the native event shape into the flat shape (flat payloads pass through)."""
    flat = dict(payload)
    data = payload.get("data")
    if isinstance(data, Mapping) and ("new_state" in data or "entity_id" in data):
        new_state = f.require_mapping(data.get("new_state"), "data.new_state")
        old_state = f.require_mapping(data.get("old_state"), "data.old_state")
        flat.setdefault("event", payload.get("event_type"))
        flat.setdefault("entity_id", data.get("entity_id") or new_state.get("entity_id"))
        flat.setdefault("state", new_state.get("state"))
        flat.setdefault("attributes", new_state.get("attributes"))
        if old_state:
            flat.setdefault("old_state", old_state.get("state"))
        if f.is_missing(flat.get("time_fired")):
            flat["time_fired"] = new_state.get("last_changed")
    elif "event_type" in payload:
        flat.setdefault("event", payload.get("event_type"))
    return flat


class HomeAssistantAdapter:
    """Adapter ``home_assistant`` (shapes and mapping documented in the module docstring)."""

    def __init__(self, *, clock: Clock | None = None) -> None:
        self._clock: Clock = clock or SystemClock()

    @property
    def name(self) -> str:
        return "home_assistant"

    def to_signals(
        self, payload: Mapping[str, Any]
    ) -> tuple[list[RawSignal], SensoryContext | None]:
        """Map a Home Assistant payload to signals and context."""
        flat = _flatten(payload)
        entity_id = f.as_str(flat.get("entity_id"), "entity_id")
        state = f.as_str(flat.get("state"), "state")
        old_state = f.as_str(flat.get("old_state"), "old_state")
        event = f.as_str(flat.get("event"), "event")
        zone = f.as_str(flat.get("zone"), "zone")
        attributes = f.require_mapping(flat.get("attributes"), "attributes")
        biometric = f.as_biometric(flat.get("biometric"))

        text = f.as_str(flat.get("text"), "text")
        if entity_id is not None:
            domain = entity_id.split(".", 1)[0]
            feature = _biometric_feature(entity_id, attributes) if domain == "sensor" else None
            if feature is not None:
                value = f.as_float(state, entity_id)
                if value is None:
                    raise ObservationError(f"sensor {entity_id!r} has no numeric state")
                unit = (_attr(attributes, "unit_of_measurement") or "").upper()
                if feature == "skin_temp" and unit in {"°F", "F", "DEGF"}:
                    value = (value - 32.0) * 5.0 / 9.0
                biometric.setdefault(feature, value)
            elif text is None:
                text = describe_state_change(
                    entity_id, state, attributes, old_state=old_state, zone=zone, event=event
                )
        elif text is None and event is not None:
            text = f"Home Assistant event: {event.replace('_', ' ')}"
        if text is None and not biometric and "signals" not in flat:
            raise ObservationError(
                "unsupported Home Assistant payload: expected 'entity_id' (+ 'state'), "
                "'event', 'text' or 'biometric'"
            )

        lat = flat.get("latitude", flat.get("lat"))
        lon = flat.get("longitude", flat.get("lon"))
        if f.is_missing(lat) and f.is_missing(lon) and f.is_missing(flat.get("geohash")):
            lat, lon = attributes.get("latitude"), attributes.get("longitude")
        at = flat.get("time_fired")
        if f.is_missing(at):
            at = flat.get("at")
        generic: dict[str, Any] = {
            "signals": flat.get("signals"),
            "text": text,
            "biometric": biometric or None,
            "at": at,
            "geohash": flat.get("geohash"),
            "lat": lat,
            "lon": lon,
            "tags": flat.get("tags"),
            "salience": flat.get("salience"),
        }
        return parse_generic_payload(generic, now=self._clock.now())
