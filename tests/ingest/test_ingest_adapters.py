from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta, timezone

import pytest

from madeleine.core import geohash as gh
from madeleine.core.types import CueAdapter, ObservationError, RawSignal
from madeleine.ingest import (
    GenericWebhookAdapter,
    HomeAssistantAdapter,
    IOSShortcutsAdapter,
    default_adapters,
)
from madeleine.ingest.adapters.home_assistant import describe_state_change
from madeleine.testing import FakeClock

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def by_channel(signals: list[RawSignal]) -> dict[str, RawSignal]:
    out = {s.channel: s for s in signals}
    assert len(out) == len(signals)
    return out


def test_default_adapters_satisfy_protocol(clock: FakeClock) -> None:
    adapters = default_adapters(clock=clock)
    assert [a.name for a in adapters] == ["generic", "home_assistant", "ios_shortcuts"]
    assert all(isinstance(a, CueAdapter) for a in adapters)


# -- generic ---------------------------------------------------------------------------


def test_generic_full_payload(clock: FakeClock) -> None:
    signals, ctx = GenericWebhookAdapter(clock=clock).to_signals(
        {
            "signals": [{"channel": "image", "data_b64": base64.b64encode(PNG).decode()}],
            "text": " rain on a tin roof ",
            "biometric": {"heart_rate": "92"},
            "at": "2026-06-01T21:30:00+02:00",
            "lat": 48.8584,
            "lon": 2.2945,
            "tags": "walk, evening",
            "salience": 0.7,
            "source": "ignored",
            "whatever": 1,
        }
    )
    s = by_channel(signals)
    assert s["text"].value == "rain on a tin roof"
    assert s["biometric"].value == {"heart_rate": 92.0}
    assert s["image"].value == PNG
    assert ctx is not None
    assert ctx.timestamp == datetime(2026, 6, 1, 21, 30, tzinfo=timezone(timedelta(hours=2)))
    assert ctx.geohash == gh.encode(48.8584, 2.2945, 9)
    assert ctx.tags == {"walk", "evening"}
    assert ctx.salience == 0.7


def test_generic_without_context_fields_returns_none_context() -> None:
    signals, ctx = GenericWebhookAdapter().to_signals({"text": "hello"})
    assert ctx is None
    assert signals == [RawSignal("text", "hello")]


def test_generic_location_only_is_a_valid_cue(clock: FakeClock) -> None:
    signals, ctx = GenericWebhookAdapter(clock=clock).to_signals({"geohash": "U09TUNQ"})
    assert signals == []
    assert ctx is not None
    assert ctx.geohash == "u09tunq"
    assert ctx.timestamp == clock.now()


def test_generic_tags_without_time_use_adapter_clock(clock: FakeClock) -> None:
    _, ctx = GenericWebhookAdapter(clock=clock).to_signals({"text": "x", "tags": ["a"]})
    assert ctx is not None
    assert ctx.timestamp == clock.now()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "empty"),
        ({"tags": ["a"]}, "empty"),
        ({"text": "x", "signals": [{"channel": "text", "value": "y"}]}, "both"),
        ({"biometric": {"hr": 1}, "signals": [{"channel": "biometric", "value": {}}]}, "both"),
        ({"signals": "nope"}, "list"),
        ({"signals": ["nope"]}, "object"),
        ({"signals": [{"value": "no channel"}]}, "channel"),
        ({"text": "x", "lat": 1.0}, "together"),
        ({"text": "x", "geohash": "aaaa"}, "geohash"),
        ({"text": "x", "at": "yesterday"}, "ISO"),
        ({"text": "x", "lat": "north", "lon": 2}, "number"),
        ({"text": "x", "biometric": [1, 2]}, "object"),
        ({"text": "x", "biometric": {"heart_rate": "fast"}}, "number"),
        ({"text": "x", "tags": [1]}, "strings"),
        ({"text": "x", "salience": 3}, "salience"),
        ({"text": ["x"]}, "string"),
        ({"signals": [{"channel": "image", "data_b64": "!!"}]}, "base64"),
    ],
)
def test_generic_rejects_malformed(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ObservationError, match=message):
        GenericWebhookAdapter().to_signals(payload)


# -- Home Assistant --------------------------------------------------------------------


def test_ha_media_player_playing(clock: FakeClock) -> None:
    signals, ctx = HomeAssistantAdapter(clock=clock).to_signals(
        {
            "event": "state_changed",
            "entity_id": "media_player.kitchen",
            "state": "playing",
            "attributes": {
                "friendly_name": "Kitchen speaker",
                "media_title": "Clair de lune",
                "media_artist": "Debussy",
            },
            "time_fired": "2026-06-01T21:30:00+02:00",
        }
    )
    assert signals == [
        RawSignal("text", "Kitchen speaker started playing Clair de lune by Debussy")
    ]
    assert ctx is not None
    assert ctx.timestamp.utcoffset() == timedelta(hours=2)
    assert ctx.geohash is None


def test_ha_template_placeholders_are_ignored() -> None:
    signals, ctx = HomeAssistantAdapter().to_signals(
        {
            "entity_id": "media_player.living_room",
            "state": "playing",
            "attributes": {"media_title": "Song", "media_artist": "None", "friendly_name": ""},
            "latitude": "None",
            "longitude": "",
            "time_fired": "",
        }
    )
    assert signals == [RawSignal("text", "Living room started playing Song")]
    assert ctx is None


def test_ha_person_arrives_home_with_coordinates(clock: FakeClock) -> None:
    signals, ctx = HomeAssistantAdapter(clock=clock).to_signals(
        {
            "event": "zone.enter",
            "entity_id": "person.ana",
            "state": "home",
            "zone": "Home",
            "attributes": {"friendly_name": "Ana"},
            "latitude": "48.8584",
            "longitude": "2.2945",
        }
    )
    assert signals == [RawSignal("text", "Ana arrived home")]
    assert ctx is not None
    assert ctx.geohash == gh.encode(48.8584, 2.2945, 9)
    assert ctx.timestamp == clock.now()


def test_ha_coordinates_from_attributes() -> None:
    _, ctx = HomeAssistantAdapter().to_signals(
        {
            "entity_id": "device_tracker.phone",
            "state": "Work",
            "attributes": {"latitude": 40.0, "longitude": -3.0},
        }
    )
    assert ctx is not None
    assert ctx.geohash == gh.encode(40.0, -3.0, 9)


@pytest.mark.parametrize(
    ("entity_id", "state", "attributes", "extra", "expected"),
    [
        ("media_player.kitchen", "playing", {}, {}, "Kitchen started playing"),
        ("media_player.tv", "playing", {"app_name": "Netflix"}, {}, "Tv started playing Netflix"),
        ("media_player.tv", "paused", {"media_title": "Up"}, {}, "Tv paused Up"),
        ("media_player.tv", "idle", {}, {}, "Tv stopped playing"),
        ("person.ana", "not_home", {}, {"old_state": "home"}, "Ana left home"),
        ("person.ana", "not_home", {}, {"old_state": "work"}, "Ana left Work"),
        ("person.ana", "gym", {}, {}, "Ana arrived at Gym"),
        ("person.ana", "home", {}, {"zone": "zone.work", "event": "leave"}, "Ana left Work"),
        ("light.hallway", "on", {"friendly_name": "Hallway light"}, {}, "Hallway light turned on"),
        ("lock.front_door", "unlocked", {}, {}, "Front door is unlocked"),
        ("weather.home", "rainy", {}, {}, "The weather is rainy"),
        ("sun.sun", "below_horizon", {}, {}, "The sun set"),
        (
            "sensor.living_room_temperature",
            "21.5",
            {"unit_of_measurement": "°C", "device_class": "temperature"},
            {},
            "Living room temperature is 21.5 °C",
        ),
        ("vacuum.robot", "cleaning", {}, {}, "Robot changed to cleaning"),
        ("vacuum.robot", None, {}, {}, "Robot changed"),
    ],
)
def test_ha_descriptions(
    entity_id: str,
    state: str | None,
    attributes: dict[str, str],
    extra: dict[str, str],
    expected: str,
) -> None:
    assert describe_state_change(entity_id, state, attributes, **extra) == expected


@pytest.mark.parametrize(
    ("entity_id", "attributes", "state", "feature", "value"),
    [
        ("sensor.pixel_watch_heart_rate", {}, "92", "heart_rate", 92.0),
        ("sensor.watch_pulse_x", {"unit_of_measurement": "bpm"}, "70", "heart_rate", 70.0),
        ("sensor.band", {"unit_of_measurement": "bpm"}, "61", "heart_rate", 61.0),
        ("sensor.watch_hrv", {}, "41", "hrv", 41.0),
        ("sensor.oura_respiratory_rate", {}, "14", "respiration_rate", 14.0),
        ("sensor.skin_temperature", {"unit_of_measurement": "°C"}, "33.5", "skin_temp", 33.5),
        ("sensor.body_temperature", {"unit_of_measurement": "°F"}, "98.6", "skin_temp", 37.0),
        ("sensor.watch_eda", {}, "2.5", "eda", 2.5),
    ],
)
def test_ha_wearable_sensors_become_biometric(
    entity_id: str, attributes: dict[str, str], state: str, feature: str, value: float
) -> None:
    signals, _ = HomeAssistantAdapter().to_signals(
        {"entity_id": entity_id, "state": state, "attributes": attributes}
    )
    assert len(signals) == 1
    assert signals[0].channel == "biometric"
    assert isinstance(signals[0].value, dict)
    assert signals[0].value == pytest.approx({feature: value})


def test_ha_ambient_temperature_is_text_not_biometric() -> None:
    signals, _ = HomeAssistantAdapter().to_signals(
        {
            "entity_id": "sensor.kitchen_temperature",
            "state": "22",
            "attributes": {"device_class": "temperature", "unit_of_measurement": "°C"},
        }
    )
    assert [s.channel for s in signals] == ["text"]


def test_ha_biometric_sensor_with_non_numeric_state_is_rejected() -> None:
    with pytest.raises(ObservationError, match="numeric"):
        HomeAssistantAdapter().to_signals(
            {"entity_id": "sensor.watch_heart_rate", "state": "unavailable"}
        )


def test_ha_native_event_shape() -> None:
    signals, ctx = HomeAssistantAdapter().to_signals(
        {
            "event_type": "state_changed",
            "data": {
                "entity_id": "weather.home",
                "old_state": {"state": "cloudy"},
                "new_state": {
                    "state": "rainy",
                    "attributes": {"friendly_name": "Home"},
                    "last_changed": "2026-06-01T19:30:12.483210+00:00",
                },
            },
        }
    )
    assert signals == [RawSignal("text", "The weather is rainy")]
    assert ctx is not None
    assert ctx.timestamp == datetime(2026, 6, 1, 19, 30, 12, 483210, tzinfo=UTC)


def test_ha_text_override_and_generic_fields() -> None:
    signals, ctx = HomeAssistantAdapter().to_signals(
        {
            "entity_id": "binary_sensor.door",
            "state": "on",
            "text": "the front door creaked open",
            "biometric": {"heart_rate": 80},
            "tags": ["home"],
            "geohash": "u09tunq",
        }
    )
    s = by_channel(signals)
    assert s["text"].value == "the front door creaked open"
    assert s["biometric"].value == {"heart_rate": 80.0}
    assert ctx is not None
    assert ctx.geohash == "u09tunq"
    assert ctx.tags == {"home"}


def test_ha_custom_event_without_entity() -> None:
    signals, _ = HomeAssistantAdapter().to_signals({"event": "doorbell_pressed"})
    assert signals == [RawSignal("text", "Home Assistant event: doorbell pressed")]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"state": "on"},
        {"entity_id": "light.x", "attributes": "nope"},
        {"event_type": "state_changed", "data": {"entity_id": "x.y", "new_state": "bad"}},
    ],
)
def test_ha_rejects_unsupported(payload: dict[str, object]) -> None:
    with pytest.raises(ObservationError):
        HomeAssistantAdapter().to_signals(payload)


# -- iOS Shortcuts ---------------------------------------------------------------------


def test_ios_full_payload(clock: FakeClock) -> None:
    b64 = base64.b64encode(PNG).decode()
    wrapped = "\n".join(b64[i : i + 20] for i in range(0, len(b64), 20))
    signals, ctx = IOSShortcutsAdapter(clock=clock).to_signals(
        {
            "text": "Back at the old bakery",
            "latitude": "48,8584",
            "longitude": "2,2945",
            "date": "2026-06-01T21:30:00+02:00",
            "heart_rate": "72",
            "photo_b64": wrapped,
            "audio_b64": f"data:audio/wav;base64,{b64}",
            "focus": "Work",
            "battery": 15,
            "tags": ["commute"],
            "biometric": {"hrv": 41},
        }
    )
    s = by_channel(signals)
    assert s["text"].value == "Back at the old bakery"
    assert s["image"].value == PNG
    assert s["image"].mime_type == "image/jpeg"
    assert s["audio"].value == PNG
    assert s["audio"].mime_type == "audio/wav"
    assert s["biometric"].value == {"hrv": 41.0, "heart_rate": 72.0}
    assert ctx is not None
    assert ctx.geohash == gh.encode(48.8584, 2.2945, 9)
    assert ctx.timestamp.utcoffset() == timedelta(hours=2)
    assert ctx.tags == {"commute", "focus:work", "battery:low"}


def test_ios_empty_strings_mean_absent(clock: FakeClock) -> None:
    signals, ctx = IOSShortcutsAdapter(clock=clock).to_signals(
        {
            "text": "",
            "latitude": 48.8,
            "longitude": 2.3,
            "date": "",
            "heart_rate": "",
            "photo_b64": "",
            "focus": "",
            "battery": 0.9,
        }
    )
    assert signals == []
    assert ctx is not None
    assert ctx.tags == frozenset()
    assert ctx.timestamp == clock.now()


def test_ios_photo_mime_override_and_aliases() -> None:
    signals, ctx = IOSShortcutsAdapter().to_signals(
        {
            "photo_b64": base64.b64encode(PNG).decode(),
            "photo_mime_type": "image/png",
            "lat": 1.0,
            "lon": 2.0,
            "at": "2026-06-01T10:00:00Z",
        }
    )
    assert signals[0].mime_type == "image/png"
    assert ctx is not None
    assert ctx.timestamp == datetime(2026, 6, 1, 10, tzinfo=UTC)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"text": "x", "date": "1 Jun 2026 at 21:30"}, "ISO 8601"),
        ({"text": "x", "date": "2026-13-45T00:00:00"}, "ISO"),
        ({"photo_b64": "data:image/png,abc"}, "base64"),
        ({"photo_b64": "not base64!"}, "base64"),
        ({"heart_rate": "fast"}, "number"),
        ({"focus": "Work"}, "empty"),
    ],
)
def test_ios_rejects_malformed(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ObservationError, match=message):
        IOSShortcutsAdapter().to_signals(payload)
