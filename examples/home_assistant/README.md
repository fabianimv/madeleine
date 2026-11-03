# madeleine + Home Assistant

Home Assistant sends cues to madeleine's HTTP API (`POST /cues/home_assistant`); events in
your home (arriving home, a song starting, the weather turning to rain) can then surface
memories without your agent asking. Recollections queue up and reach the agent on its next
prompt (Claude Code hook) or through the MCP server.

## 1. Run madeleine's HTTP server

```bash
MADELEINE_API_TOKEN=change-me madeleine serve --transport http --host 0.0.0.0 --port 8765
```

The server binds to `127.0.0.1` by default; `--host 0.0.0.0` exposes it on your LAN, so set
a token (every route except `/healthz` then needs `Authorization: Bearer <token>`).

## 2. Configure Home Assistant

Merge [`configuration.yaml`](configuration.yaml) into yours: a `rest_command`
(`madeleine_cue`) and two automations (arriving in the home zone; a media player starting to
play). Put the header value in `secrets.yaml`:

```yaml
madeleine_auth: "Bearer change-me"
```

Reload YAML (Developer tools -> YAML -> All YAML configuration) and test from Developer tools
-> Actions with:

```yaml
action: rest_command.madeleine_cue
data:
  payload:
    entity_id: media_player.kitchen
    state: playing
    attributes: {friendly_name: Kitchen speaker, media_title: Clair de lune}
```

## Supported payloads

The adapter (`madeleine.ingest.adapters.home_assistant`) accepts:

| Shape | Example | Becomes |
|---|---|---|
| Flat state change | [`home_assistant_media_playing.json`](../payloads/home_assistant_media_playing.json) | text "Kitchen speaker started playing Clair de lune by Claude Debussy" + time |
| Zone / person | [`home_assistant_arrive_home.json`](../payloads/home_assistant_arrive_home.json) | text "Ana arrived home" + geo (latitude/longitude) + time |
| Wearable sensor | [`home_assistant_heart_rate.json`](../payloads/home_assistant_heart_rate.json) | biometric `heart_rate` (+ geo, time); no text |
| Native event (Node-RED, AppDaemon, websocket API) | [`home_assistant_native_event.json`](../payloads/home_assistant_native_event.json) | text "The weather is rainy" + time |

Fields of the flat shape: `event`, `entity_id`, `state`, `old_state`, `attributes`
(`friendly_name`, `media_title`, `media_artist`, `app_name`, `unit_of_measurement`,
`device_class`, `latitude`, `longitude`), `latitude`/`longitude`, `zone`, `time_fired`
(or `at`). The generic fields (`text`, `signals`, `biometric`, `tags`, `geohash`,
`lat`/`lon`, `salience`) are accepted too; an explicit `text` replaces the generated
sentence.

Text by domain: `media_player` (started playing / paused / stopped playing), `person` and
`device_tracker` (arrived home / arrived at *zone* / left *zone*), on/off domains
(`light`, `switch`, `binary_sensor`, `input_boolean`, `fan`, `cover`, `lock`, `climate`),
`weather` ("The weather is rainy"), `sun` ("The sun set"), `sensor`
("Living room temperature is 21.5 °C") and a generic "<name> changed to <state>".

Wearable sensors become **biometric** readings when the entity id, name or device class
mentions heart rate (or the unit is `bpm`), HRV, respiration, *skin*/*body* temperature
(°F converted to °C), EDA or steps per minute. Room temperature stays text. Time and
biometric data never trigger a memory on their own (they only strengthen a match), so send
a location with wearable readings.

Placeholder values rendered by templates for missing attributes (`"None"`, `"unknown"`,
`"unavailable"`, `""`) are ignored, and numbers may arrive as strings.

### Time zones

Use `{{ now().isoformat() }}` for `time_fired`: it carries your local offset. Home
Assistant's own `time_fired` / `last_changed` are UTC, and madeleine computes "hour of day"
in the timestamp's offset, so UTC timestamps make "evening" mean UTC evening.
