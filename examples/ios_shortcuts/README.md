# madeleine + iOS Shortcuts

An iPhone knows where you are, what time it is, your heart rate and what you just
photographed. A Shortcut can send all of it to madeleine as a cue
(`POST /cues/ios_shortcuts`), so a place or a photo can bring back a memory by itself.

Prerequisite: madeleine's HTTP server reachable from the phone, with a token:

```bash
MADELEINE_API_TOKEN=change-me madeleine serve --transport http --host 0.0.0.0 --port 8765
```

(Over the internet, put it behind HTTPS, e.g. a reverse proxy or Tailscale.)

## Recipe: "Sense here"

Create a new Shortcut and add these actions in order:

1. **Get Current Location**.
2. **Get Details of Locations** -> *Latitude* (from *Current Location*). Tap the result
   and rename the variable to `Latitude`. Repeat for *Longitude*.
3. **Current Date**, then **Format Date**: *Date Format* **ISO 8601**, *Include ISO 8601
   Time* on. Rename the result `When`. ISO 8601 is required: localized formats such as
   "1 Jun 2026 at 21:30" are rejected (HTTP 422) because they are ambiguous and carry no
   UTC offset.
4. *(optional)* **Find Health Samples** where *Type* is *Heart Rate*, *Sort by* *Start
   Date*, *Order* *Latest First*, *Limit* 1. Then **Get Details of Health Sample** ->
   *Value*; rename it `HeartRate`.
5. *(optional)* **Dictate Text** (or **Ask for Input**) -> `Note`.
6. *(optional)* **Get Battery Level** -> `Battery`; **Get Current Focus** (where available) -> `Focus`.
7. **Get Contents of URL**:
   - URL: `http://192.168.1.20:8765/cues/ios_shortcuts`
   - Method: **POST**
   - Headers: `Authorization` = `Bearer change-me`
   - Request Body: **JSON**, with fields:

     | Key | Type | Value |
     |---|---|---|
     | `latitude` | Number | `Latitude` |
     | `longitude` | Number | `Longitude` |
     | `date` | Text | `When` |
     | `heart_rate` | Number | `HeartRate` |
     | `text` | Text | `Note` |
     | `battery` | Number | `Battery` |
     | `focus` | Text | `Focus` |
8. *(optional)* **Get Dictionary Value** `recollections` from *Contents of URL*, then
   **Show Notification** if it is not empty: the memories that just surfaced.

### Adding a photo

Before *Get Contents of URL*: **Take Photo** (or **Select Photos**) -> **Convert Image** to
**JPEG** (HEIC is not decodable by the image channel) -> **Resize Image** to 512 px wide
(keeps the body well under the server's 10 MB limit) -> **Base64 Encode** (*Line Breaks*:
*None*; line breaks are tolerated anyway). Add the JSON fields `photo_b64` (Text, the
encoded result) and `photo_mime_type` (Text, `image/jpeg`).

Audio works the same with `audio_b64` / `audio_mime_type`. Note that *Record Audio*
produces AAC (`audio/mp4`); whether the audio channel can decode it depends on its
backend (see `docs/channels.md`).

The photo is embedded and then discarded: raw bytes are never stored unless the server is
configured to keep raw media.

## Personal automation: arriving somewhere sends a cue

Shortcuts app -> **Automation** -> **+** -> **Arrive** -> choose a location (the café where
you used to study, your grandmother's street...) -> *Run Immediately* -> action **Run
Shortcut** -> "Sense here". Every arrival now sends a location cue; madeleine's refractory
period ensures each memory surfaces once, not on every visit that afternoon.

Other good triggers: *Leave*, *Time of Day* (combined with location), *Focus* turning on,
*CarPlay* connecting, *NFC* tag on the coffee machine.

## Payload reference

[`../payloads/ios_shortcuts_location.json`](../payloads/ios_shortcuts_location.json):

```json
{
  "text": "Back at the café where we used to study",
  "latitude": "48.8530",
  "longitude": "2.3499",
  "date": "2026-06-01T16:05:00+02:00",
  "heart_rate": "71",
  "focus": "Personal",
  "battery": 64
}
```

| Field | Meaning |
|---|---|
| `text` | Text cue (what you say / note). |
| `latitude`, `longitude` (or `lat`, `lon`) | Location -> `geo` cue. Numbers or numeric strings (`"48,853"` accepted). |
| `geohash` | Alternative to latitude/longitude. |
| `date` (or `at`) | ISO 8601 with offset; default: when the server receives it. |
| `heart_rate` | bpm -> `biometric` cue (strengthens a match, never triggers alone). |
| `biometric` | Extra features, e.g. `{"hrv": 41}`. |
| `photo_b64`, `photo_mime_type` | Image cue (default type `image/jpeg`; `data:` URIs accepted). |
| `audio_b64`, `audio_mime_type` | Sound cue (default type `audio/mp4`). |
| `focus` | Tag `focus:<name>` (lower-case). |
| `battery` | 0-100 or 0-1; at or below 20 % adds the tag `battery:low`. |
| `tags` | Extra tags (list or comma-separated text). |

Empty strings mean "not provided" (unset Shortcut variables), unknown keys are ignored, and
the cue needs at least one signal or a location. The response is
`{"cue_id": ..., "recollections": [...]}`; each recollection has a ready-to-read
`prompt_line`.
