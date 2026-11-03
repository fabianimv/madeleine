# Examples

| Path | What |
|---|---|
| [`claude_code/`](claude_code/) | `UserPromptSubmit` hook settings + MCP server registration for Claude Code. |
| [`home_assistant/`](home_assistant/) | `rest_command` + automations sending cues from Home Assistant. |
| [`ios_shortcuts/`](ios_shortcuts/) | Step-by-step Shortcut and personal automation sending location/photo/heart-rate cues. |
| [`payloads/`](payloads/) | Sample JSON bodies for every HTTP route and adapter (all exercised by the test suite). |
| [`curl.sh`](curl.sh) | Posts every sample payload to a running `madeleine serve --transport http`. |

Payload file names map to routes: `cues_*` -> `POST /cues`, `generic_*` ->
`POST /cues/generic`, `home_assistant_*` -> `POST /cues/home_assistant`, `ios_shortcuts_*`
-> `POST /cues/ios_shortcuts`, `memories_*` -> `POST /memories`, `recall_*` -> `POST /recall`.
