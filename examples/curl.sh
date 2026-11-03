#!/usr/bin/env bash
# Sample requests for every route of the madeleine HTTP ingestion API.
#
#   madeleine serve --transport http          # listens on http://127.0.0.1:8765
#   ./examples/curl.sh                        # run all samples
#
# Environment: MADELEINE_URL (default http://127.0.0.1:8765) and, if the server was started
# with MADELEINE_API_TOKEN set, the same MADELEINE_API_TOKEN here.
set -euo pipefail

URL="${MADELEINE_URL:-http://127.0.0.1:8765}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/payloads"
AUTH=()
if [[ -n "${MADELEINE_API_TOKEN:-}" ]]; then
  AUTH=(-H "Authorization: Bearer ${MADELEINE_API_TOKEN}")
fi

call() {  # call METHOD PATH [PAYLOAD_FILE]
  local method="$1" path="$2" file="${3:-}"
  echo "== ${method} ${path}${file:+  < ${file}}"
  if [[ -n "$file" ]]; then
    curl -sS -X "$method" ${AUTH[@]+"${AUTH[@]}"} -H 'Content-Type: application/json' \
      --data-binary "@${DIR}/${file}" "${URL}${path}"
  else
    curl -sS -X "$method" ${AUTH[@]+"${AUTH[@]}"} "${URL}${path}"
  fi
  echo
}

# Liveness (never needs the token)
call GET /healthz

# Encode memories (same fields as the MCP `remember` tool)
call POST /memories memories_basic.json
call POST /memories memories_place.json

# Pull recall (same fields as the MCP `recall` tool)
call POST /recall recall_text.json
call POST /recall recall_place.json

# Generic cues -> involuntary recollections that fired
call POST /cues cues_text.json
call POST /cues cues_multichannel.json
call POST /cues cues_image.json

# Adapter-specific webhooks
call POST /cues/generic generic_webhook.json
call POST /cues/home_assistant home_assistant_media_playing.json
call POST /cues/home_assistant home_assistant_arrive_home.json
call POST /cues/home_assistant home_assistant_heart_rate.json
call POST /cues/home_assistant home_assistant_native_event.json
call POST /cues/ios_shortcuts ios_shortcuts_location.json
call POST /cues/ios_shortcuts ios_shortcuts_photo.json

# Peek (GET never removes anything), then drain the queue of involuntary recollections
# (what the Claude Code hook does in --url mode)
call GET "/recollections/pending?limit=5"
call POST "/recollections/drain?limit=5"

# Forget a memory: DELETE /memories/{id}
MEMORY_ID="$(curl -sS ${AUTH[@]+"${AUTH[@]}"} -H 'Content-Type: application/json' \
  -d '{"content": "a memory to forget"}' "${URL}/memories" \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["id"])')"
call DELETE "/memories/${MEMORY_ID}"
