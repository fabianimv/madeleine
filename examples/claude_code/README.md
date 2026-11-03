# madeleine + Claude Code

Two pieces, used together:

1. **The MCP server** gives Claude the tools (`remember`, `recall`, `sense`,
   `pending_recollections`, `forget`).
2. **The `UserPromptSubmit` hook** delivers *involuntary* recollections: before each prompt
   reaches Claude, it drains madeleine's queue and injects whatever surfaced as additional
   context. This works whether or not the client honours MCP resource notifications.

## 1. Register the MCP server

```bash
claude mcp add madeleine -- uvx madeleine-mcp serve
# real semantic text embeddings (pulls in torch):
claude mcp add madeleine -- uvx --from 'madeleine-mcp[text]' madeleine serve
```

## 2. Add the hook

Merge [`settings.json`](settings.json) into `.claude/settings.json` (project) or
`~/.claude/settings.json` (user):

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {"hooks": [{"type": "command", "command": "uvx madeleine-mcp hook claude-code", "timeout": 10}]}
    ]
  }
}
```

By default the hook opens the same SQLite database as the MCP server
(`~/.madeleine/madeleine.db`, or `$MADELEINE_DB`) and drains the queue there, so cues sensed
by any process (the HTTP server receiving Home Assistant / iOS webhooks, the MCP `sense`
tool, the CLI) reach the next prompt.

### Options

| Flag | Meaning |
|---|---|
| `--url http://127.0.0.1:8765` | Drain a running `madeleine serve --transport http` over HTTP instead of opening the database (stdlib only, fast). |
| `--token TOKEN` | Bearer token for `--url` (default: `$MADELEINE_API_TOKEN`). |
| `--db PATH` | Database path in local mode. |
| `--sense-prompt` | First send the prompt itself as a `text` cue, so what you type can trigger involuntary memories, then drain. |
| `--limit N` | Inject at most N recollections per prompt (default 5). |
| `--timeout S` | HTTP timeout in seconds for `--url` (default 3). |

Example, remote mode with prompt sensing:

```json
{"type": "command", "command": "uvx madeleine-mcp hook claude-code --url http://127.0.0.1:8765 --sense-prompt", "timeout": 10}
```

With `--sense-prompt` in local mode the text embedder runs inside the hook; with the
`[text]` extra installed the first call loads the model, which can be slow. Prefer `--url`
(the model stays loaded in the server) when sensing prompts.

## What Claude sees

When something surfaced, the hook prints (and Claude Code adds `additionalContext` next to
your prompt):

```json
{
  "hookSpecificOutput": {
    "hookEventName": "UserPromptSubmit",
    "additionalContext": "Involuntary recollections surfaced by madeleine (sensory cues, not requested):\n[madeleine 0.82 via geo+time] Ana proposed at the Eiffel Tower (2024-06-01)"
  }
}
```

When the queue is empty it prints nothing. The hook **never blocks a prompt**: on any error
(database missing, server down, bad flag) it writes a message to stderr and exits 0.

## Try it by hand

```bash
echo '{"hook_event_name": "UserPromptSubmit", "prompt": "it is raining again"}' \
  | uvx madeleine-mcp hook claude-code --sense-prompt
```
