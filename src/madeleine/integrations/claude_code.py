"""Claude Code ``UserPromptSubmit`` hook: inject involuntary recollections into the prompt.

Claude Code runs the hook command before each user prompt reaches the model, passing JSON on
stdin (``session_id``, ``transcript_path``, ``cwd``, ``permission_mode``,
``hook_event_name: "UserPromptSubmit"``, ``prompt``...). If madeleine has queued
recollections, the hook prints::

    {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                            "additionalContext": "Involuntary recollections surfaced ..."}}

and Claude Code adds ``additionalContext`` next to the prompt. Nothing is printed when the
queue is empty. The hook **never blocks the prompt**: every error is reported on stderr and
the exit code is always 0 (exit code 2 would block the prompt).

Modes:

* local (default): open the madeleine database (``--db`` or ``MADELEINE_DB``) and drain
  ``engine.pending_recollections()``;
* remote: ``--url http://127.0.0.1:8765`` drains ``GET /recollections/pending`` of a running
  ``madeleine serve --transport http`` (token: ``--token`` or ``MADELEINE_API_TOKEN``).

``--sense-prompt`` first sends the prompt itself as a ``text`` cue, so what the user writes
can trigger involuntary memories too. Only the standard library is imported (fast start).
Configuration example: ``examples/claude_code/``.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from typing import IO, TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    from madeleine.core.types import MadeleineConfig, MemoryEngine

__all__ = ["HEADER", "MAX_CONTEXT_CHARS", "build_output", "main"]

HEADER = "Involuntary recollections surfaced by madeleine (sensory cues, not requested):"
MAX_CONTEXT_CHARS = 9_500
"""Claude Code caps injected strings at 10 000 characters; stay safely below."""
MAX_PROMPT_CUE_CHARS = 4_000
"""Longer prompts are truncated before being sensed as a text cue."""
CUE_SOURCE = "claude_code"

EngineFactory = Callable[["MadeleineConfig"], "MemoryEngine"]


def _warn(message: str) -> None:
    try:
        sys.stderr.write(f"madeleine hook: {message}\n")
        sys.stderr.flush()
    except Exception:  # noqa: S110 - stderr may be closed; never fail the hook
        pass


class _Parser(argparse.ArgumentParser):
    """argparse that raises instead of exiting with status 2 (which would block prompts)."""

    def error(self, message: str) -> NoReturn:
        raise ValueError(message)


def _parser() -> argparse.ArgumentParser:
    p = _Parser(
        prog="madeleine hook claude-code",
        description="Claude Code UserPromptSubmit hook: inject queued madeleine recollections.",
    )
    p.add_argument("--url", help="drain a running madeleine HTTP server instead of the local DB")
    p.add_argument("--token", help="bearer token for --url (default: $MADELEINE_API_TOKEN)")
    p.add_argument("--db", help="SQLite path in local mode (default: $MADELEINE_DB)")
    p.add_argument(
        "--sense-prompt",
        action="store_true",
        help="send the prompt as a text cue before draining (the prompt can trigger memories)",
    )
    p.add_argument("--limit", type=int, default=5, help="max recollections to inject (5)")
    p.add_argument("--timeout", type=float, default=3.0, help="HTTP timeout in seconds (3)")
    return p


def _read_prompt(stdin: IO[str]) -> str | None:
    raw = stdin.read()
    if not raw.strip():
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        _warn("stdin is not valid hook JSON; ignoring it")
        return None
    prompt = data.get("prompt") if isinstance(data, Mapping) else None
    return prompt if isinstance(prompt, str) and prompt.strip() else None


def build_output(lines: Sequence[str]) -> dict[str, Any] | None:
    """The hook's JSON output for these prompt lines (``None`` when there are none)."""
    kept: list[str] = []
    size = len(HEADER)
    for line in lines:
        line = " ".join(line.split())  # one recollection per line, whatever the content
        if not line:
            continue
        if size + 1 + len(line) > MAX_CONTEXT_CHARS:
            break
        kept.append(line)
        size += 1 + len(line)
    if not kept:
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "\n".join([HEADER, *kept]),
        }
    }


# -- local mode -------------------------------------------------------------------------


def _default_engine_factory(config: MadeleineConfig) -> MemoryEngine:
    from madeleine.engine import create_engine  # type: ignore[import-not-found,unused-ignore]

    engine: MemoryEngine = create_engine(config)
    return engine


def _local_lines(
    args: argparse.Namespace, prompt: str | None, factory: EngineFactory | None
) -> list[str]:
    from madeleine.config import load_config
    from madeleine.core.types import TEXT, RawSignal

    config = load_config(db_path=args.db)
    engine = (factory or _default_engine_factory)(config)
    try:
        if args.sense_prompt and prompt:
            try:
                cue = engine.build_cue(
                    [RawSignal(TEXT, prompt[:MAX_PROMPT_CUE_CHARS])], source=CUE_SOURCE
                )
                engine.sense(cue)
            except Exception as exc:  # still deliver what is already queued
                _warn(f"could not sense the prompt: {exc}")
        items = engine.pending_recollections(limit=args.limit, drain=True)
        return [r.to_prompt_line() for r in items]
    finally:
        engine.close()


# -- remote mode ------------------------------------------------------------------------


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _request(base: str, path: str, *, token: str | None, timeout: float, body: Any = None) -> Any:
    url = base.rstrip("/") + path
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"--url must be http(s)://..., got {base!r}")
    headers = {"Accept": "application/json"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    method = "POST" if data else "GET"
    # Scheme checked above (http/https only).
    req = urllib.request.Request(url, data=data, headers=headers, method=method)  # noqa: S310
    handlers: list[urllib.request.BaseHandler] = []
    if _is_loopback(parsed.hostname):
        handlers.append(urllib.request.ProxyHandler({}))  # never proxy a local server
    opener = urllib.request.build_opener(*handlers)
    with opener.open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode() or "null")


def _line_from_json(item: Mapping[str, Any]) -> str:
    line = item.get("prompt_line")
    if isinstance(line, str):
        return line
    # Older servers: rebuild Recollection.to_prompt_line() from to_dict().
    memory = item.get("memory") or {}
    via = "+".join(sorted(item.get("channel_scores") or {})) or "?"
    day = str((memory.get("context") or {}).get("timestamp", ""))[:10]
    score = float(item.get("score", 0.0))
    return f"[madeleine {score:.2f} via {via}] {memory.get('content', '')} ({day})"


def _remote_lines(args: argparse.Namespace, prompt: str | None) -> list[str]:
    token = args.token or os.environ.get("MADELEINE_API_TOKEN") or None
    if args.sense_prompt and prompt:
        try:
            _request(
                args.url,
                "/cues",
                token=token,
                timeout=args.timeout,
                body={"text": prompt[:MAX_PROMPT_CUE_CHARS], "source": CUE_SOURCE},
            )
        except Exception as exc:
            _warn(f"could not sense the prompt: {exc}")
    query = urllib.parse.urlencode({"drain": "true", "limit": args.limit})
    data = _request(args.url, f"/recollections/pending?{query}", token=token, timeout=args.timeout)
    items = data.get("recollections") if isinstance(data, Mapping) else None
    if not isinstance(items, list):
        raise ValueError("unexpected response from /recollections/pending")
    return [_line_from_json(i) for i in items if isinstance(i, Mapping)]


# -- entry point ------------------------------------------------------------------------


def main(
    argv: Sequence[str] | None = None,
    *,
    stdin: IO[str] | None = None,
    stdout: IO[str] | None = None,
    engine_factory: EngineFactory | None = None,
) -> int:
    """Run the hook. Always returns 0 so that a failure never blocks the user's prompt.

    ``engine_factory(config)`` builds the engine in local mode (default
    ``madeleine.engine.create_engine``); tests inject a fake.
    """
    out = stdout if stdout is not None else sys.stdout
    try:
        args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        prompt = _read_prompt(stdin if stdin is not None else sys.stdin)
        if args.url:
            lines = _remote_lines(args, prompt)
        else:
            lines = _local_lines(args, prompt, engine_factory)
        output = build_output(lines)
        if output is not None:
            out.write(json.dumps(output, ensure_ascii=False) + "\n")
            out.flush()
    except SystemExit as exc:  # --help
        if exc.code not in (0, None):
            _warn(f"exited with {exc.code}")
    except urllib.error.HTTPError as exc:
        hint = " (check --token / MADELEINE_API_TOKEN)" if exc.code == 401 else ""
        _warn(f"madeleine server answered HTTP {exc.code}{hint}; no recollections injected")
    except urllib.error.URLError as exc:
        _warn(f"madeleine server unreachable ({exc.reason}); no recollections injected")
    except Exception as exc:
        _warn(f"{type(exc).__name__}: {exc}; no recollections injected")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
