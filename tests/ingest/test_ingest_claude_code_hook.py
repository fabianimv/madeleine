from __future__ import annotations

import io
import json
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
import uvicorn

from madeleine.core.types import MadeleineConfig, SensoryContext
from madeleine.ingest import create_app
from madeleine.integrations import claude_code
from madeleine.integrations.claude_code import (
    HEADER,
    MAX_CONTEXT_CHARS,
    MAX_LINE_CHARS,
    build_output,
    fit_count,
    main,
)
from madeleine.testing import FakeClock, FakeEngine


def hook_input(prompt: str | None = "what should I cook tonight?") -> io.StringIO:
    data: dict[str, Any] = {
        "session_id": "abc123",
        "transcript_path": "/tmp/t.jsonl",
        "cwd": "/home/user/project",
        "permission_mode": "default",
        "hook_event_name": "UserPromptSubmit",
    }
    if prompt is not None:
        data["prompt"] = prompt
    return io.StringIO(json.dumps(data))


def run(argv: list[str], engine: FakeEngine | None = None, stdin: io.StringIO | None = None) -> str:
    out = io.StringIO()
    configs: list[MadeleineConfig] = []

    def factory(config: MadeleineConfig) -> FakeEngine:
        configs.append(config)
        assert engine is not None
        return engine

    rc = main(argv, stdin=stdin or hook_input(), stdout=out, engine_factory=factory)
    assert rc == 0
    return out.getvalue()


def additional_context(output: str) -> str:
    data = json.loads(output)
    assert data["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    ctx = data["hookSpecificOutput"]["additionalContext"]
    assert isinstance(ctx, str)
    return ctx


def queue_one(engine: FakeEngine, clock: FakeClock) -> None:
    engine.remember(
        "Ana proposed at the Eiffel Tower", context=SensoryContext(clock.now(), geohash="u09tunq")
    )
    engine.sense(engine.build_cue(context=SensoryContext(clock.now(), geohash="u09tunq")))
    assert engine.queue.pending_count() == 1


def queue_contents(engine: FakeEngine, clock: FakeClock, contents: list[str]) -> None:
    """Queue one recollection per content, in this (FIFO) order."""
    queue_one(engine, clock)
    [template] = engine.queue.drain()
    for i, content in enumerate(contents):
        memory = replace(template.memory, id=f"mem_{i}", content=content)
        engine.queue.put(
            replace(template, id=f"rec_{i}", match=replace(template.match, memory=memory))
        )


def context_lines(output: str) -> list[str]:
    return [line for line in additional_context(output).splitlines() if line.startswith("[")]


# -- local mode ------------------------------------------------------------------------


def test_local_empty_queue_prints_nothing(engine: FakeEngine) -> None:
    assert run([], engine) == ""
    assert engine.closed


def test_local_with_recollections(engine: FakeEngine, clock: FakeClock) -> None:
    queue_one(engine, clock)
    ctx = additional_context(run([], engine))
    lines = ctx.splitlines()
    assert lines[0] == HEADER
    assert lines[1] == "[madeleine 1.00 via geo+time] Ana proposed at the Eiffel Tower (2026-06-01)"
    assert engine.queue.pending_count() == 0  # drained
    assert run([], engine) == ""  # delivered once


def test_local_limit(engine: FakeEngine, clock: FakeClock) -> None:
    for word in ("alpha", "beta", "gamma"):
        engine.remember(f"tower {word}", context=SensoryContext(clock.now(), geohash="u09tunq"))
    engine.sense(engine.build_cue(context=SensoryContext(clock.now(), geohash="u09tunq")))
    assert len(additional_context(run(["--limit", "2"], engine)).splitlines()) == 3
    assert engine.queue.pending_count() == 1


def test_local_db_option_reaches_config(engine: FakeEngine) -> None:
    seen: list[MadeleineConfig] = []

    def factory(config: MadeleineConfig) -> FakeEngine:
        seen.append(config)
        return engine

    main(["--db", "/tmp/x.db"], stdin=hook_input(), stdout=io.StringIO(), engine_factory=factory)
    assert seen[0].db_path == "/tmp/x.db"


def test_sense_prompt_triggers_memories(engine: FakeEngine) -> None:
    engine.remember("Ana loved the rain on the tin roof")
    assert run([], engine, hook_input("the rain on the tin roof tonight")) == ""
    ctx = additional_context(
        run(["--sense-prompt"], engine, hook_input("the rain on the tin roof tonight"))
    )
    assert "Ana loved the rain on the tin roof" in ctx
    assert "via text" in ctx


def test_sense_prompt_without_prompt_still_drains(engine: FakeEngine, clock: FakeClock) -> None:
    queue_one(engine, clock)
    assert additional_context(run(["--sense-prompt"], engine, hook_input(None)))


def test_sense_prompt_failure_still_drains(
    engine: FakeEngine, clock: FakeClock, capsys: pytest.CaptureFixture[str]
) -> None:
    queue_one(engine, clock)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("embedder exploded")

    engine.sense = boom  # type: ignore[method-assign]
    assert additional_context(run(["--sense-prompt"], engine))
    assert "embedder exploded" in capsys.readouterr().err


@pytest.mark.parametrize("stdin", ["", "not json", "[1, 2]", '{"prompt": 3}'])
def test_odd_stdin_is_tolerated(engine: FakeEngine, clock: FakeClock, stdin: str) -> None:
    queue_one(engine, clock)
    assert additional_context(run(["--sense-prompt"], engine, io.StringIO(stdin)))


def test_engine_error_exits_zero_with_message(capsys: pytest.CaptureFixture[str]) -> None:
    def factory(config: MadeleineConfig) -> FakeEngine:
        raise RuntimeError("database is locked")

    out = io.StringIO()
    assert main([], stdin=hook_input(), stdout=out, engine_factory=factory) == 0
    assert out.getvalue() == ""
    assert "database is locked" in capsys.readouterr().err


def test_bad_arguments_exit_zero(capsys: pytest.CaptureFixture[str]) -> None:
    out = io.StringIO()
    assert main(["--nope"], stdin=hook_input(), stdout=out) == 0
    assert main(["--limit", "0"], stdin=hook_input(), stdout=out) == 0
    assert out.getvalue() == ""
    err = capsys.readouterr().err
    assert "--nope" in err
    assert "--limit" in err


def test_missing_engine_module_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "madeleine.engine", None)
    out = io.StringIO()
    assert main([], stdin=hook_input(), stdout=out) == 0
    assert out.getvalue() == ""
    assert "madeleine hook:" in capsys.readouterr().err


def test_build_output_caps_size() -> None:
    assert build_output([]) is None
    assert build_output(["  "]) is None
    out = build_output(["x" * 4000, "multi\nline", "y" * 6000])
    assert out is not None
    ctx = out["hookSpecificOutput"]["additionalContext"]
    assert len(ctx) <= MAX_CONTEXT_CHARS
    cut = MAX_LINE_CHARS - 1
    assert ctx.splitlines() == [HEADER, "x" * cut + "…", "multi line", "y" * cut + "…"]
    out = build_output(["z" * 900] * 20)
    assert out is not None
    assert len(out["hookSpecificOutput"]["additionalContext"]) <= MAX_CONTEXT_CHARS


def test_hook_module_imports_are_light() -> None:
    code = (
        "import sys, madeleine.integrations.claude_code as m; "
        "heavy = [n for n in ('fastapi', 'starlette', 'pydantic', 'mcp', 'numpy', 'uvicorn') "
        "if n in sys.modules]; print(heavy)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    assert result.stdout.strip() == "[]"


# -- remote mode -----------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


@pytest.fixture
def server(engine: FakeEngine, clock: FakeClock) -> Iterator[str]:
    port = _free_port()
    app = create_app(engine, clock=clock, api_token="hook-token")
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on")
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started:
        if time.monotonic() > deadline:
            pytest.fail("uvicorn did not start")
        time.sleep(0.02)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.should_exit = True
        thread.join(timeout=10)


def test_remote_drains_server_queue(server: str, engine: FakeEngine, clock: FakeClock) -> None:
    queue_one(engine, clock)
    out = run(["--url", server + "/", "--token", "hook-token"])
    ctx = additional_context(out)
    assert "Ana proposed at the Eiffel Tower" in ctx
    assert ctx.splitlines()[1].startswith("[madeleine 1.00 via geo+time]")
    assert engine.queue.pending_count() == 0
    assert run(["--url", server, "--token", "hook-token"]) == ""


def test_remote_token_from_env(
    server: str, engine: FakeEngine, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue_one(engine, clock)
    monkeypatch.setenv("MADELEINE_API_TOKEN", "hook-token")
    assert additional_context(run(["--url", server]))


def test_remote_sense_prompt(server: str, engine: FakeEngine) -> None:
    engine.remember("Ana loved the rain on the tin roof")
    out = run(
        ["--url", server, "--token", "hook-token", "--sense-prompt"],
        stdin=hook_input("rain on the tin roof again"),
    )
    assert "Ana loved the rain on the tin roof" in additional_context(out)


def test_remote_wrong_token_exits_zero(
    server: str, engine: FakeEngine, clock: FakeClock, capsys: pytest.CaptureFixture[str]
) -> None:
    queue_one(engine, clock)
    assert run(["--url", server, "--token", "wrong"]) == ""
    assert "401" in capsys.readouterr().err
    assert engine.queue.pending_count() == 1  # nothing lost


def test_remote_unreachable_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    url = f"http://127.0.0.1:{_free_port()}"
    assert run(["--url", url, "--timeout", "1"]) == ""
    assert "unreachable" in capsys.readouterr().err


def test_remote_rejects_non_http_url(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(["--url", "file:///etc/passwd"]) == ""
    assert "http" in capsys.readouterr().err


def test_line_from_json_fallback() -> None:
    item = {
        "score": 0.8234,
        "channel_scores": {"time": 0.5, "geo": 1.0},
        "memory": {"content": "hello", "context": {"timestamp": "2024-06-01T21:30:00+02:00"}},
    }
    assert claude_code._line_from_json(item) == "[madeleine 0.82 via geo+time] hello (2024-06-01)"
    assert claude_code._line_from_json({**item, "prompt_line": "given"}) == "given"


# -- R8: the hook never loses recollections --------------------------------------------


def test_long_recollection_is_truncated_not_lost(engine: FakeEngine, clock: FakeClock) -> None:
    queue_contents(engine, clock, ["L" * 12_000, "short and sweet"])
    lines = context_lines(run([], engine))
    assert len(lines) == 2
    assert len(lines[0]) == MAX_LINE_CHARS
    assert lines[0].endswith("…")
    assert "short and sweet" in lines[1]
    assert engine.queue.pending_count() == 0


def _many(n: int) -> list[str]:
    return [f"memory {i:02d} " + "w" * 900 for i in range(n)]


def test_what_does_not_fit_stays_queued(engine: FakeEngine, clock: FakeClock) -> None:
    contents = _many(15)
    queue_contents(engine, clock, contents)
    first = run(["--limit", "15"], engine)
    assert len(additional_context(first)) <= MAX_CONTEXT_CHARS
    shown = context_lines(first)
    assert 1 <= len(shown) < 15
    assert engine.queue.pending_count() == 15 - len(shown)  # drained exactly what was shown
    second = context_lines(run(["--limit", "15"], engine))
    delivered = [line.split("] ", 1)[1].rsplit(" (", 1)[0] for line in shown + second]
    assert delivered == contents  # every one exactly once, in FIFO order
    assert engine.queue.pending_count() == 0


def test_fit_count_matches_build_output() -> None:
    lines = ["x" * 5000, "y" * 5000, "z" * 5000] * 5
    count = fit_count(lines)
    out = build_output(lines)
    assert out is not None
    assert len(out["hookSpecificOutput"]["additionalContext"].splitlines()) - 1 == count
    assert fit_count(["a" * 50_000]) == 1  # a single huge line always fits (truncated)


def test_remote_what_does_not_fit_stays_queued(
    server: str, engine: FakeEngine, clock: FakeClock
) -> None:
    contents = ["L" * 12_000, *_many(14)]
    queue_contents(engine, clock, contents)
    argv = ["--url", server, "--token", "hook-token", "--limit", "15"]
    shown = context_lines(run(argv))
    assert len(shown[0]) == MAX_LINE_CHARS
    assert 1 < len(shown) < 15
    assert engine.queue.pending_count() == 15 - len(shown)
    rest = context_lines(run(argv))
    assert len(shown) + len(rest) == 15
    assert engine.queue.pending_count() == 0
