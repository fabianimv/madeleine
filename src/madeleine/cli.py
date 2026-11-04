"""``madeleine`` command-line interface (Typer).

Every command imports its implementation lazily so ``madeleine --help`` stays instant and no
model or web framework is imported unless the command needs it.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Annotated, Any

import typer

from madeleine import __version__

app = typer.Typer(
    name="madeleine",
    help="Involuntary, cue-triggered memory for AI agents.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)
hook_app = typer.Typer(help="Client hooks that deliver involuntary recollections.")
app.add_typer(hook_app, name="hook")

DbOpt = Annotated[
    str | None,
    typer.Option("--db", help="SQLite path (default ~/.madeleine/madeleine.db or $MADELEINE_DB)."),
]
TextBackendOpt = Annotated[
    str | None,
    typer.Option("--text-backend", help="auto | sentence-transformers | hash"),
]
AtOpt = Annotated[str | None, typer.Option("--at", help="ISO-8601 time (default: now).")]
GeohashOpt = Annotated[str | None, typer.Option("--geohash", help="Location as a geohash.")]
LatOpt = Annotated[float | None, typer.Option("--lat", help="Latitude (with --lon).")]
LonOpt = Annotated[float | None, typer.Option("--lon", help="Longitude (with --lat).")]
TagOpt = Annotated[list[str] | None, typer.Option("--tag", help="Tag (repeatable).")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON.")]


def _engine(db: str | None, text_backend: str | None = None) -> Any:
    from madeleine.config import load_config
    from madeleine.core.types import MadeleineError
    from madeleine.engine import create_engine

    try:
        return create_engine(load_config(db_path=db, text_backend=text_backend))
    except (MadeleineError, ValueError) as exc:  # bad env/flags, incompatible database...
        raise _fail(str(exc)) from exc


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _fail(message: str) -> typer.Exit:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"madeleine {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")] = False,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    _setup_logging(verbose)


@app.command()
def serve(
    transport: Annotated[str, typer.Option(help="stdio (for MCP clients) or http.")] = "stdio",
    host: Annotated[str, typer.Option(help="HTTP bind address.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="HTTP port.")] = 8765,
    db: DbOpt = None,
    text_backend: TextBackendOpt = None,
    allow_file_uris: Annotated[
        bool,
        typer.Option(
            help="Let clients (MCP and HTTP) reference media files on this machine. Off by default."
        ),
    ] = False,
    insecure_no_auth: Annotated[
        bool,
        typer.Option(
            help="Allow a non-loopback --host without MADELEINE_API_TOKEN (exposes your memories)."
        ),
    ] = False,
) -> None:
    """Run the MCP server (stdio), or MCP + HTTP ingestion on one port (http)."""
    from madeleine.serve import InsecureBindError
    from madeleine.serve import serve as _serve

    if transport not in {"stdio", "http"}:
        raise _fail("--transport must be 'stdio' or 'http'")
    try:
        _serve(
            transport=transport,
            host=host,
            port=port,
            db=db,
            text_backend=text_backend,
            allow_file_uris=allow_file_uris,
            insecure_no_auth=insecure_no_auth,
        )
    except InsecureBindError as exc:
        raise _fail(str(exc)) from exc


@app.command()
def remember(
    content: Annotated[str, typer.Argument(help="What to remember.")],
    at: AtOpt = None,
    geohash: GeohashOpt = None,
    lat: LatOpt = None,
    lon: LonOpt = None,
    tag: TagOpt = None,
    salience: Annotated[float | None, typer.Option(help="0..1, default 0.5.")] = None,
    db: DbOpt = None,
    text_backend: TextBackendOpt = None,
) -> None:
    """Store a memory with its sensory fingerprint (time, place, text)."""
    from madeleine.core.serde import context_from_fields
    from madeleine.core.types import MadeleineError

    engine = _engine(db, text_backend)
    try:
        ctx = context_from_fields(
            now=engine.now(),
            at=at,
            geohash=geohash,
            lat=lat,
            lon=lon,
            tags=tag,
            salience=salience,
        )
        memory = engine.remember(content, context=ctx)
    except (MadeleineError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    finally:
        engine.close()
    typer.echo(json.dumps(memory.to_dict(), indent=2))


@app.command()
def sense(
    text: Annotated[
        str | None, typer.Option(help="Text cue (overheard phrase, song title...).")
    ] = None,
    at: AtOpt = None,
    geohash: GeohashOpt = None,
    lat: LatOpt = None,
    lon: LonOpt = None,
    heart_rate: Annotated[float | None, typer.Option(help="Biometric cue: bpm.")] = None,
    db: DbOpt = None,
    text_backend: TextBackendOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Send a sensory cue; involuntary recollections that fire are queued (and printed)."""
    from madeleine.core.serde import context_from_fields
    from madeleine.core.types import MadeleineError, RawSignal

    engine = _engine(db, text_backend)
    try:
        ctx = context_from_fields(now=engine.now(), at=at, geohash=geohash, lat=lat, lon=lon)
        signals = []
        if text:
            signals.append(RawSignal("text", text))
        if heart_rate is not None:
            signals.append(RawSignal("biometric", {"heart_rate": heart_rate}))
        fired = engine.sense(engine.build_cue(signals, context=ctx, source="cli"))
    except (MadeleineError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    finally:
        engine.close()
    if as_json:
        typer.echo(json.dumps([r.to_dict() for r in fired], indent=2))
    elif not fired:
        typer.echo("(nothing surfaced)")
    else:
        for r in fired:
            typer.echo(r.to_prompt_line())


@app.command()
def recall(
    query: Annotated[str | None, typer.Argument(help="Text query.")] = None,
    at: AtOpt = None,
    geohash: GeohashOpt = None,
    lat: LatOpt = None,
    lon: LonOpt = None,
    k: Annotated[int, typer.Option("-k", help="Max results.")] = 5,
    db: DbOpt = None,
    text_backend: TextBackendOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Explicitly search memories (pull mode)."""
    from madeleine.core.serde import context_from_fields
    from madeleine.core.types import MadeleineError, RawSignal, RecallQuery

    if (lat is None) != (lon is None):
        raise _fail("--lat and --lon must be given together")
    if query is None and at is None and geohash is None and lat is None:
        raise _fail("give a query, --at, --geohash or --lat/--lon")
    engine = _engine(db, text_backend)
    try:
        ctx = context_from_fields(now=engine.now(), at=at, geohash=geohash, lat=lat, lon=lon)
        signals = [RawSignal("text", query)] if query else []
        cue = engine.build_cue(signals, context=ctx, include_time=at is not None, source="cli")
        hits = engine.recall(RecallQuery(cue=cue, k=k))
    except (MadeleineError, ValueError) as exc:
        raise _fail(str(exc)) from exc
    finally:
        engine.close()
    if as_json:
        typer.echo(json.dumps([h.to_dict() for h in hits], indent=2))
        return
    for h in hits:
        via = ", ".join(f"{c}={s:.2f}" for c, s in sorted(h.channel_scores.items()))
        typer.echo(f"{h.score:.2f}  {h.memory.content}  [{via}]  id={h.memory.id}")


@app.command()
def pending(
    peek: Annotated[bool, typer.Option(help="Do not remove them from the queue.")] = False,
    db: DbOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Show (and by default drain) queued involuntary recollections."""
    engine = _engine(db, "hash")
    try:
        recs = engine.pending_recollections(drain=not peek)
    finally:
        engine.close()
    if as_json:
        typer.echo(json.dumps([r.to_dict() for r in recs], indent=2))
    else:
        for r in recs:
            typer.echo(r.to_prompt_line())


@app.command()
def forget(
    memory_id: Annotated[str, typer.Argument(help="Memory id (mem_...).")],
    db: DbOpt = None,
) -> None:
    """Delete a memory everywhere."""
    engine = _engine(db, "hash")
    try:
        deleted = engine.forget(memory_id)
    finally:
        engine.close()
    if not deleted:
        raise _fail(f"no memory {memory_id!r}")
    typer.echo(f"forgot {memory_id}")


@app.command()
def demo(
    db: Annotated[
        str | None, typer.Option("--db", help="SQLite path (default: temporary, deleted after).")
    ] = None,
) -> None:
    """Offline end-to-end demo: remember, sense a cue, receive an involuntary recollection."""
    from madeleine.demo import run_demo

    raise typer.Exit(code=run_demo(db_path=db))


@app.command()
def bench(
    memories: Annotated[int, typer.Option(help="Synthetic memories.")] = 500,
    cues: Annotated[int, typer.Option(help="Synthetic cues.")] = 300,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    sweep: Annotated[bool, typer.Option(help="Also sweep trigger thresholds.")] = False,
    text_backend: Annotated[
        str,
        typer.Option(
            "--text-backend",
            help="hash (offline, default) or sentence-transformers (needs the [text] extra).",
        ),
    ] = "hash",
    as_json: JsonOpt = False,
    output: Annotated[str | None, typer.Option(help="Write the report to this file.")] = None,
) -> None:
    """Run the offline synthetic benchmark (hash embedder) and print the report."""
    from madeleine.evals.cli import main as bench_main

    argv = ["--memories", str(memories), "--cues", str(cues), "--seed", str(seed)]
    argv += ["--text-backend", text_backend]
    if sweep:
        argv.append("--sweep")
    if as_json:
        argv.append("--json")
    if output:
        argv += ["--output", output]
    raise typer.Exit(code=bench_main(argv))


@hook_app.command(
    "claude-code",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def hook_claude_code(ctx: typer.Context) -> None:
    """Claude Code UserPromptSubmit hook: inject pending recollections as context.

    Options are passed through (e.g. --url, --db, --sense-prompt); see the docs.
    """
    from madeleine.integrations.claude_code import main as hook_main

    raise typer.Exit(code=hook_main(list(ctx.args)))


if __name__ == "__main__":  # pragma: no cover
    app()
