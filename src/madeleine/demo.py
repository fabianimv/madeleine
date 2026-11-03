"""``madeleine demo``: the involuntary-memory loop, end to end, fully offline.

1. Remember three memories with different sensory fingerprints (place, time of day, text).
2. A cue arrives: *the same place and a similar hour*, no words at all.
3. The engine fuses the channels, the matching memory fires involuntarily and is queued.
4. The agent receives it through the MCP tool ``pending_recollections`` (in-process client).
5. The same cue again is suppressed (refractory period / habituation).

Uses the offline hash embedder and the deterministic time/geo encoders: nothing is
downloaded and nothing leaves the machine.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from madeleine.core.serde import context_from_fields
from madeleine.core.types import RecallMode

_MEMORIES: tuple[dict[str, Any], ...] = (
    {
        "content": "Sunday afternoons in grandmother's kitchen: lime-blossom tea and a madeleine",
        "lat": 48.30170,
        "lon": 1.24640,
        "at": "2019-04-14T16:00:00+02:00",  # a Sunday, years ago
        "salience": 0.9,
        "tags": ["family"],
    },
    {
        "content": "Standup with the infra team about the flaky deploy pipeline",
        "lat": 48.89190,
        "lon": 2.23800,
        "at": "2026-05-27T09:30:00+02:00",  # a Wednesday morning at the office
        "salience": 0.3,
        "tags": ["work"],
    },
    {
        "content": "Rainy evening walk along the Kamo river in Kyoto",
        "lat": 35.01160,
        "lon": 135.76810,
        "at": "2025-11-02T19:00:00+09:00",
        "salience": 0.7,
        "tags": ["travel"],
    },
)

# ~30 m from grandmother's house, a Sunday, twenty minutes later in the afternoon.
_CUE: dict[str, Any] = {"lat": 48.30195, "lon": 1.24650, "at": "2026-04-12T16:20:00+02:00"}


def _say(text: str = "") -> None:
    print(text, flush=True)


async def _drain_via_mcp(engine: Any) -> list[dict[str, Any]]:
    from mcp import Client
    from mcp_types import TextContent

    from madeleine.mcp_server import create_server

    async with Client(create_server(engine)) as client:
        tools = sorted(t.name for t in (await client.list_tools()).tools)
        _say(f"   MCP tools: {', '.join(tools)}")
        result = await client.call_tool("pending_recollections", {})
        block = result.content[0]
        assert isinstance(block, TextContent)
        if result.is_error:
            raise RuntimeError(block.text)
        data: list[dict[str, Any]] = json.loads(block.text)
        return data


def run_demo(db_path: str | None = None) -> int:
    """Run the demo; returns a process exit code (0 = the recollection arrived)."""
    from madeleine.channels.registry import hash_registry
    from madeleine.config import load_config
    from madeleine.engine import create_engine

    with tempfile.TemporaryDirectory(prefix="madeleine-demo-") as tmp:
        path = db_path or str(Path(tmp) / "demo.db")
        config = load_config(db_path=path, text_backend="hash")
        engine = create_engine(config, registry=hash_registry())
        try:
            return _run(engine, path)
        finally:
            engine.close()


def _run(engine: Any, path: str) -> int:
    now = engine.now()
    _say("madeleine demo: involuntary, cue-triggered memory (offline)")
    _say(f"database: {path}")
    _say()
    _say("1. Remembering three moments, each with its sensory fingerprint")
    ids: dict[str, str] = {}
    for item in _MEMORIES:
        fields = {k: v for k, v in item.items() if k != "content"}
        memory = engine.remember(item["content"], context=context_from_fields(now=now, **fields))
        ids[memory.id] = item["content"]
        ctx = memory.context
        _say(
            f"   - {item['content']}\n"
            f"     at {ctx.timestamp.isoformat()}  geohash {ctx.geohash}  "
            f"channels {', '.join(sorted(memory.fingerprint.channels))}"
        )
    target = next(mid for mid, c in ids.items() if "madeleine" in c)

    _say()
    _say("2. A cue arrives: same place (~30 m away), a Sunday at 16:20. No words, no query.")
    cue = engine.build_cue(context=context_from_fields(now=now, **_CUE), source="demo")
    _say(f"   cue channels: {', '.join(sorted(cue.channels))}  geohash {cue.context.geohash}")
    fired = engine.sense(cue)
    _say()
    _say("3. Fusion over the shared channels decides what surfaces on its own:")
    if not fired:
        _say("   nothing fired - unexpected; please report this as a bug")
        return 1
    for rec in fired:
        _say(f"   {rec.to_prompt_line()}")
        parts = ", ".join(f"{c}={s:.2f}" for c, s in sorted(rec.match.channel_scores.items()))
        _say(
            f"     channel similarity {parts}; salience x{rec.match.salience_factor:.2f}, "
            f"age decay x{rec.match.decay_factor:.2f}"
        )
    _say(f"   queued involuntary recollections: {engine.queue.pending_count()}")

    _say()
    _say("4. The agent receives it through the MCP tool `pending_recollections`:")
    delivered = asyncio.run(_drain_via_mcp(engine))
    for rec in delivered:
        _say(f"   <- {rec['mode']} recollection {rec['id']}: {rec['memory']['content']}")

    _say()
    _say("5. The same cue again is suppressed (refractory period, i.e. habituation):")
    again = engine.sense(engine.build_cue(context=cue.context, source="demo"))
    _say(f"   fired: {len(again)}")

    ok = (
        any(r["memory"]["id"] == target for r in delivered)
        and all(r["mode"] == RecallMode.PUSH.value for r in delivered)
        and not again
    )
    _say()
    _say("Done: the memory came back without being asked for." if ok else "Demo FAILED.")
    return 0 if ok else 1
