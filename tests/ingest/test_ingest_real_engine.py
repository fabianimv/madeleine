"""End-to-end with the real engine + hash registry (skipped until those modules exist)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from madeleine.core.types import MadeleineConfig
from madeleine.ingest import create_app
from madeleine.integrations.claude_code import HEADER, main

engine_mod = pytest.importorskip("madeleine.engine")
registry_mod = pytest.importorskip("madeleine.channels.registry")

PAYLOADS = Path(__file__).resolve().parents[2] / "examples" / "payloads"


def make_engine() -> Any:
    return engine_mod.create_engine(
        MadeleineConfig(db_path=":memory:", text_backend="hash"),
        registry=registry_mod.hash_registry(),
    )


def test_real_engine_http_flow_and_hook() -> None:
    engine = make_engine()
    with TestClient(create_app(engine, api_token="")) as c:
        mem = c.post(
            "/memories",
            json={"content": "Ana proposed at the Eiffel Tower", "lat": 48.8584, "lon": 2.2945},
        )
        assert mem.status_code == 201, mem.text
        cue = c.post("/cues/generic", json={"lat": 48.8584, "lon": 2.2945, "text": "Eiffel Tower"})
        assert cue.status_code == 200, cue.text
        assert [r["memory"]["id"] for r in cue.json()["recollections"]] == [mem.json()["id"]]

        hits = c.post("/recall", json={"query": "Eiffel Tower proposal"}).json()["results"]
        assert hits[0]["memory"]["id"] == mem.json()["id"]

        # The hook drains what the webhook enqueued.
        out = io.StringIO()
        stdin = io.StringIO(json.dumps({"hook_event_name": "UserPromptSubmit", "prompt": "hi"}))
        assert main([], stdin=stdin, stdout=out, engine_factory=lambda cfg: engine) == 0
        ctx = json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"]
        assert ctx.startswith(HEADER)
        assert "Ana proposed at the Eiffel Tower" in ctx


def test_real_engine_accepts_every_example_payload() -> None:
    engine = make_engine()
    routes = {
        "cues": "/cues",
        "generic": "/cues/generic",
        "home_assistant": "/cues/home_assistant",
        "ios_shortcuts": "/cues/ios_shortcuts",
        "memories": "/memories",
        "recall": "/recall",
    }
    with TestClient(create_app(engine, api_token="")) as c:
        for path in sorted(PAYLOADS.glob("*.json")):
            prefix = max((p for p in routes if path.stem.startswith(p + "_")), key=len)
            r = c.post(routes[prefix], json=json.loads(path.read_text()))
            assert r.status_code in (200, 201), f"{path.name}: {r.text}"
    engine.close()
