from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.types import Receive, Scope, Send

from madeleine import __version__
from madeleine.core.types import (
    ChannelNotAvailableError,
    MadeleineError,
    ObservationError,
    RawSignal,
    SensoryContext,
    StoreError,
)
from madeleine.ingest import create_app
from madeleine.ingest.app import DEFAULT_MAX_BODY_BYTES
from madeleine.testing import FakeClock, FakeEngine

PAYLOADS = Path(__file__).resolve().parents[2] / "examples" / "payloads"
PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5E"
    "rkJggg=="
)


def remember_tower(engine: FakeEngine, clock: FakeClock) -> str:
    return engine.remember(
        "Ana proposed at the Eiffel Tower",
        context=SensoryContext(clock.now(), geohash="u09tunq"),
    ).id


# -- meta ------------------------------------------------------------------------------


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok", "version": __version__}


# -- POST /cues ------------------------------------------------------------------------


def test_post_cue_fires_and_enqueues(
    client: TestClient, engine: FakeEngine, clock: FakeClock
) -> None:
    mem_id = remember_tower(engine, clock)
    r = client.post("/cues", json={"geohash": "u09tunq", "source": "test"})
    assert r.status_code == 200
    body = r.json()
    assert body["cue_id"].startswith("cue_")
    [rec] = body["recollections"]
    assert rec["memory"]["id"] == mem_id
    assert rec["cue_source"] == "test"
    assert rec["mode"] == "push"
    assert rec["prompt_line"].startswith("[madeleine 1.00 via geo+time] Ana proposed")
    assert engine.queue.pending_count() == 1


def test_post_cue_default_source_and_no_match(client: TestClient, engine: FakeEngine) -> None:
    r = client.post("/cues", json={"text": "nothing remembered yet"})
    assert r.status_code == 200
    assert r.json()["recollections"] == []


def test_post_cue_with_signals(client: TestClient, engine: FakeEngine, clock: FakeClock) -> None:
    engine.remember("rain drumming on a tin roof in Kyoto")
    r = client.post(
        "/cues",
        json={
            "signals": [
                {"channel": "text", "value": "rain drumming on a tin roof"},
                {"channel": "image", "data_b64": PNG_B64, "mime_type": "image/png"},
            ],
            "at": "2026-06-01T12:00:00Z",
            "tags": ["x"],
        },
    )
    assert r.status_code == 200
    [rec] = r.json()["recollections"]
    assert rec["cue_channels"] == ["image", "text", "time"]


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({}, 422),  # empty cue
        ({"text": "x", "at": "not a date"}, 422),
        ({"text": "x", "lat": 1.0}, 422),
        ({"text": "x", "geohash": "aaaa"}, 422),
        ({"latitude": 1.0}, 422),  # unknown field (strict body)
        ({"signals": [{"channel": "BAD", "value": 1}]}, 422),
        ({"signals": [{"channel": "smell", "value": "lavender"}]}, 400),  # unavailable
        ({"signals": [{"channel": "image", "data_b64": "@@"}]}, 422),
        ({"text": "x", "salience": 2}, 422),
    ],
)
def test_post_cue_errors(client: TestClient, body: dict[str, Any], status: int) -> None:
    r = client.post("/cues", json=body)
    assert r.status_code == status, r.text
    assert r.json()["detail"]


def test_invalid_json_is_422(client: TestClient) -> None:
    r = client.post("/cues", content=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 422


# -- POST /cues/{adapter} --------------------------------------------------------------


def test_adapter_routes(client: TestClient, engine: FakeEngine, clock: FakeClock) -> None:
    mem_id = remember_tower(engine, clock)
    r = client.post("/cues/generic", json={"geohash": "u09tunq", "source": "ignored"})
    assert r.status_code == 200
    [rec] = r.json()["recollections"]
    assert rec["memory"]["id"] == mem_id
    assert rec["cue_source"] == "webhook:generic"

    r = client.post(
        "/cues/home_assistant",
        json={"entity_id": "person.ana", "state": "home", "latitude": 48.8584, "longitude": 2.2945},
    )
    assert r.status_code == 200


def test_adapter_source_is_recorded(client: TestClient, engine: FakeEngine) -> None:
    engine.remember("Kitchen speaker started playing Clair de lune")
    r = client.post(
        "/cues/home_assistant",
        json={
            "entity_id": "media_player.kitchen",
            "state": "playing",
            "attributes": {"friendly_name": "Kitchen speaker", "media_title": "Clair de lune"},
        },
    )
    [rec] = r.json()["recollections"]
    assert rec["cue_source"] == "webhook:home_assistant"


def test_unknown_adapter_is_404(client: TestClient) -> None:
    r = client.post("/cues/telegram", json={"text": "hi"})
    assert r.status_code == 404
    assert "home_assistant" in r.json()["detail"]


def test_adapter_body_must_be_object(client: TestClient) -> None:
    assert client.post("/cues/generic", json=[1, 2]).status_code == 422


def test_adapter_payload_errors_are_422(client: TestClient) -> None:
    r = client.post("/cues/ios_shortcuts", json={"text": "x", "date": "1 Jun 2026 at 21:30"})
    assert r.status_code == 422
    assert "ISO 8601" in r.json()["detail"]


def test_custom_adapters_replace_defaults(engine: FakeEngine) -> None:
    class Echo:
        name = "echo"

        def to_signals(self, payload: Any) -> tuple[list[RawSignal], SensoryContext | None]:
            return [RawSignal("text", str(payload["say"]))], None

    with TestClient(create_app(engine, adapters=[Echo()])) as c:
        assert c.post("/cues/echo", json={"say": "hello"}).status_code == 200
        assert c.post("/cues/generic", json={"text": "x"}).status_code == 404


def test_duplicate_adapter_names_rejected(engine: FakeEngine) -> None:
    from madeleine.ingest import GenericWebhookAdapter

    with pytest.raises(ValueError, match="duplicate"):
        create_app(engine, adapters=[GenericWebhookAdapter(), GenericWebhookAdapter()])


# -- memories / recall / pending / delete ----------------------------------------------


def test_post_memory_and_delete(client: TestClient, engine: FakeEngine, clock: FakeClock) -> None:
    r = client.post(
        "/memories",
        json={
            "content": "Ana proposed at the Eiffel Tower",
            "lat": 48.8584,
            "lon": 2.2945,
            "at": "2024-06-01T21:30:00+02:00",
            "tags": ["ana"],
            "salience": 1.0,
            "media_ref": "file:///photos/proposal.jpg",
            "metadata": {"album": "paris"},
            "signals": [{"channel": "biometric", "value": {"heart_rate": 120}}],
        },
    )
    assert r.status_code == 201, r.text
    mem = r.json()
    assert mem["context"]["tags"] == ["ana"]
    assert mem["context"]["timestamp"] == "2024-06-01T21:30:00+02:00"
    assert mem["metadata"] == {"album": "paris"}
    assert mem["channels"] == ["biometric", "geo", "text", "time"]
    stored = engine.get(mem["id"])
    assert stored.media_ref == "file:///photos/proposal.jpg"

    assert client.delete(f"/memories/{mem['id']}").json() == {"deleted": True}
    assert client.delete(f"/memories/{mem['id']}").status_code == 404


def test_post_memory_without_context_uses_engine_clock(
    client: TestClient, engine: FakeEngine, clock: FakeClock
) -> None:
    mem = client.post("/memories", json={"content": "plain"}).json()
    assert mem["context"]["timestamp"] == clock.now().isoformat()


def test_post_memory_does_not_persist_raw_media(client: TestClient, engine: FakeEngine) -> None:
    mem = client.post(
        "/memories",
        json={"content": "a photo", "signals": [{"channel": "image", "data_b64": PNG_B64}]},
    ).json()
    assert engine.get(mem["id"]).fingerprint.observations["image"].data is None


@pytest.mark.parametrize(
    "body",
    [{}, {"content": ""}, {"content": "x", "at": "nope"}, {"content": "x", "metadata": {"a": 1}}],
)
def test_post_memory_errors(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/memories", json=body).status_code == 422


def test_recall(client: TestClient, engine: FakeEngine, clock: FakeClock) -> None:
    six_hours_ago = clock.now() - timedelta(hours=6)
    engine.remember(
        "rain on a tin roof in Kyoto", context=SensoryContext(six_hours_ago, tags={"t"})
    )
    engine.remember("dinner at the tower", context=SensoryContext(clock.now(), geohash="u09tunq"))
    r = client.post("/recall", json={"query": "rain roof", "k": 1})
    assert r.status_code == 200
    [hit] = r.json()["results"]
    assert hit["memory"]["content"] == "rain on a tin roof in Kyoto"
    assert set(hit["channel_scores"]) == {"text"}  # no time without "at"

    r = client.post("/recall", json={"geohash": "u09tunq", "at": "2026-06-01T12:00:00Z"})
    assert r.json()["results"][0]["memory"]["content"] == "dinner at the tower"
    assert set(r.json()["results"][0]["channel_scores"]) == {"geo", "time"}

    r = client.post("/recall", json={"query": "rain", "tags": ["t"], "channels": ["text"]})
    assert [h["memory"]["content"] for h in r.json()["results"]] == ["rain on a tin roof in Kyoto"]

    r = client.post("/recall", json={"query": "rain", "min_score": 0.99})
    assert r.json()["results"] == []
    assert engine.queue.pending_count() == 0  # pull has no side effects


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"k": 0, "query": "x"},
        {"query": "x", "signals": [{"channel": "text", "value": "y"}]},
        {"query": "x", "min_score": 2},
    ],
)
def test_recall_errors(client: TestClient, body: dict[str, Any]) -> None:
    assert client.post("/recall", json=body).status_code == 422


def test_pending_peek_drain_limit(client: TestClient, engine: FakeEngine, clock: FakeClock) -> None:
    engine.remember("one two three")
    engine.remember("one two four")
    client.post("/cues", json={"text": "one two three four"})
    assert len(client.get("/recollections/pending?drain=false").json()["recollections"]) == 2
    first = client.get("/recollections/pending?limit=1").json()["recollections"]
    assert len(first) == 1
    assert "prompt_line" in first[0]
    assert len(client.get("/recollections/pending").json()["recollections"]) == 1
    assert client.get("/recollections/pending").json() == {"recollections": []}
    assert client.get("/recollections/pending?limit=0").status_code == 422


def test_channel_not_available_from_engine_is_400(engine: FakeEngine, clock: FakeClock) -> None:
    class Broken(FakeEngine):
        def sense(self, cue: Any) -> Any:
            raise ChannelNotAvailableError(
                "channel 'image' needs: pip install 'madeleine-mcp[image]'"
            )

    with TestClient(create_app(Broken(clock=clock))) as c:
        r = c.post("/cues", json={"text": "x"})
    assert r.status_code == 400
    assert "madeleine-mcp[image]" in r.json()["detail"]


# -- auth ------------------------------------------------------------------------------


def test_auth_token(engine: FakeEngine) -> None:
    with TestClient(create_app(engine, api_token="s3cret")) as c:
        assert c.get("/healthz").status_code == 200
        r = c.get("/recollections/pending")
        assert r.status_code == 401
        assert r.headers["www-authenticate"] == "Bearer"
        assert (
            c.get("/recollections/pending", headers={"Authorization": "Bearer nope"}).status_code
            == 401
        )
        assert (
            c.get("/recollections/pending", headers={"Authorization": "Basic s3cret"}).status_code
            == 401
        )
        assert c.post("/cues", json={"text": "x"}).status_code == 401
        assert c.get("/openapi.json").status_code == 401
        ok = {"Authorization": "bearer s3cret"}
        assert c.get("/recollections/pending", headers=ok).status_code == 200
        assert c.post("/cues", json={"text": "x"}, headers=ok).status_code == 200


def test_auth_token_from_env(engine: FakeEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MADELEINE_API_TOKEN", "from-env")
    with TestClient(create_app(engine)) as c:
        assert c.get("/recollections/pending").status_code == 401
        assert (
            c.get(
                "/recollections/pending", headers={"Authorization": "Bearer from-env"}
            ).status_code
            == 200
        )
    with TestClient(create_app(engine, api_token="")) as c:  # explicit "" disables
        assert c.get("/recollections/pending").status_code == 200


def test_auth_off_by_default(client: TestClient) -> None:
    assert client.get("/recollections/pending").status_code == 200


# -- body size -------------------------------------------------------------------------


def test_body_size_limit(engine: FakeEngine) -> None:
    with TestClient(create_app(engine, max_body_bytes=200)) as c:
        small = c.post("/cues", json={"text": "short"})
        assert small.status_code == 200
        big = c.post("/cues", json={"text": "x" * 500})
        assert big.status_code == 413
        assert "200 bytes" in big.json()["detail"]

        def chunks() -> Iterator[bytes]:  # no Content-Length: counted while streaming
            yield b'{"text": "'
            yield b"y" * 300
            yield b'"}'

        streamed = c.post("/cues", content=chunks(), headers={"content-type": "application/json"})
        assert streamed.status_code == 413


def test_body_size_limit_env_and_default(
    engine: FakeEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert DEFAULT_MAX_BODY_BYTES == 10 * 1024 * 1024
    monkeypatch.setenv("MADELEINE_MAX_BODY_BYTES", "50")
    with TestClient(create_app(engine)) as c:
        assert c.post("/cues", json={"text": "x" * 100}).status_code == 413
    monkeypatch.setenv("MADELEINE_MAX_BODY_BYTES", "lots")
    with pytest.raises(ValueError, match="MADELEINE_MAX_BODY_BYTES"):
        create_app(engine)
    with pytest.raises(ValueError, match="max_body_bytes"):
        create_app(engine, max_body_bytes=0)


def test_large_photo_under_default_limit(client: TestClient) -> None:
    photo = base64.b64encode(b"\x89PNG" + b"\0" * 3_000_000).decode()
    r = client.post("/cues/ios_shortcuts", json={"photo_b64": photo, "latitude": 1, "longitude": 2})
    assert r.status_code == 200


# -- mounts and lifespans --------------------------------------------------------------


def test_mounts_and_lifespans(engine: FakeEngine) -> None:
    events: list[str] = []

    async def sub_app(scope: Scope, receive: Receive, send: Send) -> None:
        assert scope["type"] == "http"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": scope["path"].encode()})

    def make_lifespan(name: str) -> Any:
        @asynccontextmanager
        async def lifespan() -> AsyncIterator[None]:
            events.append(f"enter {name}")
            yield
            events.append(f"exit {name}")

        return lifespan

    app = create_app(
        engine,
        mounts={"/mcp": sub_app},
        lifespans=[make_lifespan("a"), make_lifespan("b")],
        api_token="t",
    )
    with TestClient(app) as c:
        assert events == ["enter a", "enter b"]
        assert c.get("/mcp/x").status_code == 401  # mounted apps are guarded too
        r = c.get("/mcp/x", headers={"Authorization": "Bearer t"})
        assert r.status_code == 200
        assert r.text.endswith("/x")
    assert events == ["enter a", "enter b", "exit b", "exit a"]


# -- example payloads ------------------------------------------------------------------

ROUTES = {
    "cues": "/cues",
    "generic": "/cues/generic",
    "home_assistant": "/cues/home_assistant",
    "ios_shortcuts": "/cues/ios_shortcuts",
    "memories": "/memories",
    "recall": "/recall",
}


def route_for(path: Path) -> str:
    matches = [p for p in ROUTES if path.stem.startswith(p + "_")]
    assert matches, f"{path.name}: unknown payload prefix"
    return ROUTES[max(matches, key=len)]


EXAMPLES = sorted(PAYLOADS.glob("*.json"))


def test_every_route_has_an_example() -> None:
    assert {route_for(p) for p in EXAMPLES} == set(ROUTES.values())


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_payload_is_accepted(client: TestClient, path: Path) -> None:
    r = client.post(route_for(path), json=json.loads(path.read_text()))
    assert r.status_code in (200, 201), r.text


def test_curl_script_covers_every_payload() -> None:
    script = (PAYLOADS.parent / "curl.sh").read_text()
    for path in EXAMPLES:
        assert f"{route_for(path)} {path.name}" in script, path.name
    for route in ("/healthz", "/recollections/pending", "/memories/${MEMORY_ID}"):
        assert route in script


def test_home_assistant_yaml_parses() -> None:
    yaml = pytest.importorskip("yaml")

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("!secret", lambda loader, node: f"secret:{node.value}")
    config = yaml.load(
        (PAYLOADS.parent / "home_assistant" / "configuration.yaml").read_text(), Loader=Loader
    )
    assert config["rest_command"]["madeleine_cue"]["url"].endswith("/cues/home_assistant")
    assert len(config["automation"]) == 2


def test_claude_code_settings_example() -> None:
    settings = json.loads((PAYLOADS.parent / "claude_code" / "settings.json").read_text())
    [entry] = settings["hooks"]["UserPromptSubmit"]
    [hook] = entry["hooks"]
    assert hook["type"] == "command"
    assert hook["command"] == "uvx madeleine-mcp hook claude-code"


# -- R12: wrong-typed webhook fields are 4xx, never 500 --------------------------------

_WRONG_TYPES: list[Any] = [123, 1.5, True, [], ["x"], {}, {"a": 1}, ""]
_TOP_FIELDS = [
    "at", "date", "time_fired", "text", "geohash", "lat", "lon", "latitude", "longitude",
    "tags", "salience", "biometric", "signals", "entity_id", "state", "old_state", "event",
    "event_type", "zone", "attributes", "data", "heart_rate", "focus", "battery",
    "photo_b64", "photo_mime_type", "audio_b64", "audio_mime_type",
]  # fmt: skip
_SIGNAL_FIELDS = ["channel", "value", "data_b64", "mime_type", "uri", "at"]


def _fuzz_bodies() -> Iterator[dict[str, Any]]:
    for field in _TOP_FIELDS:
        for bad in _WRONG_TYPES:
            yield {"text": "hello", "entity_id": "light.hall", "state": "on", field: bad}
    for field in _SIGNAL_FIELDS:
        for bad in _WRONG_TYPES:
            yield {"signals": [{"channel": "text", "value": "x", field: bad}]}


@pytest.mark.parametrize("adapter", ["generic", "home_assistant", "ios_shortcuts"])
def test_adapter_routes_never_500_on_wrong_types(engine: FakeEngine, adapter: str) -> None:
    with TestClient(create_app(engine), raise_server_exceptions=False) as c:
        for body in _fuzz_bodies():
            r = c.post(f"/cues/{adapter}", json=body)
            assert r.status_code < 500, (body, r.text)


def test_signal_at_must_be_a_string(client: TestClient) -> None:
    r = client.post(
        "/cues/generic", json={"signals": [{"channel": "text", "value": "x", "at": 123}]}
    )
    assert r.status_code == 422
    assert "signals[0].at" in r.json()["detail"]


# -- R10: error mapping ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (StoreError("text model changed: re-embed or use another database"), 409),
        (MadeleineError("something deliberate went wrong"), 500),
        (ObservationError("bad observation"), 422),
        (ChannelNotAvailableError("pip install 'madeleine-mcp[image]'"), 400),
    ],
)
def test_madeleine_errors_are_mapped_with_their_message(
    clock: FakeClock, exc: Exception, status: int
) -> None:
    class Failing(FakeEngine):
        def remember(self, *args: Any, **kwargs: Any) -> Any:
            raise exc

    with TestClient(create_app(Failing(clock=clock)), raise_server_exceptions=False) as c:
        r = c.post("/memories", json={"content": "x"})
    assert r.status_code == status
    assert r.json() == {"detail": str(exc)}
