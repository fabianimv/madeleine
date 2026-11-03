"""Security regressions for the HTTP ingestion API (review findings R1, R2)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.types import Receive, Scope, Send

from madeleine.ingest import create_app
from madeleine.ingest.app import LOOPBACK_HOSTS, default_allowed_hosts
from madeleine.testing import FakeEngine

# -- R1: Host / Origin allow-list (DNS rebinding) --------------------------------------


async def _echo_app(scope: Scope, receive: Receive, send: Send) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"mounted"})


@pytest.fixture
def loopback(engine: FakeEngine, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """The default allow-list only (no env additions), reached as http://127.0.0.1:8765."""
    monkeypatch.delenv("MADELEINE_ALLOWED_HOSTS", raising=False)
    app = create_app(engine, mounts={"/mcp": _echo_app})
    with TestClient(app, base_url="http://127.0.0.1:8765") as c:
        yield c


ROUTES: list[tuple[str, str, dict[str, Any] | None]] = [
    ("GET", "/healthz", None),
    ("POST", "/recall", {"query": "planted"}),
    ("POST", "/memories", {"content": "planted"}),
    ("POST", "/cues", {"text": "x"}),
    ("GET", "/recollections/pending", None),
    ("GET", "/mcp/", None),
]


@pytest.mark.parametrize(("method", "path", "body"), ROUTES)
@pytest.mark.parametrize("host", ["evil.example:8765", "evil.example", "127.0.0.1.evil.example"])
def test_evil_host_is_421(
    loopback: TestClient, method: str, path: str, body: dict[str, Any] | None, host: str
) -> None:
    r = loopback.request(method, path, json=body, headers={"Host": host})
    assert r.status_code == 421, r.text
    assert r.json() == {"detail": "Invalid Host header"}


@pytest.mark.parametrize(("method", "path", "body"), ROUTES)
@pytest.mark.parametrize(
    "origin", ["http://evil.example", "https://evil.example:8765", "null", "http://127.0.0.1.evil"]
)
def test_evil_origin_is_403(
    loopback: TestClient, method: str, path: str, body: dict[str, Any] | None, origin: str
) -> None:
    r = loopback.request(method, path, json=body, headers={"Origin": origin})
    assert r.status_code == 403, r.text
    assert r.json() == {"detail": "Invalid Origin header"}


@pytest.mark.parametrize(
    "host", ["127.0.0.1:8765", "localhost:8765", "LOCALHOST", "[::1]:8765", "[::1]", "127.0.0.1"]
)
def test_loopback_hosts_pass(loopback: TestClient, host: str) -> None:
    assert loopback.get("/healthz", headers={"Host": host}).status_code == 200
    assert loopback.post("/recall", json={"query": "x"}, headers={"Host": host}).status_code == 200
    assert loopback.get("/mcp/", headers={"Host": host}).text == "mounted"


@pytest.mark.parametrize(
    "origin", ["http://127.0.0.1:8765", "http://localhost:3000", "http://[::1]"]
)
def test_loopback_origins_pass(loopback: TestClient, origin: str) -> None:
    r = loopback.post("/recall", json={"query": "x"}, headers={"Origin": origin})
    assert r.status_code == 200, r.text


def test_missing_host_header_is_421(loopback: TestClient) -> None:
    import anyio

    app = loopback.app
    sent: list[dict[str, Any]] = []

    async def run() -> None:
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/healthz",
            "raw_path": b"/healthz",
            "root_path": "",
            "query_string": b"",
            "headers": [],
            "http_version": "1.0",
            "scheme": "http",
            "server": ("127.0.0.1", 8765),
        }

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        await app(scope, receive, send)  # type: ignore[arg-type]

    anyio.run(run)
    assert sent[0]["status"] == 421


def test_default_allow_list_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MADELEINE_ALLOWED_HOSTS", raising=False)
    assert default_allowed_hosts() == list(LOOPBACK_HOSTS)
    assert set(LOOPBACK_HOSTS) == {"127.0.0.1", "localhost", "[::1]", "::1"}
    monkeypatch.setenv("MADELEINE_ALLOWED_HOSTS", " madeleine.lan , 192.168.1.20,")
    assert default_allowed_hosts() == [*LOOPBACK_HOSTS, "madeleine.lan", "192.168.1.20"]


def test_env_hosts_are_allowed(engine: FakeEngine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MADELEINE_ALLOWED_HOSTS", "madeleine.lan")
    with TestClient(create_app(engine), base_url="http://madeleine.lan:8765") as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/healthz", headers={"Origin": "http://madeleine.lan"}).status_code == 200
        assert c.get("/healthz", headers={"Host": "evil.example"}).status_code == 421


def test_explicit_allowed_hosts_replace_defaults(engine: FakeEngine) -> None:
    app = create_app(engine, allowed_hosts=["192.168.1.20"], mounts={"/mcp": _echo_app})
    with TestClient(app, base_url="http://192.168.1.20:8765") as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/mcp/").text == "mounted"
        assert c.get("/healthz", headers={"Host": "localhost"}).status_code == 421
        assert c.get("/mcp/", headers={"Host": "evil.example"}).status_code == 421
        assert c.get("/mcp/", headers={"Origin": "http://evil.example"}).status_code == 403


def test_star_disables_the_check(engine: FakeEngine) -> None:
    with TestClient(create_app(engine, allowed_hosts=["*"])) as c:
        headers = {"Host": "evil.example", "Origin": "http://evil.example"}
        assert c.post("/recall", json={"query": "x"}, headers=headers).status_code == 200


def test_host_check_runs_before_auth(engine: FakeEngine) -> None:
    with TestClient(create_app(engine, api_token="t")) as c:
        assert c.get("/recollections/pending", headers={"Host": "evil.example"}).status_code == 421
        assert c.get("/healthz", headers={"Host": "evil.example"}).status_code == 421
