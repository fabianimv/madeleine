"""GET/PATCH /memories/{id} (ADR 0003)."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_and_patch_memory(client: TestClient) -> None:
    created = client.post("/memories", json={"content": "tea at the tower", "tags": ["paris"]})
    assert created.status_code == 201
    mid = created.json()["id"]

    got = client.get(f"/memories/{mid}")
    assert got.status_code == 200
    assert got.json()["content"] == "tea at the tower"

    patched = client.patch(
        f"/memories/{mid}", json={"content": "espresso by the river", "salience": 0.9}
    )
    assert patched.status_code == 200, patched.text
    body = patched.json()
    assert body["content"] == "espresso by the river"
    assert body["context"]["salience"] == 0.9
    assert body["context"]["tags"] == ["paris"]
    assert client.get(f"/memories/{mid}").json()["content"] == "espresso by the river"


def test_patch_and_get_errors(client: TestClient) -> None:
    mid = client.post("/memories", json={"content": "x"}).json()["id"]
    assert client.get("/memories/mem_nope").status_code == 404
    assert client.patch("/memories/mem_nope", json={"content": "y"}).status_code == 404
    assert client.patch(f"/memories/{mid}", json={"salience": 2}).status_code == 422
    assert client.patch(f"/memories/{mid}", json={"content": ""}).status_code == 422
    # the sensory context is not editable
    assert client.patch(f"/memories/{mid}", json={"geohash": "u09tunq"}).status_code == 422
