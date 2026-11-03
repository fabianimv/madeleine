"""The local-file policy is enforced by the media adapters themselves (R2/R3, R17).

Every way of naming a server file (``uri`` with ``file:``/no scheme/a Windows drive, a string
``value``, a local ``uri`` next to bytes) is refused unless the adapter was built with
``allow_local_files=True``, before any filesystem access and with a message that does not
depend on whether the file exists.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from madeleine.channels import _media, registry
from madeleine.channels._media import MAX_MEDIA_BYTES, checked_local_file, read_media_bytes
from madeleine.channels.audio import AudioAdapter, make_audio_channel
from madeleine.channels.hashing import HashEmbedder
from madeleine.channels.image import ImageAdapter, make_image_channel
from madeleine.channels.registry import default_registry, hash_registry
from madeleine.core.types import (
    AUDIO,
    IMAGE,
    MadeleineConfig,
    Observation,
    ObservationError,
    RawSignal,
)
from madeleine.engine import create_engine

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 24

Adapter = ImageAdapter | AudioAdapter
MEDIA: list[tuple[str, Callable[..., Adapter], bytes, str]] = [
    (IMAGE, ImageAdapter, PNG, "a.png"),
    (AUDIO, AudioAdapter, WAV, "a.wav"),
]
IDS = [m[0] for m in MEDIA]


def _refs(tmp_path: Path, name: str) -> list[str]:
    existing = tmp_path / name
    missing = tmp_path / ("missing-" + name)
    return [
        existing.as_uri(),
        missing.as_uri(),
        str(existing),
        str(missing),
        "relative/" + name,
        "~/" + name,
        "file:/etc/passwd",
        "file://localhost/etc/passwd",
        "C:\\Users\\me\\" + name,
        "/etc/passwd",
    ]


def _message(fn: Callable[[], object]) -> str:
    with pytest.raises(ObservationError) as info:
        fn()
    return str(info.value)


@pytest.mark.parametrize(("channel", "make", "data", "name"), MEDIA, ids=IDS)
def test_every_local_reference_is_refused_by_default(
    tmp_path: Path, channel: str, make: Callable[..., Adapter], data: bytes, name: str
) -> None:
    (tmp_path / name).write_bytes(data)
    adapter = make()
    assert adapter.allow_local_files is False
    messages = set()
    for ref in _refs(tmp_path, name):
        messages.add(_message(lambda r=ref: adapter.normalize(RawSignal(channel, None, uri=r))))
        messages.add(_message(lambda r=ref: adapter.normalize(RawSignal(channel, r))))
        # A local uri next to bytes is a label, but still a server path: refused too.
        messages.add(_message(lambda r=ref: adapter.normalize(RawSignal(channel, data, uri=r))))
    # One constant message: existing and missing files are indistinguishable.
    assert len(messages) == 1
    (message,) = messages
    assert "disabled" in message
    assert "data_b64" in message
    assert str(tmp_path) not in message


@pytest.mark.parametrize(("channel", "make", "data", "name"), MEDIA, ids=IDS)
def test_policy_is_checked_before_any_filesystem_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    channel: str,
    make: Callable[..., Adapter],
    data: bytes,
    name: str,
) -> None:
    def no_io(*_a: object, **_k: object) -> object:
        raise AssertionError("the filesystem was touched")

    for attr in ("stat", "is_file", "exists", "resolve", "read_bytes", "open"):
        monkeypatch.setattr(Path, attr, no_io)
    adapter = make()
    for ref in _refs(tmp_path, name):
        with pytest.raises(ObservationError, match="disabled"):
            adapter.normalize(RawSignal(channel, ref))
        with pytest.raises(ObservationError, match="disabled"):
            adapter.normalize(RawSignal(channel, None, uri=ref))


@pytest.mark.parametrize(("channel", "make", "data", "name"), MEDIA, ids=IDS)
def test_bytes_always_work_and_remote_schemes_stay_rejected(
    channel: str, make: Callable[..., Adapter], data: bytes, name: str
) -> None:
    for adapter in (make(), make(allow_local_files=True)):
        assert adapter.normalize(RawSignal(channel, data)).data == data
        labelled = adapter.normalize(RawSignal(channel, data, uri="https://example.org/" + name))
        assert labelled.uri == "https://example.org/" + name
        for ref in ("https://example.org/" + name, "s3://bucket/" + name):
            with pytest.raises(ObservationError, match="never downloads"):
                adapter.normalize(RawSignal(channel, None, uri=ref))
            with pytest.raises(ObservationError, match="never downloads"):
                adapter.normalize(RawSignal(channel, ref))


@pytest.mark.parametrize(("channel", "make", "data", "name"), MEDIA, ids=IDS)
def test_local_files_work_when_allowed(
    tmp_path: Path, channel: str, make: Callable[..., Adapter], data: bytes, name: str
) -> None:
    f = tmp_path / name
    f.write_bytes(data)
    adapter = make(allow_local_files=True)
    assert adapter.normalize(RawSignal(channel, str(f))).uri == f.resolve().as_uri()
    assert adapter.normalize(RawSignal(channel, None, uri=f.as_uri())).data is None
    with pytest.raises(ObservationError, match="file not found"):
        adapter.normalize(RawSignal(channel, str(tmp_path / "nope" / name)))


# -- R17: files are capped like in-memory payloads ---------------------------------------


@pytest.mark.parametrize(("channel", "make", "data", "name"), MEDIA, ids=IDS)
def test_oversized_files_are_refused(
    tmp_path: Path, channel: str, make: Callable[..., Adapter], data: bytes, name: str
) -> None:
    f = tmp_path / name
    f.write_bytes(data)
    with pytest.raises(ObservationError, match="larger than 4 bytes"):
        make(allow_local_files=True, max_bytes=4).normalize(RawSignal(channel, str(f)))
    big = tmp_path / ("big-" + name)
    with big.open("wb") as fh:  # sparse: no real disk use
        fh.truncate(MAX_MEDIA_BYTES + 1)
    with pytest.raises(ObservationError, match="larger than"):
        make(allow_local_files=True).normalize(RawSignal(channel, str(big)))


def test_embedder_reads_refuse_oversized_files(tmp_path: Path) -> None:
    big = tmp_path / "big.png"
    with big.open("wb") as fh:
        fh.truncate(MAX_MEDIA_BYTES + 1)
    with pytest.raises(ObservationError, match="larger than"):
        read_media_bytes(Observation(IMAGE, uri=big.as_uri()))
    small = tmp_path / "small.png"
    small.write_bytes(PNG)
    assert read_media_bytes(Observation(IMAGE, uri=small.as_uri())) == PNG
    with pytest.raises(ObservationError, match="larger than 4 bytes"):
        checked_local_file(small.as_uri(), IMAGE, max_bytes=4)


# -- the hash embedder never reads files --------------------------------------------------


def test_hash_embedder_never_reads_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    f = tmp_path / "secret.png"
    f.write_bytes(PNG)

    def no_read(*_a: object, **_k: object) -> bytes:
        raise AssertionError("HashEmbedder read a file")

    monkeypatch.setattr(Path, "read_bytes", no_read)
    monkeypatch.setattr(Path, "open", no_read)
    emb = HashEmbedder(IMAGE)
    vec = emb.embed(Observation(IMAGE, uri=f.as_uri()))
    assert vec == emb.content_vector(b"uri:" + f.as_uri().encode())


# -- wiring: factories and registries -------------------------------------------------------


def test_factories_take_the_flag_from_config_or_argument() -> None:
    on = MadeleineConfig(allow_local_files=True)
    for make in (make_image_channel, make_audio_channel):
        emb = HashEmbedder(IMAGE if make is make_image_channel else AUDIO)
        assert make(embedder=emb).adapter.allow_local_files is False  # type: ignore[attr-defined]
        assert make(on, embedder=emb).adapter.allow_local_files is True  # type: ignore[attr-defined]
        explicit = make(on, embedder=emb, allow_local_files=False)
        assert explicit.adapter.allow_local_files is False  # type: ignore[attr-defined]


def test_hash_registry_flag() -> None:
    for allow in (False, True):
        reg = hash_registry(allow_local_files=allow)
        assert reg.get(IMAGE).adapter.allow_local_files is allow  # type: ignore[attr-defined]
        assert reg.get(AUDIO).adapter.allow_local_files is allow  # type: ignore[attr-defined]
    assert hash_registry().get(IMAGE).adapter.allow_local_files is False  # type: ignore[attr-defined]


def test_default_registry_passes_config_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "missing_modules", lambda _extra: [])  # pretend extras exist
    for allow in (False, True):
        reg = default_registry(MadeleineConfig(allow_local_files=allow, text_backend="hash"))
        assert reg.get(IMAGE).adapter.allow_local_files is allow  # type: ignore[attr-defined]
        assert reg.get(AUDIO).adapter.allow_local_files is allow  # type: ignore[attr-defined]


def test_engine_cannot_be_made_to_read_or_probe_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The bypasses from the review, through the engine (any front-end ends up here)."""
    secret = tmp_path / "secret.png"
    secret.write_bytes(PNG)
    reads: list[str] = []
    real_read = Path.read_bytes

    def spy(self: Path) -> bytes:
        reads.append(str(self))
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", spy)
    monkeypatch.setattr(_media.Path, "read_bytes", spy)
    engine = create_engine(MadeleineConfig(db_path=":memory:"), registry=hash_registry())
    try:
        messages = set()
        for signal in (
            RawSignal(IMAGE, None, uri=secret.as_uri()),
            RawSignal(IMAGE, str(secret)),
            RawSignal(IMAGE, str(tmp_path / "missing.png")),
            RawSignal(AUDIO, "/etc/hostname"),
            RawSignal(AUDIO, "/nonexistent/x.wav"),
        ):
            messages.add(_message(lambda s=signal: engine.remember("x", [s])).split(":", 1)[1])
            messages.add(_message(lambda s=signal: engine.build_cue([s])).split(":", 1)[1])
        assert len(messages) == 1
        assert reads == []
        assert engine.store.count() == 0
    finally:
        engine.close()
