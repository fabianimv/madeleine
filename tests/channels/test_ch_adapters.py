"""Adapters: valid inputs normalise, invalid inputs raise ObservationError with clear messages."""

from __future__ import annotations

import struct
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from madeleine.channels.audio import AudioAdapter, decode_audio, resample
from madeleine.channels.biometric import BiometricAdapter, BiometricEncoder
from madeleine.channels.geo import GeoAdapter
from madeleine.channels.hashing import HashEmbedder
from madeleine.channels.image import ImageAdapter, parse_image_model
from madeleine.channels.smell import SmellAdapter, make_smell_channel
from madeleine.channels.text import TextAdapter
from madeleine.channels.time import TimeAdapter
from madeleine.core.types import (
    AUDIO,
    BIOMETRIC,
    GEO,
    IMAGE,
    SMELL,
    TEXT,
    TIME,
    ChannelNotAvailableError,
    Observation,
    ObservationError,
    RawSignal,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def test_text_adapter() -> None:
    assert TextAdapter().normalize(RawSignal(TEXT, "  hello  ")).text == "hello"
    with pytest.raises(ObservationError, match="empty"):
        TextAdapter().normalize(RawSignal(TEXT, "   "))
    with pytest.raises(ObservationError, match="must be a string"):
        TextAdapter().normalize(RawSignal(TEXT, 42))


def test_time_adapter_inputs() -> None:
    a = TimeAdapter()
    tz = timezone(timedelta(hours=2))
    dt = datetime(2026, 6, 1, 21, 30, tzinfo=tz)
    assert a.normalize(RawSignal(TIME, dt)).timestamp == dt
    parsed = a.normalize(RawSignal(TIME, "2026-06-01T21:30:00+02:00")).timestamp
    assert parsed == dt
    assert parsed is not None
    assert parsed.utcoffset() == timedelta(hours=2)
    naive = a.normalize(RawSignal(TIME, "2026-06-01T21:30:00")).timestamp
    assert naive is not None
    assert naive.tzinfo is UTC
    assert a.normalize(RawSignal(TIME, 0)).timestamp == datetime(1970, 1, 1, tzinfo=UTC)
    assert a.normalize(RawSignal(TIME, None, observed_at=dt)).timestamp == dt
    with pytest.raises(ObservationError, match="invalid ISO-8601"):
        a.normalize(RawSignal(TIME, "yesterday-ish"))
    with pytest.raises(ObservationError, match="empty string"):
        a.normalize(RawSignal(TIME, ""))
    with pytest.raises(ObservationError, match="needs a datetime"):
        a.normalize(RawSignal(TIME, None))
    with pytest.raises(ObservationError, match="boolean"):
        a.normalize(RawSignal(TIME, True))
    with pytest.raises(ObservationError, match="got list"):
        a.normalize(RawSignal(TIME, [2026, 6, 1]))


def test_geo_adapter_inputs() -> None:
    a = GeoAdapter()
    o = a.normalize(RawSignal(GEO, " U09TUNQ "))
    assert o.text == "u09tunq"
    assert not o.numbers
    o = a.normalize(RawSignal(GEO, (48.8584, 2.2945)))
    assert o.text is not None
    assert len(o.text) == 9
    assert o.text.startswith("u09tunq")
    assert dict(o.numbers) == {"lat": 48.8584, "lon": 2.2945}
    assert a.normalize(RawSignal(GEO, [48.8584, 2.2945])) == o
    assert a.normalize(RawSignal(GEO, {"lat": 48.8584, "lon": 2.2945})) == o
    for bad, msg in [
        ("aaaa", "invalid geohash"),
        ("", "invalid geohash"),
        ((91.0, 0.0), "out of range"),
        ((1.0,), "exactly two"),
        ({"lat": 1.0}, "both 'lat' and 'lon'"),
        ((True, 1.0), "must be a number"),
        ((float("nan"), 1.0), "finite"),
        (("x", 1.0), "not a number"),
        (42, "geohash string"),
    ]:
        with pytest.raises(ObservationError, match=msg):
            a.normalize(RawSignal(GEO, bad))


def test_biometric_adapter_inputs() -> None:
    a = BiometricAdapter()
    o = a.normalize(RawSignal(BIOMETRIC, {"heart_rate": 92, "eda": 3.5}))
    assert dict(o.numbers) == {"heart_rate": 92.0, "eda": 3.5}
    with pytest.raises(ObservationError, match=r"unknown biometric feature 'pulse'.*heart_rate"):
        a.normalize(RawSignal(BIOMETRIC, {"pulse": 80}))
    with pytest.raises(ObservationError, match="empty"):
        a.normalize(RawSignal(BIOMETRIC, {}))
    with pytest.raises(ObservationError, match="must be a number"):
        a.normalize(RawSignal(BIOMETRIC, {"hrv": "high"}))
    with pytest.raises(ObservationError, match="finite"):
        a.normalize(RawSignal(BIOMETRIC, {"hrv": float("inf")}))
    with pytest.raises(ObservationError, match="mapping"):
        a.normalize(RawSignal(BIOMETRIC, 92))
    with pytest.raises(ObservationError, match="unknown biometric feature"):
        BiometricEncoder().embed(Observation(BIOMETRIC, numbers={"pulse": 1.0}))


def test_image_adapter_bytes() -> None:
    a = ImageAdapter()
    o = a.normalize(RawSignal(IMAGE, PNG))
    assert o.mime_type == "image/png"
    assert o.data == PNG
    assert "PNG" not in repr(o)
    o = a.normalize(RawSignal(IMAGE, bytearray(b"rawstuff"), mime_type="image/x-custom"))
    assert o.mime_type == "image/x-custom"
    with pytest.raises(ObservationError, match="unrecognised image"):
        a.normalize(RawSignal(IMAGE, b"not an image"))
    with pytest.raises(ObservationError, match="non-image"):
        a.normalize(RawSignal(IMAGE, PNG, mime_type="audio/wav"))
    with pytest.raises(ObservationError, match="empty bytes"):
        a.normalize(RawSignal(IMAGE, b""))
    with pytest.raises(ObservationError, match="larger than"):
        ImageAdapter(max_bytes=4).normalize(RawSignal(IMAGE, PNG))


def test_image_adapter_uris(tmp_path: Path) -> None:
    a = ImageAdapter()
    f = tmp_path / "beach photo.png"
    f.write_bytes(PNG)
    by_path = a.normalize(RawSignal(IMAGE, str(f)))
    by_uri = a.normalize(RawSignal(IMAGE, None, uri=f.as_uri()))
    assert by_path == by_uri
    assert by_path.uri == f.resolve().as_uri()
    assert by_path.data is None
    assert by_path.mime_type == "image/png"
    for uri, msg in [
        ("https://example.com/cat.jpg", "refusing to fetch https"),
        ("http://example.com/cat.jpg", "never downloads"),
        ("s3://bucket/key.png", "only file://"),
        ("file://otherhost/x.png", "remote file host"),
        (str(tmp_path / "missing.png"), "file not found"),
    ]:
        with pytest.raises(ObservationError, match=msg):
            a.normalize(RawSignal(IMAGE, None, uri=uri))
    with pytest.raises(ObservationError, match="needs bytes or a file uri"):
        a.normalize(RawSignal(IMAGE, None))
    with pytest.raises(ObservationError, match="got int"):
        a.normalize(RawSignal(IMAGE, 7))
    txt = tmp_path / "notes.txt"
    txt.write_text("x")
    with pytest.raises(ObservationError, match="non-image"):
        a.normalize(RawSignal(IMAGE, str(txt)))


def test_hash_embedder_uri_hashes_file_content(tmp_path: Path) -> None:
    f = tmp_path / "a.png"
    f.write_bytes(PNG)
    a = ImageAdapter()
    emb = HashEmbedder(IMAGE)
    assert emb.embed(a.normalize(RawSignal(IMAGE, str(f)))) == emb.embed(
        a.normalize(RawSignal(IMAGE, PNG))
    )


def _wav(samples: list[int], rate: int = 16000) -> bytes:
    data = struct.pack(f"<{len(samples)}h", *samples)
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    return header + b"data" + struct.pack("<I", len(data)) + data


def test_audio_adapter_inputs(tmp_path: Path) -> None:
    a = AudioAdapter()
    wav = _wav([0, 1000, -1000, 0])
    assert a.normalize(RawSignal(AUDIO, wav)).mime_type == "audio/wav"
    assert a.normalize(RawSignal(AUDIO, b"fLaC....")).mime_type == "audio/flac"
    pcm = a.normalize(RawSignal(AUDIO, b"\x00\x01" * 8, mime_type="audio/L16; rate=16000"))
    assert pcm.mime_type == "audio/l16"
    assert dict(pcm.numbers) == {"sample_rate": 16000.0, "channels": 1.0}
    with pytest.raises(ObservationError, match="needs a rate"):
        a.normalize(RawSignal(AUDIO, b"\x00\x01", mime_type="audio/pcm"))
    with pytest.raises(ObservationError, match="whole number"):
        a.normalize(RawSignal(AUDIO, b"\x00\x01\x02", mime_type="audio/pcm;rate=8000"))
    with pytest.raises(ObservationError, match="not an int"):
        a.normalize(RawSignal(AUDIO, b"\x00\x01", mime_type="audio/pcm;rate=fast"))
    with pytest.raises(ObservationError, match="unrecognised audio"):
        a.normalize(RawSignal(AUDIO, b"hello"))
    with pytest.raises(ObservationError, match="non-audio"):
        a.normalize(RawSignal(AUDIO, wav, mime_type="image/png"))
    with pytest.raises(ObservationError, match="refusing to fetch"):
        a.normalize(RawSignal(AUDIO, "https://example.com/a.wav"))
    f = tmp_path / "clip.wav"
    f.write_bytes(wav)
    o = a.normalize(RawSignal(AUDIO, None, uri=f.as_uri()))
    assert o.uri == f.resolve().as_uri()
    assert o.data is None
    assert o.mime_type in {"audio/wav", "audio/x-wav"}


def test_raw_pcm_decode_and_resample() -> None:
    a = AudioAdapter()
    data = struct.pack(">4h", 16384, -16384, 0, 32767)
    wave, rate = decode_audio(a.normalize(RawSignal(AUDIO, data, mime_type="audio/L16;rate=8000")))
    assert rate == 8000
    assert list(wave) == pytest.approx([0.5, -0.5, 0.0, 32767 / 32768])
    stereo = struct.pack("<4h", 16384, 0, 0, 16384)
    obs = a.normalize(RawSignal(AUDIO, stereo, mime_type="audio/pcm;rate=8000;channels=2"))
    wave, _ = decode_audio(obs)
    assert list(wave) == pytest.approx([0.25, 0.25])
    up = resample([0.0, 1.0, 0.0, -1.0], 8000, 48000)
    assert len(up) == 24
    assert up.dtype.name == "float32"
    down = resample([0.0] * 4800, 48000, 16000)
    assert len(down) == 1600


def test_smell_is_future_work() -> None:
    with pytest.raises(ChannelNotAvailableError, match="future work"):
        SmellAdapter().normalize(RawSignal(SMELL, {"voc_1": 1.0}))
    with pytest.raises(ChannelNotAvailableError, match="future work"):
        make_smell_channel()


def test_parse_image_model() -> None:
    assert parse_image_model("ViT-B-32/laion2b_s34b_b79k") == ("ViT-B-32", "laion2b_s34b_b79k")
    assert parse_image_model("ViT-B-32") == ("ViT-B-32", None)
    assert parse_image_model("hf-hub:org/repo") == ("hf-hub:org/repo", None)


def test_embedders_reject_other_channels() -> None:
    with pytest.raises(ObservationError, match="observation for 'geo'"):
        HashEmbedder(TEXT).embed(Observation(GEO, text="u09"))
    with pytest.raises(ObservationError, match="empty observation"):
        HashEmbedder(TEXT).embed(Observation(TEXT))
