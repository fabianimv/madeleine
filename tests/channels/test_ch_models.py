"""Real-model tests (download weights; run with ``uv run pytest -m models``)."""

from __future__ import annotations

import io
import math
import struct

import pytest

from madeleine.channels.audio import make_audio_channel
from madeleine.channels.image import make_image_channel
from madeleine.channels.similarity import cosine01
from madeleine.channels.text import make_text_channel
from madeleine.core.types import AUDIO, IMAGE, TEXT, MadeleineConfig, RawSignal

pytestmark = pytest.mark.models


def test_sentence_transformers_text() -> None:
    pytest.importorskip("sentence_transformers")
    ch = make_text_channel(MadeleineConfig(text_backend="sentence-transformers"))

    def v(text: str) -> tuple[float, ...]:
        return ch.embedder.embed(ch.adapter.normalize(RawSignal(TEXT, text)))

    rain = v("rain drumming on a tin roof")
    storm = v("a thunderstorm hammering the metal roof of the shed")
    stocks = v("quarterly earnings report of the stock market")
    assert len(rain) == ch.spec.dim == 384
    assert math.fsum(x * x for x in rain) == pytest.approx(1.0, abs=1e-4)
    assert ch.similarity(rain, storm) > ch.similarity(rain, stocks) + 0.1
    assert v("rain drumming on a tin roof") == rain
    obs = [ch.adapter.normalize(RawSignal(TEXT, t)) for t in ("a", "b c")]
    assert ch.embedder.embed_many(obs) == [ch.embedder.embed(o) for o in obs]


def _png(color: tuple[int, int, int], stripes: bool = False) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (64, 64), color)
    if stripes:
        for x in range(0, 64, 8):
            for y in range(64):
                img.putpixel((x, y), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_open_clip_image() -> None:
    pytest.importorskip("open_clip")
    ch = make_image_channel(MadeleineConfig())
    red = ch.embedder.embed(ch.adapter.normalize(RawSignal(IMAGE, _png((220, 20, 20)))))
    red2 = ch.embedder.embed(ch.adapter.normalize(RawSignal(IMAGE, _png((200, 30, 30)))))
    blue = ch.embedder.embed(ch.adapter.normalize(RawSignal(IMAGE, _png((20, 20, 220), True))))
    assert len(red) == ch.spec.dim == 512
    assert ch.similarity(red, red) == pytest.approx(1.0, abs=1e-6)
    assert cosine01(red, red2) > cosine01(red, blue)


def _tone_wav(freq: float, seconds: float = 2.0, rate: int = 16000) -> bytes:
    n = int(seconds * rate)
    samples = [int(12000 * math.sin(2 * math.pi * freq * i / rate)) for i in range(n)]
    data = struct.pack(f"<{n}h", *samples)
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    return header + b"data" + struct.pack("<I", len(data)) + data


def test_clap_audio() -> None:
    pytest.importorskip("transformers")
    pytest.importorskip("soundfile")
    ch = make_audio_channel(MadeleineConfig())
    low = ch.embedder.embed(ch.adapter.normalize(RawSignal(AUDIO, _tone_wav(220))))
    low2 = ch.embedder.embed(ch.adapter.normalize(RawSignal(AUDIO, _tone_wav(233))))
    assert len(low) == ch.spec.dim == 512
    assert ch.embedder.embed(ch.adapter.normalize(RawSignal(AUDIO, _tone_wav(220)))) == low
    assert ch.similarity(low, low) == pytest.approx(1.0, abs=1e-6)
    assert cosine01(low, low2) > 0.5
