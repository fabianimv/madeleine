"""Small deterministic fake channels for engine tests (no models, no network)."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime

from madeleine.core import geohash as gh
from madeleine.core.types import (
    GEO,
    IMAGE,
    TEXT,
    TIME,
    Channel,
    ChannelName,
    ChannelNotAvailableError,
    ChannelSpec,
    Observation,
    ObservationError,
    RawSignal,
    Vector,
)

TEXT_DIM = 32


def _unit(values: Sequence[float]) -> Vector:
    n = math.sqrt(sum(v * v for v in values))
    if n == 0:
        return tuple(0.0 for _ in values)
    return tuple(v / n for v in values)


def _cos01(a: Vector, b: Vector) -> float:
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b, strict=True))))


def _tokens(text: str) -> list[str]:
    return "".join(c.lower() if c.isalnum() else " " for c in text).split()


class _Embedder:
    def __init__(self, channel: ChannelName, dim: int, model_id: str | None, fn: object) -> None:
        self._channel = channel
        self._dim = dim
        self._model_id = model_id
        self._fn = fn
        self.calls = 0

    @property
    def channel(self) -> ChannelName:
        return self._channel

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str | None:
        return self._model_id

    def embed(self, observation: Observation) -> Vector:
        self.calls += 1
        out: Vector = self._fn(observation)  # type: ignore[operator]
        return out

    def embed_many(self, observations: Sequence[Observation]) -> list[Vector]:
        return [self.embed(o) for o in observations]


class _Adapter:
    def __init__(self, channel: ChannelName, fn: object) -> None:
        self._channel = channel
        self._fn = fn

    @property
    def channel(self) -> ChannelName:
        return self._channel

    def normalize(self, signal: RawSignal) -> Observation:
        out: Observation = self._fn(signal)  # type: ignore[operator]
        return out


# -- text ------------------------------------------------------------------------------
def _text_norm(signal: RawSignal) -> Observation:
    if not isinstance(signal.value, str) or not signal.value.strip():
        raise ObservationError("text needs a non-empty string")
    return Observation(TEXT, text=signal.value.strip())


def _text_embed(obs: Observation) -> Vector:
    acc = [0.0] * TEXT_DIM
    for tok in _tokens(obs.text or ""):
        h = hashlib.blake2b(tok.encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % TEXT_DIM
        acc[idx] += 1.0 if h[4] & 1 else -1.0
    return _unit(acc)


def text_channel(model_id: str | None = "fake-text-v1", dim: int = TEXT_DIM) -> Channel:
    spec = ChannelSpec(TEXT, dim=dim, default_weight=1.0, solo_trigger=True, model_id=model_id)
    return Channel(
        spec, _Adapter(TEXT, _text_norm), _Embedder(TEXT, dim, model_id, _text_embed), _cos01
    )


# -- geo -------------------------------------------------------------------------------
def _geo_norm(signal: RawSignal) -> Observation:
    v = signal.value
    if isinstance(v, str) and gh.is_valid(v):
        return Observation(GEO, text=v.lower())
    if isinstance(v, tuple) and len(v) == 2:
        lat, lon = float(v[0]), float(v[1])
        return Observation(GEO, text=gh.encode(lat, lon), numbers={"lat": lat, "lon": lon})
    raise ObservationError("geo needs a geohash or (lat, lon)")


def _geo_embed(obs: Observation) -> Vector:
    lat, lon = gh.decode(obs.text or "")
    la, lo = math.radians(lat), math.radians(lon)
    return (math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la))


def _geo_sim(a: Vector, b: Vector) -> float:
    chord = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))
    angle = 2 * math.asin(min(1.0, chord / 2))
    return math.exp(-(gh.EARTH_RADIUS_KM * angle) / 1.0)


def geo_channel() -> Channel:
    spec = ChannelSpec(GEO, dim=3, default_weight=1.0, solo_trigger=True, metric="exp(-km)")
    return Channel(spec, _Adapter(GEO, _geo_norm), _Embedder(GEO, 3, None, _geo_embed), _geo_sim)


# -- time ------------------------------------------------------------------------------
def _time_norm(signal: RawSignal) -> Observation:
    v = signal.value
    if v is None:
        v = signal.observed_at or datetime.now(UTC)
    if not isinstance(v, datetime):
        raise ObservationError("time needs a datetime")
    return Observation(TIME, timestamp=v if v.tzinfo else v.replace(tzinfo=UTC))


def _time_embed(obs: Observation) -> Vector:
    ts = obs.timestamp
    assert ts is not None
    angle = 2 * math.pi * (ts.hour + ts.minute / 60) / 24
    return (math.sin(angle), math.cos(angle))


def _time_sim(a: Vector, b: Vector) -> float:
    return max(0.0, min(1.0, (1 + sum(x * y for x, y in zip(a, b, strict=True))) / 2))


def time_channel() -> Channel:
    spec = ChannelSpec(TIME, dim=2, default_weight=0.5, solo_trigger=False, metric="(1+cos)/2")
    return Channel(
        spec, _Adapter(TIME, _time_norm), _Embedder(TIME, 2, None, _time_embed), _time_sim
    )


# -- image (raw bytes) -----------------------------------------------------------------
def _image_norm(signal: RawSignal) -> Observation:
    if not isinstance(signal.value, bytes):
        raise ObservationError("image needs bytes")
    return Observation(IMAGE, data=signal.value, mime_type=signal.mime_type or "image/png")


def _image_embed(obs: Observation) -> Vector:
    digest = hashlib.blake2b(obs.data or b"", digest_size=16).digest()
    return _unit([b - 127.5 for b in digest])


def image_channel() -> Channel:
    spec = ChannelSpec(IMAGE, dim=16, default_weight=1.0, model_id="fake-image-v1")
    return Channel(
        spec,
        _Adapter(IMAGE, _image_norm),
        _Embedder(IMAGE, 16, "fake-image-v1", _image_embed),
        _cos01,
    )


class DictRegistry:
    """Minimal :class:`~madeleine.core.types.ChannelRegistry`."""

    def __init__(self, channels: Iterable[Channel] = ()) -> None:
        self._channels: dict[ChannelName, Channel] = {}
        for ch in channels:
            self.register(ch)

    def register(self, channel: Channel, *, replace: bool = False) -> None:
        if channel.name in self._channels and not replace:
            raise ValueError(f"channel {channel.name!r} already registered")
        self._channels[channel.name] = channel

    def get(self, name: ChannelName) -> Channel:
        try:
            return self._channels[name]
        except KeyError:
            raise ChannelNotAvailableError(f"unknown channel {name!r}") from None

    def names(self) -> tuple[ChannelName, ...]:
        return tuple(sorted(self._channels))

    def __contains__(self, name: object) -> bool:
        return name in self._channels


def fake_registry(*names: ChannelName) -> DictRegistry:
    makers = {TEXT: text_channel, GEO: geo_channel, TIME: time_channel, IMAGE: image_channel}
    return DictRegistry(makers[n]() for n in (names or (TEXT, GEO, TIME, IMAGE)))
