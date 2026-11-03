"""Generic contract tests over every channel available offline (hash registry)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from madeleine.channels import hash_registry
from madeleine.channels.biometric import BIOMETRIC_FEATURES
from madeleine.channels.time import TimeEncoder
from madeleine.core.types import (
    AUDIO,
    BIOMETRIC,
    GEO,
    IMAGE,
    TEXT,
    TIME,
    Channel,
    ChannelRegistry,
    ChannelSpec,
    Observation,
    RawSignal,
)

REGISTRY = hash_registry()
CHANNELS = (TEXT, TIME, GEO, BIOMETRIC, IMAGE, AUDIO)

offsets = st.integers(min_value=-14 * 60, max_value=14 * 60).map(
    lambda m: timezone(timedelta(minutes=m))
)
aware_datetimes = st.datetimes(
    min_value=datetime(1900, 1, 1),  # noqa: DTZ001
    max_value=datetime(2200, 1, 1),  # noqa: DTZ001
    timezones=offsets,
)


def _signal_strategy(channel: str) -> st.SearchStrategy[RawSignal]:
    if channel == TEXT:
        return st.text(min_size=1, max_size=80).filter(str.strip).map(lambda t: RawSignal(TEXT, t))
    if channel == TIME:
        return aware_datetimes.map(lambda d: RawSignal(TIME, d))
    if channel == GEO:
        coords = st.tuples(
            st.floats(min_value=-90, max_value=90), st.floats(min_value=-180, max_value=180)
        ).map(lambda ll: RawSignal(GEO, ll))
        hashes = st.text(alphabet="0123456789bcdefghjkmnpqrstuvwxyz", min_size=1, max_size=12)
        return coords | hashes.map(lambda g: RawSignal(GEO, g))
    if channel == BIOMETRIC:
        reading = st.dictionaries(
            st.sampled_from(BIOMETRIC_FEATURES),
            st.floats(min_value=-50, max_value=500, allow_nan=False),
            min_size=1,
        )
        return reading.map(lambda r: RawSignal(BIOMETRIC, r))
    mime = "image/png" if channel == IMAGE else "audio/wav"
    return st.binary(min_size=1, max_size=256).map(lambda b: RawSignal(channel, b, mime_type=mime))


def _obs(channel: str, signal: RawSignal) -> Observation:
    return REGISTRY.get(channel).adapter.normalize(signal)


def test_registry_conforms_to_protocol() -> None:
    reg: ChannelRegistry = REGISTRY
    assert isinstance(reg, ChannelRegistry)
    assert reg.names() == tuple(sorted(CHANNELS))
    assert all(c in reg for c in CHANNELS)
    assert "smell" not in reg


@pytest.mark.parametrize("channel", CHANNELS)
@settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_channel_contract(channel: str, data: st.DataObject) -> None:
    ch = REGISTRY.get(channel)
    sa = data.draw(_signal_strategy(channel))
    sb = data.draw(_signal_strategy(channel))
    oa, ob = _obs(channel, sa), _obs(channel, sb)
    va, vb = ch.embedder.embed(oa), ch.embedder.embed(ob)
    assert ch.embedder.dim == ch.spec.dim == len(va) == len(vb)
    assert isinstance(va, tuple)
    assert all(isinstance(x, float) and math.isfinite(x) for x in va)
    assert ch.embedder.embed(oa) == va  # deterministic
    assert ch.embedder.embed_many([oa, ob]) == [va, vb]
    s = ch.similarity(va, vb)
    assert 0.0 <= s <= 1.0
    assert s == ch.similarity(vb, va)
    assert ch.similarity(va, va) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("channel", CHANNELS)
def test_spec_matches_architecture(channel: str) -> None:
    spec = REGISTRY.get(channel).spec
    expected_weight = 0.5 if channel in (TIME, BIOMETRIC) else 1.0
    assert spec.default_weight == expected_weight
    assert spec.solo_trigger is (channel not in (TIME, BIOMETRIC))
    if channel in (TEXT, IMAGE, AUDIO):
        assert spec.model_id == "hash-v1/dim=256"
        assert spec.dim == 256
    else:
        assert spec.model_id is None
    assert {TIME: 6, GEO: 3, BIOMETRIC: 12}.get(channel, 256) == spec.dim


def test_hash_dim_parameter() -> None:
    reg = hash_registry((TEXT, IMAGE), dim=64)
    assert reg.names() == (IMAGE, TEXT)
    assert reg.get(TEXT).spec.dim == 64
    assert reg.get(IMAGE).spec.model_id == "hash-v1/dim=64"


def test_channel_construction_validates() -> None:
    text = REGISTRY.get(TEXT)
    with pytest.raises(ValueError, match="dim"):
        Channel(
            spec=ChannelSpec(TEXT, dim=128),
            adapter=text.adapter,
            embedder=text.embedder,
            similarity=text.similarity,
        )
    with pytest.raises(ValueError, match="agree"):
        Channel(
            spec=ChannelSpec(TEXT, dim=6),
            adapter=text.adapter,
            embedder=TimeEncoder(),
            similarity=text.similarity,
        )


def determinism_vectors() -> dict[str, list[float]]:
    """Vectors for fixed observations on every channel (also run in a subprocess)."""
    obs = {
        TEXT: Observation(TEXT, text="rain on a tin roof"),
        IMAGE: Observation(IMAGE, data=b"\x89PNG fake", mime_type="image/png"),
        AUDIO: Observation(AUDIO, uri="file:///nonexistent/clip.wav"),
        TIME: Observation(TIME, timestamp=datetime(2026, 6, 1, 21, 30, tzinfo=UTC)),
        GEO: Observation(GEO, text="u09tunq"),
        BIOMETRIC: Observation(BIOMETRIC, numbers={"heart_rate": 92.0}),
    }
    return {k: list(REGISTRY.get(k).embedder.embed(o)) for k, o in obs.items()}


@pytest.mark.parametrize("hashseed", ["0", "12345"])
def test_embeddings_identical_across_processes(hashseed: str) -> None:
    code = (
        "import json\n"
        "from tests.channels.test_ch_contract import determinism_vectors\n"
        "print(json.dumps(determinism_vectors()))"
    )
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
        env={**os.environ, "PYTHONHASHSEED": hashseed},
    )
    assert json.loads(out.stdout) == determinism_vectors()
