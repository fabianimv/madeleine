"""Contract tests for madeleine.core.types (owned by the lead)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from madeleine.core.types import (
    GEO,
    TEXT,
    TIME,
    Cue,
    Memory,
    Observation,
    RawSignal,
    RecallConfig,
    RecallMode,
    RecallQuery,
    Recollection,
    ScoredMemory,
    SensoryContext,
    SensoryFingerprint,
    new_id,
)

T0 = datetime(2026, 6, 1, 21, 30, tzinfo=UTC)


def _memory() -> Memory:
    ctx = SensoryContext(timestamp=T0, geohash="U09TUNQ", tags=frozenset({"paris"}), salience=0.9)
    fp = SensoryFingerprint(
        embeddings={TEXT: (1.0, 0.0), GEO: (0.0, 1.0)},
        context=ctx,
        observations={
            TEXT: Observation(TEXT, text="dinner"),
            GEO: Observation(GEO, text="u09tunq"),
        },
    )
    return Memory(id=new_id("mem"), content="Dinner by the tower", fingerprint=fp, created_at=T0)


def test_context_normalises_and_validates() -> None:
    ctx = SensoryContext(timestamp=T0, geohash="U09TUNQ")
    assert ctx.geohash == "u09tunq"
    with pytest.raises(ValueError, match="timezone-aware"):
        SensoryContext(timestamp=datetime(2026, 1, 1))  # noqa: DTZ001
    with pytest.raises(ValueError, match="salience"):
        SensoryContext(timestamp=T0, salience=1.5)
    with pytest.raises(ValueError, match="geohash"):
        SensoryContext(timestamp=T0, geohash="aaaa")  # 'a' is not in the geohash alphabet


def test_channel_names_are_validated() -> None:
    with pytest.raises(ValueError, match="channel name"):
        RawSignal("Text!", "x")
    with pytest.raises(ValueError, match="stored under key"):
        Cue(id="c", observations={GEO: Observation(TEXT, text="x")}, context=SensoryContext(T0))


def test_observation_strips_raw_and_serialises() -> None:
    obs = Observation("image", data=b"\x89PNG", mime_type="image/png")
    assert obs.without_raw().data is None
    assert "data_b64" not in obs.to_dict()
    assert obs.to_dict(include_raw=True)["data_b64"] == "iVBORw=="
    assert "PNG" not in repr(obs)


def test_fingerprint_rejects_observations_without_embeddings() -> None:
    with pytest.raises(ValueError, match="without embeddings"):
        SensoryFingerprint(
            embeddings={}, context=SensoryContext(T0), observations={TEXT: Observation(TEXT)}
        )


def test_recollection_round_trips_to_json() -> None:
    mem = _memory()
    match = ScoredMemory(memory=mem, score=0.8123456789, channel_scores={GEO: 1.0, TIME: 0.9})
    rec = Recollection(
        id=new_id("rec"),
        match=match,
        cue_id="cue_1",
        cue_channels=frozenset({GEO, TIME}),
        triggered_at=T0 + timedelta(days=1),
    )
    payload = json.loads(json.dumps(rec.to_dict()))
    assert payload["mode"] == RecallMode.PUSH.value
    assert payload["memory"]["content"] == "Dinner by the tower"
    assert payload["cue_channels"] == ["geo", "time"]
    assert payload["score"] == 0.812346
    assert rec.to_prompt_line().startswith("[madeleine 0.81 via geo+time] Dinner by the tower")


def test_recall_config_validation() -> None:
    RecallConfig()
    with pytest.raises(ValueError, match="trigger_threshold"):
        RecallConfig(trigger_threshold=1.2)
    with pytest.raises(ValueError, match="half_life"):
        RecallConfig(half_life=timedelta(0))
    with pytest.raises(ValueError, match="weight"):
        RecallConfig(channel_weights={TEXT: -1.0})
    with pytest.raises(ValueError, match="k must"):
        RecallQuery(cue=Cue(id="c", observations={}, context=SensoryContext(T0)), k=0)


def test_new_id_is_prefixed_and_unique() -> None:
    ids = {new_id("mem") for _ in range(1000)}
    assert len(ids) == 1000
    assert all(i.startswith("mem_") for i in ids)


def test_package_import_is_light() -> None:
    import subprocess
    import sys

    code = (
        "import sys, madeleine, madeleine.core.types; "
        "heavy = {'torch', 'sentence_transformers', 'open_clip', 'transformers', "
        "'fastapi', 'mcp'}; "
        "print(sorted(heavy & set(sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
