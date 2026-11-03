"""JSON (de)serialisation of contract types for persistence.

The dict formats are the contract's ``to_dict`` output **plus** the fields needed to invert
it exactly:

* :func:`memory_to_dict` = ``Memory.to_dict()`` + ``"fingerprint": {"embeddings": {channel:
  base64(float32 little-endian)}, "observations": {channel: Observation.to_dict()}}``.
* :func:`recollection_to_dict` = ``Recollection.to_dict()`` with full-precision floats
  (``to_dict`` rounds to 6 digits) and the full memory dict above.

Vectors are persisted as float32 (contract rule), so a round trip is exact for vectors whose
components are float32-representable (the engine produces only such vectors).
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from datetime import datetime
from typing import Any

import numpy as np

from madeleine.core.types import (
    ChannelName,
    JSONValue,
    Memory,
    Observation,
    RecallMode,
    Recollection,
    ScoredMemory,
    SensoryContext,
    SensoryFingerprint,
    Vector,
)

__all__ = [
    "CODEC_VERSION",
    "context_from_dict",
    "memory_from_dict",
    "memory_to_dict",
    "observation_from_dict",
    "recollection_from_dict",
    "recollection_to_dict",
    "vector_from_bytes",
    "vector_to_bytes",
]

CODEC_VERSION = 1


def vector_to_bytes(vector: Vector) -> bytes:
    """Little-endian float32 bytes (the on-disk / sqlite-vec representation)."""
    return np.asarray(vector, dtype="<f4").tobytes()


def vector_from_bytes(data: bytes) -> Vector:
    return tuple(float(x) for x in np.frombuffer(data, dtype="<f4").tolist())


def _vec_b64(vector: Vector) -> str:
    return base64.b64encode(vector_to_bytes(vector)).decode("ascii")


def _vec_from_b64(text: str) -> Vector:
    try:
        return vector_from_bytes(base64.b64decode(text, validate=True))
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid base64 vector") from exc


def _dt(value: Any, what: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{what} must be an ISO-8601 string")
    return datetime.fromisoformat(value)


def _mapping(value: Any, what: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{what} must be an object")
    return value


def observation_from_dict(d: Mapping[str, Any]) -> Observation:
    """Invert ``Observation.to_dict`` (``data_b64`` decoded when present)."""
    data_b64 = d.get("data_b64")
    ts = d.get("timestamp")
    numbers = _mapping(d.get("numbers"), "numbers")
    return Observation(
        channel=str(d["channel"]),
        text=d.get("text"),
        data=base64.b64decode(data_b64) if isinstance(data_b64, str) else None,
        mime_type=d.get("mime_type"),
        uri=d.get("uri"),
        numbers={str(k): float(v) for k, v in numbers.items()},
        timestamp=_dt(ts, "timestamp") if ts is not None else None,
    )


def context_from_dict(d: Mapping[str, Any]) -> SensoryContext:
    """Invert ``SensoryContext.to_dict``."""
    return SensoryContext(
        timestamp=_dt(d["timestamp"], "context.timestamp"),
        geohash=d.get("geohash"),
        tags=frozenset(str(t) for t in d.get("tags") or ()),
        salience=float(d.get("salience", 0.5)),
    )


def memory_to_dict(memory: Memory, *, include_raw: bool = False) -> dict[str, JSONValue]:
    """``Memory.to_dict()`` plus the fingerprint (vectors, observations)."""
    out = memory.to_dict()
    fp = memory.fingerprint
    out["fingerprint"] = {
        "embeddings": {c: _vec_b64(v) for c, v in sorted(fp.embeddings.items())},
        "observations": {
            c: o.to_dict(include_raw=include_raw) for c, o in sorted(fp.observations.items())
        },
    }
    return out


def memory_from_dict(d: Mapping[str, Any]) -> Memory:
    """Invert :func:`memory_to_dict`. Raises ``ValueError`` if the fingerprint is missing."""
    fp_raw = d.get("fingerprint")
    if not isinstance(fp_raw, Mapping):
        raise ValueError("memory dict has no 'fingerprint' (not produced by memory_to_dict)")
    embeddings: dict[ChannelName, Vector] = {
        str(c): _vec_from_b64(v) for c, v in _mapping(fp_raw.get("embeddings"), "emb").items()
    }
    observations = {
        str(c): observation_from_dict(o)
        for c, o in _mapping(fp_raw.get("observations"), "observations").items()
    }
    fingerprint = SensoryFingerprint(
        embeddings=embeddings,
        context=context_from_dict(_mapping(d.get("context"), "context")),
        observations=observations,
    )
    return Memory(
        id=str(d["id"]),
        content=str(d["content"]),
        fingerprint=fingerprint,
        created_at=_dt(d["created_at"], "created_at"),
        media_ref=d.get("media_ref"),
        metadata={str(k): str(v) for k, v in _mapping(d.get("metadata"), "metadata").items()},
    )


def recollection_to_dict(rec: Recollection) -> dict[str, JSONValue]:
    """``Recollection.to_dict()`` with exact floats and the full memory (no raw bytes)."""
    out = rec.to_dict()
    m = rec.match
    out["v"] = CODEC_VERSION
    out["memory"] = memory_to_dict(m.memory, include_raw=False)
    out["score"] = m.score
    out["channel_scores"] = {k: float(v) for k, v in sorted(m.channel_scores.items())}
    out["salience_factor"] = m.salience_factor
    out["decay_factor"] = m.decay_factor
    return out


def recollection_from_dict(d: Mapping[str, Any]) -> Recollection:
    """Invert :func:`recollection_to_dict`."""
    match = ScoredMemory(
        memory=memory_from_dict(_mapping(d.get("memory"), "memory")),
        score=float(d["score"]),
        channel_scores={
            str(k): float(v) for k, v in _mapping(d.get("channel_scores"), "cs").items()
        },
        salience_factor=float(d.get("salience_factor", 1.0)),
        decay_factor=float(d.get("decay_factor", 1.0)),
    )
    return Recollection(
        id=str(d["id"]),
        match=match,
        cue_id=str(d["cue_id"]),
        cue_channels=frozenset(str(c) for c in d.get("cue_channels") or ()),
        triggered_at=_dt(d["triggered_at"], "triggered_at"),
        mode=RecallMode(d.get("mode", RecallMode.PUSH.value)),
        cue_source=str(d.get("cue_source", "unknown")),
    )
