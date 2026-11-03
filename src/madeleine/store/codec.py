"""JSON (de)serialisation of contract types for persistence.

The dict formats are the contract's ``to_dict`` output **plus** the fields needed to invert
it exactly:

* :func:`memory_to_dict` = ``Memory.to_dict()`` + ``"fingerprint": {"embeddings": {channel:
  base64(float32 little-endian)}, "observations": {channel: Observation.to_dict()}}``.
* :func:`recollection_to_dict` = ``Recollection.to_dict()`` with full-precision floats
  (``to_dict`` rounds to 6 digits) and the full memory dict above. This was the queue
  payload of codec version 1; it is kept for exports and to read (and migrate) old rows.
* :func:`queued_to_dict` = the queue payload since codec version 2 (R20): the memory **id**,
  the score explanation and the recollection fields; no content, embeddings or
  observations. :func:`queued_from_dict` reads both versions; the queue rehydrates the
  memory from the store (a forgotten memory's recollection is dropped).

Vectors are persisted as float32 (contract rule), so a round trip is exact for vectors whose
components are float32-representable (the engine produces only such vectors).
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass
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
    "QueuedRecollection",
    "context_from_dict",
    "memory_from_dict",
    "memory_to_dict",
    "observation_from_dict",
    "queued_from_dict",
    "queued_to_dict",
    "recollection_from_dict",
    "recollection_to_dict",
    "vector_from_bytes",
    "vector_to_bytes",
]

CODEC_VERSION = 2


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
    out["v"] = 1
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


@dataclass(frozen=True, slots=True)
class QueuedRecollection:
    """A queue entry without its memory: what the queue table stores since codec v2."""

    id: str
    memory_id: str
    score: float
    channel_scores: Mapping[ChannelName, float]
    salience_factor: float
    decay_factor: float
    cue_id: str
    cue_channels: frozenset[ChannelName]
    triggered_at: datetime
    mode: RecallMode
    cue_source: str

    @classmethod
    def from_recollection(cls, rec: Recollection) -> QueuedRecollection:
        m = rec.match
        return cls(
            id=rec.id,
            memory_id=m.memory.id,
            score=m.score,
            channel_scores=dict(m.channel_scores),
            salience_factor=m.salience_factor,
            decay_factor=m.decay_factor,
            cue_id=rec.cue_id,
            cue_channels=rec.cue_channels,
            triggered_at=rec.triggered_at,
            mode=rec.mode,
            cue_source=rec.cue_source,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        """The codec v2 payload (exact floats; no memory content)."""
        cue_channels: list[JSONValue] = [*sorted(self.cue_channels)]
        return {
            "v": CODEC_VERSION,
            "id": self.id,
            "memory_id": self.memory_id,
            "score": self.score,
            "channel_scores": {k: float(v) for k, v in sorted(self.channel_scores.items())},
            "salience_factor": self.salience_factor,
            "decay_factor": self.decay_factor,
            "cue_id": self.cue_id,
            "cue_channels": cue_channels,
            "triggered_at": self.triggered_at.isoformat(),
            "mode": self.mode.value,
            "cue_source": self.cue_source,
        }

    def with_memory(self, memory: Memory) -> Recollection:
        """Rehydrate with the memory fetched from the store."""
        if memory.id != self.memory_id:
            raise ValueError(f"memory {memory.id!r} is not {self.memory_id!r}")
        return Recollection(
            id=self.id,
            match=ScoredMemory(
                memory=memory,
                score=self.score,
                channel_scores=self.channel_scores,
                salience_factor=self.salience_factor,
                decay_factor=self.decay_factor,
            ),
            cue_id=self.cue_id,
            cue_channels=self.cue_channels,
            triggered_at=self.triggered_at,
            mode=self.mode,
            cue_source=self.cue_source,
        )


def queued_to_dict(rec: Recollection) -> dict[str, JSONValue]:
    """Slim queue payload (codec v2): memory id + score explanation + recollection fields."""
    return QueuedRecollection.from_recollection(rec).to_dict()


def queued_from_dict(d: Mapping[str, Any]) -> QueuedRecollection:
    """Read a queue payload of codec v2 (:func:`queued_to_dict`) or v1
    (:func:`recollection_to_dict`, whose embedded memory is ignored: the store is the
    source of truth). Raises ``ValueError``/``KeyError``/``TypeError`` if malformed."""
    memory_id = d.get("memory_id")
    if memory_id is None:
        memory_id = _mapping(d.get("memory"), "memory")["id"]
    triggered_at = _dt(d["triggered_at"], "triggered_at")
    if triggered_at.tzinfo is None:
        raise ValueError("triggered_at must be timezone-aware")
    return QueuedRecollection(
        id=str(d["id"]),
        memory_id=str(memory_id),
        score=float(d["score"]),
        channel_scores={
            str(k): float(v) for k, v in _mapping(d.get("channel_scores"), "cs").items()
        },
        salience_factor=float(d.get("salience_factor", 1.0)),
        decay_factor=float(d.get("decay_factor", 1.0)),
        cue_id=str(d["cue_id"]),
        cue_channels=frozenset(str(c) for c in d.get("cue_channels") or ()),
        triggered_at=triggered_at,
        mode=RecallMode(d.get("mode", RecallMode.PUSH.value)),
        cue_source=str(d.get("cue_source", "unknown")),
    )
