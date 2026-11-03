"""The madeleine contract: every protocol and dataclass the rest of the codebase builds on.

This module is the single source of truth shared by all workstreams (engine, channels,
MCP server, ingestion, evals). It contains **no behaviour beyond validation and
serialisation helpers** and imports nothing heavier than the standard library.

Changing anything here is a contract change: it requires an ADR under ``docs/adr/``
(see ADR 0001) and a sign-off from the lead.

Vocabulary (see ``docs/concepts.md`` and ADR 0001 for the reasoning):

* **Observation** - one normalised reading on one channel ("the text 'rain on tin roof'",
  "geohash u4pruyd", "13:05 on a Tuesday in spring").
* **Channel** - a pluggable sense: an :class:`Adapter` (raw signal -> observation), an
  :class:`Embedder` (observation -> vector) and a :data:`SimilarityFn` (vector x vector ->
  similarity in ``[0, 1]``), described by a :class:`ChannelSpec`.
* **Memory** - content plus the :class:`SensoryFingerprint` captured when it was encoded.
* **Cue** - an incoming bundle of observations (one or more channels).
* **Recall** - fusing per-channel similarities between a cue and memories
  (:class:`RecallConfig` documents the formula).
* **Recollection** - a memory surfaced by a cue. *Pull* recollections are returned to a
  caller who asked; *push* (involuntary) recollections are queued in a
  :class:`RecollectionQueue` for the agent to receive without asking.
"""

from __future__ import annotations

import base64
import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final, Literal, Protocol, TypeAlias, runtime_checkable

__all__ = [
    "AUDIO",
    "BIOMETRIC",
    "GEO",
    "IMAGE",
    "SMELL",
    "TEXT",
    "TIME",
    "Adapter",
    "Channel",
    "ChannelName",
    "ChannelNotAvailableError",
    "ChannelRegistry",
    "ChannelSpec",
    "Clock",
    "Cue",
    "CueAdapter",
    "Embedder",
    "JSONValue",
    "MadeleineConfig",
    "MadeleineError",
    "Memory",
    "MemoryEngine",
    "MemoryNotFoundError",
    "MemoryStore",
    "Observation",
    "ObservationError",
    "RawSignal",
    "RecallConfig",
    "RecallMode",
    "RecallQuery",
    "Recollection",
    "RecollectionListener",
    "RecollectionQueue",
    "ScoredMemory",
    "SensoryContext",
    "SensoryFingerprint",
    "SimilarityFn",
    "StoreError",
    "SystemClock",
    "TextBackend",
    "Unsubscribe",
    "Vector",
    "new_id",
    "utc_now",
]

# --------------------------------------------------------------------------------------
# Primitive aliases
# --------------------------------------------------------------------------------------

ChannelName: TypeAlias = str
"""Name of a channel, e.g. ``"text"``. Lower-case ``[a-z][a-z0-9_]*``; see :data:`TEXT` etc."""

Vector: TypeAlias = tuple[float, ...]
"""An embedding. Immutable, JSON-friendly, and exactly ``ChannelSpec.dim`` long.

Embedders may compute with numpy internally but must return a plain tuple of Python
floats (``tuple(arr.astype(float).tolist())``). Stores persist vectors as float32.
"""

JSONValue: TypeAlias = str | int | float | bool | list["JSONValue"] | dict[str, "JSONValue"] | None
"""Any JSON-serialisable value (used by ``to_dict`` helpers and webhook payloads)."""

# Well-known channel names. Third parties may register more (e.g. "smell" in the future).
TEXT: Final[ChannelName] = "text"
"""Natural-language text. Observation field: ``text``."""
IMAGE: Final[ChannelName] = "image"
"""Still images. Observation fields: ``data`` (bytes) and/or ``uri``; ``mime_type``."""
AUDIO: Final[ChannelName] = "audio"
"""Sound clips. Observation fields: ``data`` (bytes) and/or ``uri``; ``mime_type``;
``numbers["sample_rate"]`` when ``data`` is raw PCM."""
TIME: Final[ChannelName] = "time"
"""Time of day / week / year. Observation field: ``timestamp`` (tz-aware)."""
GEO: Final[ChannelName] = "geo"
"""Place. Observation field: ``text`` = geohash (lower-case base32); optionally
``numbers["lat"]``, ``numbers["lon"]``."""
BIOMETRIC: Final[ChannelName] = "biometric"
"""Bodily state. Observation field: ``numbers`` with any of ``heart_rate`` (bpm),
``hrv`` (ms), ``skin_temp`` (deg C), ``eda`` (microsiemens), ``steps_per_min``,
``respiration_rate`` (breaths/min)."""
SMELL: Final[ChannelName] = "smell"
"""Reserved for a future olfactory channel (out of scope for v0; adapter stub only)."""

_CHANNEL_NAME_CHARS: Final = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")


def _check_channel_name(name: str) -> None:
    if not name or not name[0].isalpha() or not set(name) <= _CHANNEL_NAME_CHARS:
        raise ValueError(f"invalid channel name {name!r}: expected [a-z][a-z0-9_]*")


def utc_now() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(UTC)


def _require_aware(dt: datetime, what: str) -> None:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{what} must be timezone-aware (got naive datetime {dt!r})")


def new_id(prefix: str) -> str:
    """Return a new, roughly time-sortable identifier such as ``"mem_0189a3f2c1d4_9f3c2a7b1e"``.

    Format: ``{prefix}_{12 hex digits of ms since epoch}_{10 random hex digits}``.
    Prefixes in use: ``mem`` (Memory), ``cue`` (Cue), ``rec`` (Recollection).
    """
    millis = time.time_ns() // 1_000_000
    return f"{prefix}_{millis:012x}_{os.urandom(5).hex()}"


def _json_strs(items: Iterable[str]) -> list[JSONValue]:
    out: list[JSONValue] = []
    out.extend(sorted(items))
    return out


def _freeze_mapping(value: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(dict(value))


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class MadeleineError(Exception):
    """Base class for every error raised deliberately by madeleine."""


class ObservationError(MadeleineError, ValueError):
    """A raw signal or observation is malformed or unsupported by its channel."""


class ChannelNotAvailableError(MadeleineError):
    """A channel is unknown, disabled, or needs an optional extra that is not installed.

    The message must tell the user how to fix it, e.g.
    ``"channel 'image' needs: pip install 'madeleine-mcp[image]'"``.
    """


class MemoryNotFoundError(MadeleineError, KeyError):
    """No memory with the given id exists."""


class StoreError(MadeleineError):
    """The storage backend failed (I/O, schema mismatch, missing sqlite extension...)."""


# --------------------------------------------------------------------------------------
# Signals, observations and context
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawSignal:
    """A raw, un-normalised reading on one channel, as it arrives from the outside world.

    Examples::

        RawSignal("text", "smell of rain on hot asphalt")
        RawSignal("geo", (48.8584, 2.2945))           # (lat, lon)
        RawSignal("geo", "u09tunq")                    # geohash
        RawSignal("image", b"\\x89PNG...", mime_type="image/png")
        RawSignal("image", None, uri="file:///photos/beach.jpg")
        RawSignal("biometric", {"heart_rate": 92.0})
        RawSignal("time", datetime(2026, 7, 1, 21, 30, tzinfo=UTC))

    An :class:`Adapter` turns it into an :class:`Observation`.
    """

    channel: ChannelName
    value: object
    mime_type: str | None = None
    uri: str | None = None
    observed_at: datetime | None = None
    """When the signal was sensed. ``None`` means "now" (the engine's clock)."""

    def __post_init__(self) -> None:
        _check_channel_name(self.channel)
        if self.observed_at is not None:
            _require_aware(self.observed_at, "RawSignal.observed_at")


@dataclass(frozen=True, slots=True)
class Observation:
    """A normalised reading on one channel: the input of an :class:`Embedder`.

    One generic shape for every channel keeps channels pluggable. Each channel documents
    which fields it reads (see the channel-name constants above). Fields a channel does
    not use must be left at their defaults.

    Privacy: ``data`` holds raw media bytes. It is used for embedding and then dropped;
    it is persisted only when the caller explicitly passes ``store_raw=True``.
    """

    channel: ChannelName
    text: str | None = None
    data: bytes | None = field(default=None, repr=False)
    mime_type: str | None = None
    uri: str | None = None
    numbers: Mapping[str, float] = field(default_factory=dict)
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        _check_channel_name(self.channel)
        if self.timestamp is not None:
            _require_aware(self.timestamp, "Observation.timestamp")
        object.__setattr__(self, "numbers", _freeze_mapping(self.numbers))

    def without_raw(self) -> Observation:
        """Return a copy with ``data`` removed (what gets persisted by default)."""
        return Observation(
            channel=self.channel,
            text=self.text,
            data=None,
            mime_type=self.mime_type,
            uri=self.uri,
            numbers=self.numbers,
            timestamp=self.timestamp,
        )

    def to_dict(self, *, include_raw: bool = False) -> dict[str, JSONValue]:
        """JSON-friendly representation. ``data`` is base64-encoded and only if asked."""
        out: dict[str, JSONValue] = {"channel": self.channel}
        if self.text is not None:
            out["text"] = self.text
        if include_raw and self.data is not None:
            out["data_b64"] = base64.b64encode(self.data).decode("ascii")
        if self.mime_type is not None:
            out["mime_type"] = self.mime_type
        if self.uri is not None:
            out["uri"] = self.uri
        if self.numbers:
            out["numbers"] = {k: float(v) for k, v in self.numbers.items()}
        if self.timestamp is not None:
            out["timestamp"] = self.timestamp.isoformat()
        return out


@dataclass(frozen=True, slots=True)
class SensoryContext:
    """Structured context captured alongside a memory (or a cue).

    * ``timestamp`` - when the experience happened (memory) or the cue was sensed (cue).
      Timezone-aware. Memory age for decay is measured from this value.
    * ``geohash`` - where, as a lower-case geohash (any precision 1..12), or ``None``.
    * ``tags`` - free-form labels ("work", "family"); used for filtering, not scoring.
    * ``salience`` - how emotionally / practically significant, in ``[0, 1]``.
      Default 0.5 = neutral. Boosts recall score (see :class:`RecallConfig`).
      On a cue it is the cue's intensity; v0 records it but does not use it for scoring.
    """

    timestamp: datetime
    geohash: str | None = None
    tags: frozenset[str] = frozenset()
    salience: float = 0.5

    def __post_init__(self) -> None:
        _require_aware(self.timestamp, "SensoryContext.timestamp")
        if not 0.0 <= self.salience <= 1.0:
            raise ValueError(f"salience must be in [0, 1], got {self.salience}")
        if self.geohash is not None:
            gh = self.geohash.lower()
            if not (1 <= len(gh) <= 12) or not set(gh) <= set("0123456789bcdefghjkmnpqrstuvwxyz"):
                raise ValueError(f"invalid geohash {self.geohash!r}")
            object.__setattr__(self, "geohash", gh)
        object.__setattr__(self, "tags", frozenset(self.tags))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "geohash": self.geohash,
            "tags": _json_strs(self.tags),
            "salience": self.salience,
        }


@dataclass(frozen=True, slots=True)
class SensoryFingerprint:
    """Everything sensed at encoding time: per-channel embeddings plus structured context.

    ``embeddings`` maps channel name -> vector. Only channels that were actually observed
    are present; recall fuses over channels present in *both* cue and memory.
    ``observations`` keeps the normalised observations (``data`` stripped unless the
    memory was stored with ``store_raw=True``) so fingerprints can be re-embedded when a
    model changes and explained to users.
    """

    embeddings: Mapping[ChannelName, Vector]
    context: SensoryContext
    observations: Mapping[ChannelName, Observation] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "embeddings", _freeze_mapping(self.embeddings))
        object.__setattr__(self, "observations", _freeze_mapping(self.observations))
        stray = set(self.observations) - set(self.embeddings)
        if stray:
            raise ValueError(f"observations without embeddings for channels: {sorted(stray)}")

    @property
    def channels(self) -> frozenset[ChannelName]:
        return frozenset(self.embeddings)


@dataclass(frozen=True, slots=True)
class Memory:
    """A remembered item: content plus the sensory fingerprint captured when encoded.

    * ``content`` - what the agent should get back (free text, usually a sentence).
    * ``media_ref`` - optional pointer to external media (URI/path). madeleine never
      dereferences it on its own.
    * ``created_at`` - when the record was stored (bookkeeping; decay uses
      ``fingerprint.context.timestamp``).
    * ``metadata`` - free-form string key/values for the caller (not scored).
    """

    id: str
    content: str
    fingerprint: SensoryFingerprint
    created_at: datetime
    media_ref: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_aware(self.created_at, "Memory.created_at")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def context(self) -> SensoryContext:
        return self.fingerprint.context

    def to_dict(self) -> dict[str, JSONValue]:
        """JSON view for tools/APIs. Vectors and raw bytes are never included."""
        return {
            "id": self.id,
            "content": self.content,
            "media_ref": self.media_ref,
            "created_at": self.created_at.isoformat(),
            "context": self.context.to_dict(),
            "channels": _json_strs(self.fingerprint.channels),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Cue:
    """An incoming bundle of observations that may trigger recall.

    ``observations`` maps channel -> observation (at most one per channel).
    ``context.timestamp`` is when the cue was sensed (used for refractory bookkeeping and
    as "now" for decay); ``context.geohash``/``tags`` are informational. Note that the
    ``time`` and ``geo`` *channels* participate in recall only if present in
    ``observations`` - the engine's cue builder adds them from context when asked.
    """

    id: str
    observations: Mapping[ChannelName, Observation]
    context: SensoryContext
    source: str = "unknown"
    """Where it came from: ``"mcp"``, ``"http"``, ``"webhook:home_assistant"``, ``"cli"``..."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", _freeze_mapping(self.observations))
        for name, obs in self.observations.items():
            if obs.channel != name:
                raise ValueError(f"observation for {obs.channel!r} stored under key {name!r}")

    @property
    def channels(self) -> frozenset[ChannelName]:
        return frozenset(self.observations)

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "source": self.source,
            "context": self.context.to_dict(),
            "observations": {k: v.to_dict() for k, v in sorted(self.observations.items())},
        }


class RecallMode(StrEnum):
    PULL = "pull"
    """The agent asked (``recall``)."""
    PUSH = "push"
    """Involuntary: surfaced by a cue (``sense``) without the agent asking."""


@dataclass(frozen=True, slots=True)
class ScoredMemory:
    """A memory with its fused recall score and an explanation of where it came from.

    * ``score`` - final score in ``[0, 1]`` (after coverage, salience and decay).
    * ``channel_scores`` - raw per-channel similarity in ``[0, 1]`` for every channel
      shared by cue and memory (the explanation).
    * ``salience_factor`` / ``decay_factor`` - the multipliers that were applied.
    """

    memory: Memory
    score: float
    channel_scores: Mapping[ChannelName, float]
    salience_factor: float = 1.0
    decay_factor: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_scores", _freeze_mapping(self.channel_scores))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "memory": self.memory.to_dict(),
            "score": round(self.score, 6),
            "channel_scores": {k: round(v, 6) for k, v in sorted(self.channel_scores.items())},
            "salience_factor": round(self.salience_factor, 6),
            "decay_factor": round(self.decay_factor, 6),
        }


@dataclass(frozen=True, slots=True)
class Recollection:
    """A memory surfaced by a cue, as delivered to the agent.

    ``id`` is unique per delivery (prefix ``rec``); the same memory triggered twice (after
    its refractory period) yields two recollections.
    """

    id: str
    match: ScoredMemory
    cue_id: str
    cue_channels: frozenset[ChannelName]
    triggered_at: datetime
    mode: RecallMode = RecallMode.PUSH
    cue_source: str = "unknown"

    def __post_init__(self) -> None:
        _require_aware(self.triggered_at, "Recollection.triggered_at")
        object.__setattr__(self, "cue_channels", frozenset(self.cue_channels))

    @property
    def memory(self) -> Memory:
        return self.match.memory

    @property
    def score(self) -> float:
        return self.match.score

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "id": self.id,
            "mode": self.mode.value,
            "cue_id": self.cue_id,
            "cue_source": self.cue_source,
            "cue_channels": _json_strs(self.cue_channels),
            "triggered_at": self.triggered_at.isoformat(),
            **self.match.to_dict(),
        }

    def to_prompt_line(self) -> str:
        """One human/LLM-readable line, used by hooks and the MCP resource.

        Example: ``"[madeleine 0.82 via geo+time] Ana proposed at the Eiffel Tower (2024-06-01)"``.
        """
        via = "+".join(sorted(self.match.channel_scores)) or "?"
        day = self.memory.context.timestamp.date().isoformat()
        return f"[madeleine {self.score:.2f} via {via}] {self.memory.content} ({day})"


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RecallConfig:
    """Parameters of the recall fusion. The formula (normative, see ADR 0001):

    Let ``S`` = channels present in both cue and memory with weight ``w_c > 0`` and
    ``s_c in [0, 1]`` the channel similarity. Let ``C`` = channels in the cue with
    ``w_c > 0``.

    1. ``base     = sum(w_c * s_c for c in S) / sum(w_c for c in S)``  (0 if S is empty)
    2. ``coverage = (sum(w_c for c in S) / sum(w_c for c in C)) ** coverage_exponent``
    3. ``salience_factor = 1 + salience_weight * (memory.salience - 0.5)``
    4. ``age = max(0, cue.timestamp - memory.timestamp)``;
       ``decay_factor = decay_floor + (1 - decay_floor) * 0.5 ** (age / half_life)``
    5. ``score = clamp(base * coverage * salience_factor * decay_factor, 0, 1)``

    PULL (``recall``): rank by score, keep ``score >= pull_threshold``, top ``k``.
    No refractory filtering, no side effects.

    PUSH (``sense``): a memory becomes a recollection iff
    ``score >= trigger_threshold`` **and** ``len(S) >= min_shared_channels`` **and**
    at least one channel in ``S`` has ``ChannelSpec.solo_trigger`` (so "same hour" alone
    never fires) **and** the memory is not refractory (it last fired less than
    ``refractory_period`` before the cue's timestamp). At most ``max_push_per_cue``
    recollections per cue, highest score first. Each one records a trigger time
    (habituation) and is appended to the queue.
    """

    channel_weights: Mapping[ChannelName, float] = field(default_factory=dict)
    """Per-channel weight override; channels absent here use ``ChannelSpec.default_weight``."""
    trigger_threshold: float = 0.62
    pull_threshold: float = 0.0
    min_shared_channels: int = 1
    coverage_exponent: float = 0.5
    salience_weight: float = 0.4
    half_life: timedelta = timedelta(days=365)
    decay_floor: float = 0.7
    refractory_period: timedelta = timedelta(hours=6)
    max_push_per_cue: int = 3
    candidate_k: int = 64
    """How many nearest neighbours to fetch per channel before fusion."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_weights", _freeze_mapping(self.channel_weights))
        for name, w in self.channel_weights.items():
            if w < 0:
                raise ValueError(f"channel weight for {name!r} must be >= 0")
        for attr in ("trigger_threshold", "pull_threshold", "decay_floor"):
            v = getattr(self, attr)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{attr} must be in [0, 1], got {v}")
        if not 0.0 <= self.salience_weight <= 2.0:
            raise ValueError("salience_weight must be in [0, 2]")
        if self.coverage_exponent < 0:
            raise ValueError("coverage_exponent must be >= 0")
        if self.half_life <= timedelta(0):
            raise ValueError("half_life must be positive")
        if self.refractory_period < timedelta(0):
            raise ValueError("refractory_period must be >= 0")
        if self.min_shared_channels < 1 or self.max_push_per_cue < 1 or self.candidate_k < 1:
            raise ValueError("min_shared_channels, max_push_per_cue, candidate_k must be >= 1")


TextBackend: TypeAlias = Literal["auto", "sentence-transformers", "hash"]


@dataclass(frozen=True, slots=True)
class MadeleineConfig:
    """Top-level configuration (CLI flags and ``MADELEINE_*`` env vars map onto it).

    * ``db_path`` - SQLite file; ``":memory:"`` for ephemeral. Env: ``MADELEINE_DB``.
      Default ``~/.madeleine/madeleine.db``.
    * ``channels`` - enabled channels. Unavailable optional channels (missing extras)
      are skipped with a warning at startup, never at import time.
    * ``text_backend`` - ``"auto"`` uses sentence-transformers if installed (the
      ``[text]`` extra) else the hash embedder; ``"hash"`` forces the offline fake.
      Env: ``MADELEINE_TEXT_BACKEND``.
    * ``text_model`` / ``image_model`` / ``audio_model`` - model identifiers, downloaded
      lazily on first use (never at import).
    * ``store_raw`` - default for persisting raw media bytes. **False**: only embeddings
      and metadata are stored.
    * ``recall`` - fusion parameters.
    """

    db_path: str = "~/.madeleine/madeleine.db"
    channels: tuple[ChannelName, ...] = (TEXT, TIME, GEO, BIOMETRIC, IMAGE, AUDIO)
    text_backend: TextBackend = "auto"
    text_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_model: str = "ViT-B-32/laion2b_s34b_b79k"
    audio_model: str = "laion/clap-htsat-unfused"
    store_raw: bool = False
    recall: RecallConfig = field(default_factory=RecallConfig)


# --------------------------------------------------------------------------------------
# Channel protocols
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChannelSpec:
    """Static description of a channel.

    * ``dim`` - embedding dimensionality (fixed for the lifetime of a database).
    * ``default_weight`` - fusion weight when :class:`RecallConfig` does not override it.
    * ``solo_trigger`` - whether a match on this channel can, by itself, make a push
      recollection fire. ``time`` is ``False`` (every memory made at 9am must not fire at
      every 9am); ``geo``/``text``/``image``/``audio`` are ``True``.
    * ``metric`` - human-readable description of the similarity function.
    * ``model_id`` - identifier of the underlying model (``None`` for deterministic
      encoders). Stores record it to detect incompatible re-opens.
    * ``extra`` - the pip extra providing it (``"image"``), ``None`` if built in.
    """

    name: ChannelName
    dim: int
    default_weight: float = 1.0
    solo_trigger: bool = True
    metric: str = "cosine mapped to [0, 1]"
    model_id: str | None = None
    extra: str | None = None

    def __post_init__(self) -> None:
        _check_channel_name(self.name)
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        if self.default_weight < 0:
            raise ValueError("default_weight must be >= 0")


@runtime_checkable
class Adapter(Protocol):
    """Normalises a :class:`RawSignal` into an :class:`Observation` for one channel.

    Must be cheap and must not load models. Raises :class:`ObservationError` for values it
    cannot interpret (wrong type, invalid geohash, empty text...).
    """

    @property
    def channel(self) -> ChannelName: ...

    def normalize(self, signal: RawSignal) -> Observation: ...


@runtime_checkable
class Embedder(Protocol):
    """Turns an :class:`Observation` into a :data:`Vector` of length ``dim``.

    Rules:
    * Deterministic for a given model (same observation -> same vector).
    * Heavy models are loaded lazily on the first ``embed`` call, never in ``__init__``
      or at import time. A missing optional dependency raises
      :class:`ChannelNotAvailableError` naming the extra to install.
    * Vectors must be finite. Unless the channel documents otherwise they are L2-normalised.
    """

    @property
    def channel(self) -> ChannelName: ...

    @property
    def dim(self) -> int: ...

    @property
    def model_id(self) -> str | None: ...

    def embed(self, observation: Observation) -> Vector: ...

    def embed_many(self, observations: Sequence[Observation]) -> list[Vector]:
        """Batch version; must equal ``[embed(o) for o in observations]``."""
        ...


SimilarityFn: TypeAlias = Callable[[Vector, Vector], float]
"""Similarity between two vectors of one channel.

Contract: returns a float in ``[0, 1]``; symmetric; ``sim(v, v) == 1`` (within 1e-6) for
any valid vector; pure (no I/O). Example: ``max(0, cosine)``; geo uses
``exp(-distance_km / scale_km)``.
"""


@dataclass(frozen=True, slots=True)
class Channel:
    """A pluggable sense = spec + adapter + embedder + similarity function."""

    spec: ChannelSpec
    adapter: Adapter
    embedder: Embedder
    similarity: SimilarityFn

    def __post_init__(self) -> None:
        if not (self.spec.name == self.adapter.channel == self.embedder.channel):
            raise ValueError("spec, adapter and embedder must agree on the channel name")
        if self.embedder.dim != self.spec.dim:
            raise ValueError("embedder.dim must equal spec.dim")

    @property
    def name(self) -> ChannelName:
        return self.spec.name


@runtime_checkable
class ChannelRegistry(Protocol):
    """The set of senses an engine knows about."""

    def register(self, channel: Channel, *, replace: bool = False) -> None:
        """Add a channel; raises ``ValueError`` if the name exists and ``replace`` is False."""
        ...

    def get(self, name: ChannelName) -> Channel:
        """Return the channel or raise :class:`ChannelNotAvailableError`."""
        ...

    def names(self) -> tuple[ChannelName, ...]:
        """Registered channel names, sorted."""
        ...

    def __contains__(self, name: object) -> bool: ...


@runtime_checkable
class CueAdapter(Protocol):
    """Maps an external multi-channel payload (webhook JSON, Home Assistant event, iOS
    Shortcut...) to raw signals plus context hints. The engine then builds the cue.

    ``name`` identifies the adapter in URLs (``POST /cues/{name}``) and in
    ``Cue.source`` (``"webhook:{name}"``).
    """

    @property
    def name(self) -> str: ...

    def to_signals(
        self, payload: Mapping[str, Any]
    ) -> tuple[list[RawSignal], SensoryContext | None]:
        """Return (signals, context). ``context`` ``None`` means "now, no location"."""
        ...


# --------------------------------------------------------------------------------------
# Storage and queue protocols
# --------------------------------------------------------------------------------------


@runtime_checkable
class MemoryStore(Protocol):
    """Persistence for memories, their vectors and trigger history.

    The reference implementation is SQLite + sqlite-vec (one ``vec0`` table per channel)
    with a brute-force numpy fallback when the extension cannot be loaded. Also see the
    in-memory implementation used by tests. All methods are synchronous and thread-safe.
    """

    def register_channel(self, spec: ChannelSpec) -> None:
        """Declare a channel before use (idempotent). Raises :class:`StoreError` if the
        store already holds vectors for ``spec.name`` with a different ``dim`` or
        ``model_id`` (vectors from different models are not comparable)."""
        ...

    def add(self, memory: Memory) -> None:
        """Persist a memory. Raises ``ValueError`` if the id exists or a vector's length
        differs from the dimension already registered for its channel."""
        ...

    def get(self, memory_id: str) -> Memory:
        """Return the memory or raise :class:`MemoryNotFoundError`."""
        ...

    def get_many(self, memory_ids: Sequence[str]) -> list[Memory]:
        """Return the memories that exist, in the order requested (missing ids skipped)."""
        ...

    def delete(self, memory_id: str) -> bool:
        """Delete memory, vectors, raw data and trigger history. False if it did not exist."""
        ...

    def iter_memories(self, *, tags: frozenset[str] | None = None) -> Iterator[Memory]:
        """Iterate all memories (optionally only those having *all* ``tags``), oldest first."""
        ...

    def count(self) -> int: ...

    def nearest(self, channel: ChannelName, vector: Vector, k: int) -> list[tuple[str, float]]:
        """Candidate generation: up to ``k`` ``(memory_id, distance)`` pairs on ``channel``,
        nearest first. Distance is L2 between stored and query vectors (float32). The
        engine re-scores candidates with the channel's own similarity function, so this
        only needs to be a good pre-filter. Unknown channel -> empty list."""
        ...

    def record_trigger(self, memory_id: str, at: datetime) -> None:
        """Remember that ``memory_id`` fired involuntarily at ``at`` (refractory state)."""
        ...

    def last_triggered(self, memory_id: str) -> datetime | None:
        """Most recent involuntary trigger time, or ``None``."""
        ...

    def close(self) -> None: ...


Unsubscribe: TypeAlias = Callable[[], None]
RecollectionListener: TypeAlias = Callable[["Recollection"], None]


@runtime_checkable
class RecollectionQueue(Protocol):
    """FIFO of involuntary recollections waiting for the agent (the PUSH path).

    The reference implementation persists to the same SQLite file as the store so that
    separate processes (HTTP ingestion, a stdio MCP server, a Claude Code hook) share it.
    ``drain`` is atomic: a recollection is delivered to exactly one drainer.
    """

    def put(self, recollection: Recollection) -> None:
        """Append and notify in-process listeners (listener errors are logged, not raised)."""
        ...

    def peek(self, limit: int | None = None) -> list[Recollection]:
        """Pending recollections, oldest first, without removing them."""
        ...

    def drain(self, limit: int | None = None) -> list[Recollection]:
        """Remove and return pending recollections, oldest first."""
        ...

    def pending_count(self) -> int: ...

    def discard_memory(self, memory_id: str) -> int:
        """Remove pending recollections of ``memory_id`` (used by ``forget``); returns count."""
        ...

    def subscribe(self, listener: RecollectionListener) -> Unsubscribe:
        """Call ``listener`` for each recollection ``put`` *in this process*. Other
        processes' puts are discovered by polling ``pending_count``."""
        ...


# --------------------------------------------------------------------------------------
# Engine facade
# --------------------------------------------------------------------------------------


@runtime_checkable
class Clock(Protocol):
    """Injectable time source (tests use a fake)."""

    def now(self) -> datetime:
        """Timezone-aware current time."""
        ...


class SystemClock:
    """The real clock."""

    def now(self) -> datetime:
        return utc_now()


@dataclass(frozen=True, slots=True)
class RecallQuery:
    """A PULL request.

    * ``cue`` - what to match against.
    * ``k`` - maximum results.
    * ``channels`` - restrict fusion to these channels (``None`` = all shared).
    * ``tags`` - only memories having all of these tags.
    * ``min_score`` - overrides ``RecallConfig.pull_threshold`` when set.
    """

    cue: Cue
    k: int = 5
    channels: frozenset[ChannelName] | None = None
    tags: frozenset[str] | None = None
    min_score: float | None = None

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")


@runtime_checkable
class MemoryEngine(Protocol):
    """The facade used by the MCP server, HTTP ingestion, CLI, hooks and evals.

    Implemented by ``madeleine.engine.Madeleine``; construct one with
    ``madeleine.engine.create_engine(config: MadeleineConfig, *, clock=None,
    registry=None) -> Madeleine``.
    """

    @property
    def config(self) -> MadeleineConfig: ...

    @property
    def channels(self) -> ChannelRegistry: ...

    @property
    def store(self) -> MemoryStore: ...

    @property
    def queue(self) -> RecollectionQueue: ...

    def build_cue(
        self,
        signals: Sequence[RawSignal] = (),
        *,
        context: SensoryContext | None = None,
        include_time: bool = True,
        include_geo: bool = True,
        source: str = "unknown",
    ) -> Cue:
        """Normalise signals via channel adapters into a :class:`Cue`.

        ``context`` defaults to ``SensoryContext(timestamp=clock.now())``. When
        ``include_time`` the ``time`` observation is derived from ``context.timestamp``;
        when ``include_geo`` and ``context.geohash`` is set, the ``geo`` observation is
        derived from it. Explicit ``time``/``geo`` signals win over derived ones. Signals
        on unknown/disabled channels raise :class:`ChannelNotAvailableError`.
        """
        ...

    def remember(
        self,
        content: str,
        signals: Sequence[RawSignal] = (),
        *,
        context: SensoryContext | None = None,
        media_ref: str | None = None,
        metadata: Mapping[str, str] | None = None,
        embed_content: bool = True,
        store_raw: bool | None = None,
    ) -> Memory:
        """Encode a memory with its sensory fingerprint and persist it.

        The fingerprint gets: every signal's embedding; ``time`` from
        ``context.timestamp`` (default now); ``geo`` from ``context.geohash`` if set; and,
        if ``embed_content`` and no explicit ``text`` signal was given, ``text`` from
        ``content``. ``store_raw`` (default ``config.store_raw``) controls whether raw
        media bytes are persisted.
        """
        ...

    def recall(self, query: RecallQuery) -> list[ScoredMemory]:
        """PULL: score and rank memories for the cue. No side effects."""
        ...

    def sense(self, cue: Cue) -> list[Recollection]:
        """PUSH: evaluate the cue for involuntary recall; enqueue and return the
        recollections that fired (possibly none). Records trigger times."""
        ...

    def pending_recollections(
        self, *, limit: int | None = None, drain: bool = True
    ) -> list[Recollection]:
        """Read (``drain=True``: and remove) queued involuntary recollections."""
        ...

    def forget(self, memory_id: str) -> bool:
        """Delete a memory everywhere (store, vectors, raw data, queue entries)."""
        ...

    def close(self) -> None: ...
