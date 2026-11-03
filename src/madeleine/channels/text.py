"""The ``text`` channel: natural language.

* **Adapter** - ``str`` -> ``Observation(text=stripped)``; empty/non-string -> error.
* **Embedders** - :class:`SentenceTransformerEmbedder` (``[text]`` extra; default model
  ``sentence-transformers/all-MiniLM-L6-v2``, 384-d, ~90 MB download on first use) or the
  offline :class:`~madeleine.channels.hashing.HashEmbedder` (lexical, 256-d).
* **Similarity** - ``max(0, cosine)`` (vectors are L2-normalised).

Backend selection (:func:`make_text_channel`, ``MadeleineConfig.text_backend``):

* ``"auto"`` - sentence-transformers if importable (checked with ``find_spec``, without
  importing it), else the hash embedder with a one-time warning.
* ``"hash"`` - always the hash embedder (tests, demo, CI).
* ``"sentence-transformers"`` - always sentence-transformers; if it is not installed the
  first ``embed`` raises :class:`~madeleine.core.types.ChannelNotAvailableError` with the
  pip hint.
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Sequence
from typing import Any, Final

from madeleine.channels._extras import import_extra, module_available
from madeleine.channels.hashing import HashEmbedder
from madeleine.channels.similarity import cosine01, describe_similarity, rescaled_cosine
from madeleine.core.types import (
    TEXT,
    Channel,
    ChannelName,
    ChannelNotAvailableError,
    ChannelSpec,
    Embedder,
    MadeleineConfig,
    Observation,
    ObservationError,
    RawSignal,
    TextBackend,
    Vector,
)

__all__ = [
    "DEFAULT_TEXT_MODEL",
    "HASH_TEXT_COSINE_RANGE",
    "SentenceTransformerEmbedder",
    "TextAdapter",
    "hash_text_similarity",
    "make_text_channel",
    "text_similarity",
]

_log = logging.getLogger(__name__)

DEFAULT_TEXT_MODEL: Final = "sentence-transformers/all-MiniLM-L6-v2"

# Embedding sizes of common sentence-transformers models, so ``dim`` is known without
# loading the model. Any other model is loaded lazily the first time ``dim`` is read.
_KNOWN_DIMS: Final[dict[str, int]] = {
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "multi-qa-MiniLM-L6-cos-v1": 384,
    "paraphrase-MiniLM-L6-v2": 384,
    "paraphrase-multilingual-MiniLM-L12-v2": 384,
}

text_similarity: Final = cosine01
"""``max(0, cosine)``: the similarity for sentence-transformers text embeddings."""

HASH_TEXT_COSINE_RANGE: Final = (0.10, 0.45)
"""Calibration of the lexical hash embedder's compressed cosine scale.

Measured on the synthetic benchmark generator with seed 1 (the reported benchmark uses
seed 0): unrelated pairs have median cosine 0.02 and p99 0.20 (mostly same-topic texts);
paraphrases share only part of their words, median 0.34, p90 0.40. Without rescaling a
paraphrase would count as "34 % similar" and drag down every multi-channel cue it is part
of. ``rescaled_cosine(0.10, 0.45)`` maps "unrelated" to ~0 and "paraphrase" to ~0.7,
the same treatment CLIP/CLAP get for their (opposite) compression.
"""
hash_text_similarity: Final = rescaled_cosine(*HASH_TEXT_COSINE_RANGE)
"""Cosine mapped from ``[0.10, 0.45]`` to ``[0, 1]`` (clamped): the similarity for the
hash text backend (see :data:`HASH_TEXT_COSINE_RANGE`)."""

_auto_fallback_warned = False
_warn_lock = threading.Lock()


class TextAdapter:
    """``str`` -> ``Observation(TEXT, text=value.strip())``."""

    @property
    def channel(self) -> ChannelName:
        return TEXT

    def normalize(self, signal: RawSignal) -> Observation:
        if signal.channel != TEXT:
            raise ObservationError(f"TextAdapter got a signal for {signal.channel!r}")
        if not isinstance(signal.value, str):
            raise ObservationError(
                f"text signal must be a string, got {type(signal.value).__name__}"
            )
        text = signal.value.strip()
        if not text:
            raise ObservationError("text signal is empty")
        return Observation(TEXT, text=text)


def _known_dim(model_name: str) -> int | None:
    return _KNOWN_DIMS.get(model_name.removeprefix("sentence-transformers/"))


class SentenceTransformerEmbedder:
    """Text embedder backed by ``sentence-transformers`` (``[text]`` extra).

    The model is imported and loaded on the first :meth:`embed` call (thread-safe), never
    in ``__init__``. ``dim`` is known up front for the models in ``_KNOWN_DIMS`` (or when
    passed explicitly); for any other model reading ``dim`` loads the model.
    Embeddings are L2-normalised (``normalize_embeddings=True``).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        *,
        device: str | None = None,
        dim: int | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._dim = dim if dim is not None else _known_dim(model_name)
        self._model: Any = None
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"SentenceTransformerEmbedder({self._model_name!r})"

    @property
    def channel(self) -> ChannelName:
        return TEXT

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        if self._dim is None:
            model = self._load()
            getter = getattr(model, "get_sentence_embedding_dimension", None) or getattr(
                model, "get_embedding_dimension", None
            )
            dim = getter() if getter is not None else None
            if not isinstance(dim, int):
                raise ChannelNotAvailableError(
                    f"cannot determine embedding size of {self._model_name!r}; pass dim="
                )
            self._dim = dim
        return self._dim

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                st = import_extra("sentence_transformers", channel=TEXT, extra="text")
                _log.info("loading text model %s (first use; may download)", self._model_name)
                self._model = st.SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def embed(self, observation: Observation) -> Vector:
        if observation.channel != TEXT:
            raise ObservationError(
                f"SentenceTransformerEmbedder got an observation for {observation.channel!r}"
            )
        if not observation.text:
            raise ObservationError("text observation has no text")
        model = self._load()
        arr = model.encode(
            [observation.text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vec = tuple(float(x) for x in arr[0].tolist())
        if not all(math.isfinite(x) for x in vec):
            raise ObservationError("text model produced a non-finite embedding")
        if self._dim is not None and len(vec) != self._dim:
            raise ChannelNotAvailableError(
                f"text model {self._model_name!r} returned {len(vec)}-d vectors, "
                f"expected {self._dim}"
            )
        return vec

    def embed_many(self, observations: Sequence[Observation]) -> list[Vector]:
        # One call per item keeps embed_many bit-identical to embed (batch padding can
        # perturb float32 results in the last bits).
        return [self.embed(o) for o in observations]


def _hash_embedder(dim: int) -> HashEmbedder:
    return HashEmbedder(TEXT, dim)


def _resolve_backend(backend: TextBackend) -> TextBackend:
    global _auto_fallback_warned
    if backend != "auto":
        return backend
    if module_available("sentence_transformers"):
        return "sentence-transformers"
    with _warn_lock:
        if not _auto_fallback_warned:
            _auto_fallback_warned = True
            _log.warning(
                "sentence-transformers is not installed: text recall uses the offline lexical "
                "hash embedder. For semantic text recall: pip install 'madeleine-mcp[text]'"
            )
    return "hash"


def make_text_channel(
    config: MadeleineConfig | None = None,
    *,
    backend: TextBackend | None = None,
    model: str | None = None,
    hash_dim: int = 256,
    embedder: Embedder | None = None,
) -> Channel:
    """The ``text`` channel.

    ``backend``/``model`` default to ``config.text_backend``/``config.text_model``.
    Pass ``embedder`` to plug in any custom text embedder (``backend`` is then ignored).
    """
    cfg = config or MadeleineConfig()
    if embedder is None:
        chosen = _resolve_backend(backend or cfg.text_backend)
        if chosen == "hash":
            embedder = _hash_embedder(hash_dim)
        elif chosen == "sentence-transformers":
            embedder = SentenceTransformerEmbedder(model or cfg.text_model)
        else:  # pragma: no cover - guarded by the TextBackend literal
            raise ValueError(f"unknown text backend {chosen!r}")
    is_st = isinstance(embedder, SentenceTransformerEmbedder)
    is_hash = embedder.model_id is not None and embedder.model_id.startswith("hash-")
    similarity = hash_text_similarity if is_hash else cosine01
    spec = ChannelSpec(
        name=TEXT,
        dim=embedder.dim,
        default_weight=1.0,
        solo_trigger=True,
        metric=describe_similarity(similarity),
        model_id=embedder.model_id,
        extra="text" if is_st else None,
    )
    return Channel(spec=spec, adapter=TextAdapter(), embedder=embedder, similarity=similarity)
