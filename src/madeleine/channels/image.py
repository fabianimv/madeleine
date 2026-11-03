"""The ``image`` channel: still images.

* **Adapter** - bytes (``mime_type`` given or sniffed from the magic number) or a local
  reference (``uri=`` / a string value: ``file://`` URI or a path). Remote URLs
  (``http(s)://``...) are rejected: madeleine never downloads media. The adapter does not
  read files; it validates that the file exists and records a ``file://`` URI.
* **Embedder** - :class:`OpenClipEmbedder` (``[image]`` extra; default ``ViT-B-32`` with
  ``laion2b_s34b_b79k`` weights, 512-d, ~600 MB download on first use), or any
  :class:`~madeleine.core.types.Embedder` you pass (the hash registry uses
  :class:`~madeleine.channels.hashing.HashEmbedder`).
* **Similarity** - :func:`~madeleine.channels.similarity.rescaled_cosine` with
  ``IMAGE_COSINE_LO = 0.5`` and ``IMAGE_COSINE_HI = 0.95``: CLIP image-image cosines of
  unrelated photos typically sit around 0.4-0.6 and near-duplicates above 0.9, so plain
  ``max(0, cos)`` would make everything look alike. These constants are heuristics; the
  evals are the place to tune them.

Privacy: bytes live only in the observation; the engine drops them before persisting
unless ``store_raw=True``. Embedders never log image content.
"""

from __future__ import annotations

import io
import logging
import math
import threading
from collections.abc import Sequence
from typing import Any, Final

from madeleine.channels._extras import import_extra
from madeleine.channels._media import (
    base_mime,
    coerce_bytes,
    guess_mime_from_path,
    normalize_file_uri,
    read_media_bytes,
    sniff_image_mime,
)
from madeleine.channels.hashing import HashEmbedder
from madeleine.channels.similarity import cosine01, describe_similarity, rescaled_cosine
from madeleine.core.types import (
    IMAGE,
    Channel,
    ChannelName,
    ChannelNotAvailableError,
    ChannelSpec,
    Embedder,
    MadeleineConfig,
    Observation,
    ObservationError,
    RawSignal,
    SimilarityFn,
    Vector,
)

__all__ = [
    "DEFAULT_IMAGE_MODEL",
    "DEFAULT_IMAGE_PRETRAINED",
    "IMAGE_COSINE_HI",
    "IMAGE_COSINE_LO",
    "MAX_MEDIA_BYTES",
    "ImageAdapter",
    "OpenClipEmbedder",
    "image_similarity",
    "make_image_channel",
    "parse_image_model",
]

_log = logging.getLogger(__name__)

DEFAULT_IMAGE_MODEL: Final = "ViT-B-32"
DEFAULT_IMAGE_PRETRAINED: Final = "laion2b_s34b_b79k"
IMAGE_COSINE_LO: Final = 0.5
IMAGE_COSINE_HI: Final = 0.95
MAX_MEDIA_BYTES: Final = 32 * 1024 * 1024
"""Largest accepted in-memory media payload (32 MiB)."""

_KNOWN_DIMS: Final[dict[str, int]] = {
    "ViT-B-32": 512,
    "ViT-B-32-quickgelu": 512,
    "ViT-B-16": 512,
    "ViT-L-14": 768,
    "ViT-H-14": 1024,
}

image_similarity: Final = rescaled_cosine(IMAGE_COSINE_LO, IMAGE_COSINE_HI)
"""Cosine mapped from ``[0.5, 0.95]`` to ``[0, 1]`` (clamped)."""


def parse_image_model(spec: str) -> tuple[str, str | None]:
    """``"ViT-B-32/laion2b_s34b_b79k"`` -> ``("ViT-B-32", "laion2b_s34b_b79k")``.

    ``"hf-hub:org/repo"`` is returned whole with ``pretrained=None``.
    """
    if spec.startswith("hf-hub:"):
        return spec, None
    model, sep, pretrained = spec.rpartition("/")
    if not sep:
        return spec, None
    return model, pretrained or None


class ImageAdapter:
    """Bytes (+ optional ``mime_type``) or a local ``uri``/path -> image observation."""

    def __init__(self, *, max_bytes: int = MAX_MEDIA_BYTES) -> None:
        self._max_bytes = max_bytes

    @property
    def channel(self) -> ChannelName:
        return IMAGE

    def normalize(self, signal: RawSignal) -> Observation:
        if signal.channel != IMAGE:
            raise ObservationError(f"ImageAdapter got a signal for {signal.channel!r}")
        data = coerce_bytes(signal.value)
        if data is not None:
            if not data:
                raise ObservationError("image signal has empty bytes")
            if len(data) > self._max_bytes:
                raise ObservationError(f"image is larger than {self._max_bytes} bytes")
            mime = base_mime(signal.mime_type) or sniff_image_mime(data)
            if mime is None:
                raise ObservationError(
                    "unrecognised image bytes; pass mime_type (e.g. 'image/png')"
                )
            if not mime.startswith("image/"):
                raise ObservationError(f"image signal has non-image mime_type {mime!r}")
            return Observation(IMAGE, data=data, mime_type=mime, uri=signal.uri)
        ref = signal.value if isinstance(signal.value, str) else None
        ref = ref or signal.uri
        if signal.value is not None and not isinstance(signal.value, str):
            raise ObservationError(
                f"image signal must be bytes or a file uri/path, got {type(signal.value).__name__}"
            )
        if not ref:
            raise ObservationError("image signal needs bytes or a file uri")
        uri, path = normalize_file_uri(ref, IMAGE)
        mime = base_mime(signal.mime_type) or guess_mime_from_path(path)
        if mime is not None and not mime.startswith("image/"):
            raise ObservationError(f"image file has non-image mime_type {mime!r}")
        return Observation(IMAGE, uri=uri, mime_type=mime)


class OpenClipEmbedder:
    """Image embedder backed by ``open_clip`` (``[image]`` extra), loaded lazily.

    Vectors are the L2-normalised image features. Deterministic in eval mode on a given
    device. Reads ``data`` bytes, else the local file at ``uri``.
    """

    def __init__(
        self,
        model: str = DEFAULT_IMAGE_MODEL,
        pretrained: str | None = DEFAULT_IMAGE_PRETRAINED,
        *,
        device: str | None = None,
        dim: int | None = None,
    ) -> None:
        self._model_name = model
        self._pretrained = pretrained
        self._device = device
        self._dim = dim if dim is not None else _KNOWN_DIMS.get(model)
        self._state: tuple[Any, Any, Any] | None = None  # (model, preprocess, torch)
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"OpenClipEmbedder({self._model_name!r}, {self._pretrained!r})"

    @property
    def channel(self) -> ChannelName:
        return IMAGE

    @property
    def model_id(self) -> str:
        if self._pretrained:
            return f"open_clip/{self._model_name}/{self._pretrained}"
        return f"open_clip/{self._model_name}"

    @property
    def dim(self) -> int:
        if self._dim is None:
            model, _, _ = self._load()
            dim = getattr(getattr(model, "visual", None), "output_dim", None)
            if not isinstance(dim, int):
                raise ChannelNotAvailableError(
                    f"cannot determine embedding size of {self._model_name!r}; pass dim="
                )
            self._dim = dim
        return self._dim

    @property
    def loaded(self) -> bool:
        return self._state is not None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._state is not None:
            return self._state
        with self._lock:
            if self._state is None:
                open_clip = import_extra("open_clip", channel=IMAGE, extra="image")
                torch = import_extra("torch", channel=IMAGE, extra="image")
                import_extra("PIL.Image", channel=IMAGE, extra="image")
                _log.info("loading image model %s (first use; may download)", self.model_id)
                device = self._device or "cpu"
                model, _, preprocess = open_clip.create_model_and_transforms(
                    self._model_name, pretrained=self._pretrained, device=device
                )
                model.eval()
                self._state = (model, preprocess, torch)
        return self._state

    def embed(self, observation: Observation) -> Vector:
        if observation.channel != IMAGE:
            raise ObservationError(
                f"OpenClipEmbedder got an observation for {observation.channel!r}"
            )
        model, preprocess, torch = self._load()
        pil_image = import_extra("PIL.Image", channel=IMAGE, extra="image")
        raw = read_media_bytes(observation)
        try:
            with pil_image.open(io.BytesIO(raw)) as img:
                rgb = img.convert("RGB")
        except Exception as exc:  # PIL raises several unrelated types
            raise ObservationError(f"cannot decode image: {type(exc).__name__}") from exc
        device = self._device or "cpu"
        with torch.no_grad():
            batch = preprocess(rgb).unsqueeze(0).to(device)
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        vec = tuple(float(x) for x in feats[0].cpu().tolist())
        if not all(math.isfinite(x) for x in vec):
            raise ObservationError("image model produced a non-finite embedding")
        return vec

    def embed_many(self, observations: Sequence[Observation]) -> list[Vector]:
        return [self.embed(o) for o in observations]


def make_image_channel(
    config: MadeleineConfig | None = None,
    *,
    embedder: Embedder | None = None,
    similarity: SimilarityFn | None = None,
) -> Channel:
    """The ``image`` channel.

    Defaults to :class:`OpenClipEmbedder` for ``config.image_model``
    (``"<model>/<pretrained>"``) with :data:`image_similarity`. A custom ``embedder``
    defaults to :data:`image_similarity` too unless it is a ``HashEmbedder`` (then
    ``cosine01``); pass ``similarity`` to override.
    """
    cfg = config or MadeleineConfig()
    if embedder is None:
        model, pretrained = parse_image_model(cfg.image_model)
        embedder = OpenClipEmbedder(model, pretrained)
    if similarity is None:
        similarity = cosine01 if isinstance(embedder, HashEmbedder) else image_similarity
    is_clip = isinstance(embedder, OpenClipEmbedder)
    spec = ChannelSpec(
        name=IMAGE,
        dim=embedder.dim,
        default_weight=1.0,
        solo_trigger=True,
        metric=describe_similarity(similarity),
        model_id=embedder.model_id,
        extra="image" if is_clip else None,
    )
    return Channel(spec=spec, adapter=ImageAdapter(), embedder=embedder, similarity=similarity)
