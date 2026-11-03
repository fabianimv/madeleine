"""The ``audio`` channel: sound clips.

* **Adapter** - encoded bytes (WAV/FLAC/OGG, anything ``soundfile`` decodes; ``mime_type``
  given or sniffed), raw 16-bit PCM (``mime_type="audio/L16; rate=16000"`` big-endian per
  RFC 2586, or ``"audio/pcm; rate=16000"`` little-endian; optional ``channels=N``), or,
  only with ``allow_local_files=True``, a local reference (``uri=`` / string value:
  ``file://`` or a path; refused by default without touching the filesystem). Remote URLs
  are always rejected. Files are capped at ``MAX_MEDIA_BYTES``. Raw PCM sets
  ``numbers["sample_rate"]`` (and ``numbers["channels"]``).
* **Embedder** - :class:`ClapEmbedder` (``[audio]`` extra; ``laion/clap-htsat-unfused``
  via ``transformers``' ``ClapModel``/``ClapProcessor``, 512-d, ~600 MB download on first
  use). Audio is mixed down to mono, **resampled to 48 kHz** (the rate CLAP was trained
  on) with a numpy box-filter + linear-interpolation resampler, and **truncated to the
  first 10 s** (CLAP's window) so that the processor's random cropping never kicks in and
  embeddings are deterministic. Shorter clips are repeat-padded by the processor.
* **Similarity** - :func:`~madeleine.channels.similarity.rescaled_cosine` with
  ``AUDIO_COSINE_LO = 0.4`` and ``AUDIO_COSINE_HI = 0.95`` (CLAP audio-audio cosines of
  unrelated clips are well above zero). Heuristic constants; tune with the evals.

Privacy: bytes are never persisted by default and never logged.
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
    MAX_MEDIA_BYTES,
    base_mime,
    check_uri_label,
    checked_local_file,
    coerce_bytes,
    guess_mime_from_path,
    mime_params,
    normalize_file_uri,
    sniff_audio_mime,
)
from madeleine.channels.hashing import HashEmbedder
from madeleine.channels.similarity import cosine01, describe_similarity, rescaled_cosine
from madeleine.core.types import (
    AUDIO,
    Channel,
    ChannelName,
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
    "AUDIO_COSINE_HI",
    "AUDIO_COSINE_LO",
    "CLAP_MAX_SECONDS",
    "CLAP_SAMPLE_RATE",
    "DEFAULT_AUDIO_MODEL",
    "AudioAdapter",
    "ClapEmbedder",
    "audio_similarity",
    "make_audio_channel",
    "resample",
]

_log = logging.getLogger(__name__)

DEFAULT_AUDIO_MODEL: Final = "laion/clap-htsat-unfused"
CLAP_SAMPLE_RATE: Final = 48_000
CLAP_MAX_SECONDS: Final = 10.0
AUDIO_COSINE_LO: Final = 0.4
AUDIO_COSINE_HI: Final = 0.95
_RAW_PCM: Final = {"audio/l16": ">i2", "audio/pcm": "<i2"}

audio_similarity: Final = rescaled_cosine(AUDIO_COSINE_LO, AUDIO_COSINE_HI)
"""Cosine mapped from ``[0.4, 0.95]`` to ``[0, 1]`` (clamped)."""


def _positive_int(text: str | None, what: str) -> int | None:
    if text is None:
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise ObservationError(f"audio mime_type parameter {what}={text!r} is not an int") from exc
    if value < 1:
        raise ObservationError(f"audio mime_type parameter {what} must be >= 1")
    return value


class AudioAdapter:
    """Encoded/raw-PCM bytes or a local ``uri``/path -> audio observation.

    ``allow_local_files`` (default **False**) gates every local file reference: a
    ``file://``/path ``uri`` or string ``value``, and a local ``uri`` sent alongside bytes.
    """

    def __init__(
        self, *, max_bytes: int = MAX_MEDIA_BYTES, allow_local_files: bool = False
    ) -> None:
        self._max_bytes = max_bytes
        self._allow_local_files = allow_local_files

    @property
    def allow_local_files(self) -> bool:
        return self._allow_local_files

    @property
    def channel(self) -> ChannelName:
        return AUDIO

    def normalize(self, signal: RawSignal) -> Observation:
        if signal.channel != AUDIO:
            raise ObservationError(f"AudioAdapter got a signal for {signal.channel!r}")
        data = coerce_bytes(signal.value)
        if data is not None:
            return self._from_bytes(data, signal)
        if signal.value is not None and not isinstance(signal.value, str):
            raise ObservationError(
                f"audio signal must be bytes or a file uri/path, got {type(signal.value).__name__}"
            )
        ref = (signal.value if isinstance(signal.value, str) else None) or signal.uri
        if not ref:
            raise ObservationError("audio signal needs bytes or a file uri")
        uri, path = normalize_file_uri(
            ref, AUDIO, allow_local_files=self._allow_local_files, max_bytes=self._max_bytes
        )
        mime = base_mime(signal.mime_type) or guess_mime_from_path(path)
        if mime is not None and not mime.startswith("audio/"):
            raise ObservationError(f"audio file has non-audio mime_type {mime!r}")
        return Observation(AUDIO, uri=uri, mime_type=mime)

    def _from_bytes(self, data: bytes, signal: RawSignal) -> Observation:
        if not data:
            raise ObservationError("audio signal has empty bytes")
        if len(data) > self._max_bytes:
            raise ObservationError(f"audio is larger than {self._max_bytes} bytes")
        mime = base_mime(signal.mime_type) or sniff_audio_mime(data)
        if mime is None:
            raise ObservationError(
                "unrecognised audio bytes; pass mime_type (e.g. 'audio/wav', "
                "'audio/L16; rate=16000')"
            )
        if not mime.startswith("audio/"):
            raise ObservationError(f"audio signal has non-audio mime_type {mime!r}")
        check_uri_label(signal.uri, AUDIO, allow_local_files=self._allow_local_files)
        if mime in _RAW_PCM:
            params = mime_params(signal.mime_type)
            rate = _positive_int(params.get("rate"), "rate")
            if rate is None:
                raise ObservationError(f"raw PCM ({mime}) needs a rate, e.g. '{mime}; rate=16000'")
            channels = _positive_int(params.get("channels"), "channels") or 1
            if len(data) % (2 * channels):
                raise ObservationError("raw PCM length is not a whole number of 16-bit frames")
            return Observation(
                AUDIO,
                data=data,
                mime_type=mime,
                uri=signal.uri,
                numbers={"sample_rate": float(rate), "channels": float(channels)},
            )
        return Observation(AUDIO, data=data, mime_type=mime, uri=signal.uri)


def resample(wave: Any, rate_in: int, rate_out: int) -> Any:
    """Resample a 1-D float numpy array (box-filter anti-alias + linear interpolation).

    Simple and dependency-free (numpy only); adequate for CLAP's 48 kHz input, not for
    hi-fi audio work.
    """
    import numpy as np

    x = np.asarray(wave, dtype=np.float64)
    if rate_in == rate_out or x.size == 0:
        return x.astype(np.float32)
    if rate_out < rate_in:
        width = math.ceil(rate_in / rate_out)
        if width > 1:
            kernel = np.ones(width) / width
            x = np.convolve(x, kernel, mode="same")
    n_out = max(1, round(x.size * rate_out / rate_in))
    t_out = np.arange(n_out) * (rate_in / rate_out)
    return np.interp(t_out, np.arange(x.size), x).astype(np.float32)


def decode_audio(observation: Observation) -> tuple[Any, int]:
    """Decode an audio observation to ``(mono float32 numpy array, sample_rate)``."""
    import numpy as np

    mime = base_mime(observation.mime_type)
    if observation.data is not None and mime in _RAW_PCM:
        rate = int(observation.numbers.get("sample_rate", 0))
        channels = int(observation.numbers.get("channels", 1)) or 1
        if rate < 1:
            raise ObservationError("raw PCM observation needs numbers['sample_rate']")
        pcm = np.frombuffer(observation.data, dtype=_RAW_PCM[mime]).astype(np.float32) / 32768.0
        return pcm.reshape(-1, channels).mean(axis=1), rate
    sf = import_extra("soundfile", channel=AUDIO, extra="audio")
    if observation.data is not None:
        source: Any = io.BytesIO(observation.data)
    elif observation.uri is not None:
        source = str(checked_local_file(observation.uri, AUDIO))
    else:
        raise ObservationError("audio observation has neither data nor uri")
    try:
        wave, rate = sf.read(source, dtype="float32", always_2d=True)
    except Exception as exc:  # libsndfile errors vary by version
        raise ObservationError(f"cannot decode audio: {type(exc).__name__}") from exc
    if wave.size == 0:
        raise ObservationError("audio clip is empty")
    return wave.mean(axis=1), int(rate)


class ClapEmbedder:
    """Audio embedder backed by LAION CLAP via ``transformers`` (``[audio]`` extra).

    Loaded lazily on the first :meth:`embed`. Vectors are the L2-normalised projected
    audio features (512-d).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_AUDIO_MODEL,
        *,
        device: str | None = None,
        dim: int = 512,
        max_seconds: float = CLAP_MAX_SECONDS,
    ) -> None:
        self._model_name = model_name
        self._device = device or "cpu"
        self._dim = dim
        self._max_seconds = max_seconds
        self._state: tuple[Any, Any, Any] | None = None  # (model, processor, torch)
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"ClapEmbedder({self._model_name!r})"

    @property
    def channel(self) -> ChannelName:
        return AUDIO

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_name

    @property
    def loaded(self) -> bool:
        return self._state is not None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._state is not None:
            return self._state
        with self._lock:
            if self._state is None:
                transformers = import_extra("transformers", channel=AUDIO, extra="audio")
                torch = import_extra("torch", channel=AUDIO, extra="audio")
                import_extra("soundfile", channel=AUDIO, extra="audio")
                _log.info("loading audio model %s (first use; may download)", self._model_name)
                processor = transformers.ClapProcessor.from_pretrained(self._model_name)
                model = transformers.ClapModel.from_pretrained(self._model_name)
                model.eval()
                model.to(self._device)
                self._state = (model, processor, torch)
        return self._state

    def embed(self, observation: Observation) -> Vector:
        if observation.channel != AUDIO:
            raise ObservationError(f"ClapEmbedder got an observation for {observation.channel!r}")
        model, processor, torch = self._load()
        wave, rate = decode_audio(observation)
        wave = resample(wave, rate, CLAP_SAMPLE_RATE)[: int(self._max_seconds * CLAP_SAMPLE_RATE)]
        inputs = processor.feature_extractor(
            [wave], sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items() if hasattr(v, "to")}
        with torch.no_grad():
            out = model.get_audio_features(**inputs)
        feats = out if hasattr(out, "norm") else out.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        vec = tuple(float(x) for x in feats[0].cpu().tolist())
        if len(vec) != self._dim or not all(math.isfinite(x) for x in vec):
            raise ObservationError(
                f"audio model produced an invalid embedding (len {len(vec)}, expected {self._dim})"
            )
        return vec

    def embed_many(self, observations: Sequence[Observation]) -> list[Vector]:
        return [self.embed(o) for o in observations]


def make_audio_channel(
    config: MadeleineConfig | None = None,
    *,
    embedder: Embedder | None = None,
    similarity: SimilarityFn | None = None,
    allow_local_files: bool | None = None,
) -> Channel:
    """The ``audio`` channel.

    Defaults to :class:`ClapEmbedder` for ``config.audio_model`` with
    :data:`audio_similarity`; a ``HashEmbedder`` defaults to ``cosine01``.
    ``allow_local_files`` (default ``config.allow_local_files``, itself False) is passed to
    :class:`AudioAdapter`.
    """
    cfg = config or MadeleineConfig()
    allow = cfg.allow_local_files if allow_local_files is None else allow_local_files
    if embedder is None:
        embedder = ClapEmbedder(cfg.audio_model)
    if similarity is None:
        similarity = cosine01 if isinstance(embedder, HashEmbedder) else audio_similarity
    spec = ChannelSpec(
        name=AUDIO,
        dim=embedder.dim,
        default_weight=1.0,
        solo_trigger=True,
        metric=describe_similarity(similarity),
        model_id=embedder.model_id,
        extra="audio" if isinstance(embedder, ClapEmbedder) else None,
    )
    adapter = AudioAdapter(allow_local_files=allow)
    return Channel(spec=spec, adapter=adapter, embedder=embedder, similarity=similarity)
