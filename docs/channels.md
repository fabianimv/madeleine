# Channels (senses)

A **channel** is one sense: a `ChannelSpec` plus an `Adapter` (raw signal → `Observation`),
an `Embedder` (`Observation` → vector) and a `SimilarityFn` (vector × vector → `[0, 1]`).
The contract lives in `src/madeleine/core/types.py`; the implementations live in
`src/madeleine/channels/`, one module per sense.

Importing `madeleine.channels` is cheap. torch, open_clip, transformers and
sentence-transformers are imported only on the **first `embed` call** of the embedder that
needs them. If the library is missing, that call raises `ChannelNotAvailableError` with
the pip extra to install.

## Built-in channels

| Channel | Observation fields read | Embedder (model) | dim | Similarity → [0, 1] | weight | solo | Extra | First-use download |
|---|---|---|---|---|---|---|---|---|
| `text` | `text` | `SentenceTransformerEmbedder` (`sentence-transformers/all-MiniLM-L6-v2`) or `HashEmbedder` | 384 (hash: 256) | `max(0, cos)` (hash: cosine rescaled from [0.10, 0.45]) | 1.0 | yes | `[text]` | ~90 MB |
| `image` | `data` (bytes) or `uri` (local file), `mime_type` | `OpenClipEmbedder` (`ViT-B-32`, `laion2b_s34b_b79k`) | 512 | cosine rescaled `[0.6, 0.95] → [0, 1]` | 1.0 | yes | `[image]` | ~600 MB |
| `audio` | `data` or `uri`, `mime_type`, `numbers.sample_rate` (raw PCM) | `ClapEmbedder` (`laion/clap-htsat-unfused`) | 512 | cosine rescaled `[0.4, 0.95] → [0, 1]` | 1.0 | yes | `[audio]` | ~600 MB |
| `time` | `timestamp` | `TimeEncoder` (cyclical, deterministic) | 6 | weighted mean of `(1 + cos Δθ)/2` | 0.5 | **no** | – | – |
| `geo` | `text` (geohash), `numbers.lat`/`lon` | `GeoEmbedder` (unit-sphere xyz) | 3 | `exp(-d_km / 1.0)` | 1.0 | yes | – | – |
| `biometric` | `numbers` | `BiometricEncoder` (fixed ranges + presence mask) | 12 | mean Gaussian kernel over shared features | 0.5 | **no** | – | – |
| `smell` | – | none: the adapter raises `ChannelNotAvailableError` | – | – | – | – | future | – |

Download sizes are approximate. Weights are fetched from the Hugging Face Hub on first use
and cached there (`~/.cache/huggingface`).

### Building registries

```python
from madeleine.channels import default_registry, hash_registry
from madeleine.config import load_config

registry = default_registry(load_config())  # real backends; nothing loads until first use
offline = hash_registry()                   # HashEmbedder for text/image/audio; never downloads
```

* `default_registry(config)` registers `config.channels`. `image`/`audio` are registered
  only if their extra is installed, which it checks with `importlib.util.find_spec` (this
  never imports torch). If the extra is missing, the channel is skipped and a warning is
  logged. `text` is always registered. `smell` is skipped with a warning. An unknown name
  raises `ChannelNotAvailableError`.
* `text_backend`: `"auto"` uses sentence-transformers if it is installed, and otherwise
  the hash embedder (with one warning per process). `"hash"` always uses the hash
  embedder. `"sentence-transformers"` requires the library, and the first `embed` raises
  if it is missing.
* `hash_registry(channels=..., dim=256)` uses `HashEmbedder` + `cosine01` for
  text/image/audio and the real deterministic encoders for time/geo/biometric. Tests, the
  demo and the evals use it.

The stores record each channel's `dim` and `model_id`, so switching the text backend on
an existing database is refused (hash: `hash-v1/dim=256`; MiniLM:
`sentence-transformers/all-MiniLM-L6-v2`).

## Per-channel details

### text (`channels/text.py`)

`TextAdapter` accepts a `str`, strips it, and rejects empty or non-string values.
`SentenceTransformerEmbedder(model_name, *, device=None, dim=None)` returns
L2-normalised embeddings (`normalize_embeddings=True`). `dim` is known without loading the
model for common models (MiniLM-L6/L12 = 384, mpnet = 768, …) or when you pass `dim=`.
For any other model, reading `dim` loads the model.

### Offline `HashEmbedder` (`channels/hashing.py`)

This embedder is deterministic, offline and fast, and is built on BLAKE2b. It never uses
Python's salted `hash()`, so vectors are identical across processes and machines.

* **Text**: signed feature hashing of NFKC-casefolded word unigrams (weight 1; English
  stop words 0.25) and bigrams (0.5; stop-word/stop-word pairs skipped), optionally
  character trigrams (`char_ngrams=True`, model id `…+char3`), L2-normalised. Texts that
  share words get a high cosine ("rain on a tin roof" vs "tin roof in the rain" ≈ 0.84).
  Texts with no words in common are near-orthogonal. It is **lexical only**: "car" and
  "automobile" are unrelated.
* **Bytes / URIs**: a pseudo-random unit vector seeded by the content hash. Identical
  content gives an identical vector. Different content gives a cosine of about
  N(0, 1/dim). A readable `file://` URI is hashed by the file's content, so bytes and a path
  to the same file agree. Any other URI is hashed by its string.

### time (`channels/time.py`)

The encoder computes three angles in the timestamp's **own** UTC offset: hour-of-day
(including minutes), weekday (continuous, so Sunday 23:59 is close to Monday 00:01) and
day-of-year. Each angle becomes a `(sin, cos)` pair scaled by `√w`, with weights hour
0.6, weekday 0.15 and season 0.25. Similarity recovers each angle difference from its pair
and takes the weighted mean of `(1 + cos Δθ)/2`. Because the weights sum to 1, this equals
`1 − |a − b|² / 4`, so L2 KNN ranks exactly like the similarity. Some reference values:
21:00 vs 21:30 the same day ≈ 0.997; 21:00 vs 09:00 ≈ 0.39; the same time six months
apart ≈ 0.75. `make_time_channel(weights={...})` overrides the weights.

The adapter accepts a `datetime` (naive is treated as UTC), an ISO-8601 string (parsed by
`core.serde.parse_timestamp`), or epoch seconds. A `None` value uses `RawSignal.observed_at`.
Send local-offset timestamps (`2026-06-01T21:30:00+02:00`).

### geo (`channels/geo.py`)

The adapter accepts a geohash string (precision kept), `(lat, lon)` as a tuple or list,
or `{"lat", "lon"}`. Coordinates are encoded at precision 9 and kept in `numbers`.
The embedder uses the exact lat/lon when present, otherwise the cell centre, and maps it
to `(cos φ cos λ, cos φ sin λ, sin φ)`. Similarity is `exp(-d_km / scale_km)` (default
1 km), with `d` the great-circle distance recovered from the chord `|a − b|`. Points 500 m
apart score ≈ 0.61 and points 10 km apart ≈ 0. The chord grows monotonically with
distance, so sqlite-vec L2 KNN is an exact pre-filter. Use `make_geo_channel(scale_km=…)`
for a city-scale or neighbourhood-scale sense of place.

### biometric (`channels/biometric.py`)

| feature | unit | reference range |
|---|---|---|
| `heart_rate` | bpm | 40–180 |
| `hrv` | ms (RMSSD) | 0–200 |
| `skin_temp` | °C | 28–38 |
| `eda` | µS | 0–20 |
| `steps_per_min` | steps/min | 0–200 |
| `respiration_rate` | breaths/min | 6–40 |

Values are clamped to their range and scaled to `[0, 1]`. The vector holds
`[value_norm · present, present]` per feature (dim 12, not L2-normalised). Similarity is
the mean of `exp(-(Δ/σ)²/2)` with σ = 0.15, taken over the features **present in both**
readings, so a missing feature never penalises a match. Readings with no feature in common
score 0. The empty reading is not a valid embedding: the adapter and encoder reject it,
and a similarity involving it is 0. Unknown feature names raise `ObservationError`, and the
message lists the allowed names.

### image (`channels/image.py`)

The adapter accepts bytes, using `mime_type` if given or sniffing PNG/JPEG/GIF/WebP/BMP/TIFF
from the data. It also accepts a local file as `uri=` or as a string value (`file://`
or a path). The adapter checks that the file exists but does not read it. **Remote URLs
(`http(s)://`, `s3://`, …) are rejected**, because madeleine never downloads media.
Payloads are limited to 32 MiB.
`OpenClipEmbedder(model="ViT-B-32", pretrained="laion2b_s34b_b79k")` returns L2-normalised
image features. `config.image_model` has the form `"<model>/<pretrained>"`.

**Rescaling.** CLIP image-image cosines of unrelated images are well above zero. In our
check with ViT-B-32, flat colours, stripes, gradients and noise scored 0.54–0.67 against
each other, and near-duplicates scored ≥ 0.99. The default similarity therefore maps cosine
0.6 → 0 and 0.95 → 1, clamped. These constants (`IMAGE_COSINE_LO/HI`) are heuristics and
should be tuned with the evals on real photos.

### audio (`channels/audio.py`)

The adapter accepts encoded bytes (WAV/FLAC/OGG/MP3/AIFF sniffed, or any `audio/*` mime
that libsndfile decodes), raw 16-bit PCM, or a local file. For raw PCM, give
`mime_type="audio/L16; rate=16000"` (big-endian, RFC 2586) or `"audio/pcm; rate=16000"`
(little-endian), optionally with `channels=N`. Remote URLs are rejected.
`ClapEmbedder("laion/clap-htsat-unfused")` processes audio in four steps:

1. It mixes the clip down to mono.
2. It **resamples to 48 kHz**, the CLAP training rate. The resampler is numpy-only: a box
   filter for anti-aliasing, then linear interpolation.
3. It **keeps only the first 10 s**. Otherwise the processor's random crop would make
   embeddings non-deterministic.
4. It returns L2-normalised projected audio features.

**Rescaling.** In our check, pure tones vs white noise scored 0.28–0.42, and tones of
different pitch scored 0.83–0.97. The default similarity maps cosine 0.4 → 0 and 0.95 → 1.
This is a heuristic.

### smell (`channels/smell.py`)

`SmellAdapter.normalize` always raises `ChannelNotAvailableError`. The smell channel is
not registered by default. The module docstring sketches a future e-nose channel: VOC
sensor arrays become baseline-corrected, log-scaled `numbers`, and the channel compares
normalised response patterns by cosine, with `solo_trigger=True`.

## Writing a custom channel

Any object with the right attributes satisfies the protocols. No base class is needed.

```python
from collections.abc import Sequence
from madeleine.channels import DefaultChannelRegistry, cosine01, hash_registry
from madeleine.core.types import Channel, ChannelSpec, Observation, ObservationError, RawSignal

class WeatherAdapter:
    channel = "weather"
    def normalize(self, signal: RawSignal) -> Observation:
        if not isinstance(signal.value, dict):
            raise ObservationError("weather needs {'temp_c', 'rain_mm'}")
        return Observation("weather", numbers={k: float(v) for k, v in signal.value.items()})

class WeatherEmbedder:
    channel, dim, model_id = "weather", 2, None
    def embed(self, o: Observation) -> tuple[float, ...]:
        return (o.numbers.get("temp_c", 15.0) / 40, min(o.numbers.get("rain_mm", 0.0), 20) / 20)
    def embed_many(self, obs: Sequence[Observation]) -> list[tuple[float, ...]]:
        return [self.embed(o) for o in obs]

def weather_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return max(0.0, 1.0 - (abs(a[0] - b[0]) + abs(a[1] - b[1])) / 2)

registry = hash_registry()
registry.register(Channel(
    spec=ChannelSpec("weather", dim=2, default_weight=0.5, solo_trigger=False,
                     metric="1 - mean |Δ|"),
    adapter=WeatherAdapter(), embedder=WeatherEmbedder(), similarity=weather_similarity,
))
# then: create_engine(config, registry=registry)
```

Rules your channel must follow:

* The similarity must return a value in `[0, 1]`, be symmetric and pure, and give
  `sim(v, v) == 1` for any valid vector.
* `embed` must be deterministic and return finite values, and `embed_many` must equal
  `[embed(o) for o in obs]`.
* Load heavy models lazily inside `embed`, never at import or in `__init__`.
* Set `model_id` whenever the vectors depend on a model, so that stores can refuse to mix
  vectors from different models.
* A channel that is only context (it matches too many memories to cause recall on its
  own) should set `solo_trigger=False`.
* Never log raw media.

The generic contract test in `tests/channels/test_ch_contract.py` is a good template for
testing your own channel.

## ImageBind (not included)

[ImageBind](https://github.com/facebookresearch/ImageBind) embeds images, audio, text,
depth, thermal and IMU data in **one** space. That would allow cross-modal cues, such as a
sound recalling a photo. Its weights are licensed **CC-BY-NC 4.0** (non-commercial), which
is incompatible with distributing madeleine under Apache-2.0. ImageBind is therefore
**never a dependency**. To use it privately, install it yourself, write an `Embedder` per
modality that calls the ImageBind model with its modality key, and register each one under
a channel name such as `image`, `audio` or `text` with
`registry.register(channel, replace=True)`. Use `model_id="imagebind_huge"`. For this to
work cross-modally the engine would have to compare vectors *across* channels, which v0
does not do: fusion only compares cue and memory on the *same* channel. Even same-channel,
ImageBind is a drop-in for CLIP/CLAP.
