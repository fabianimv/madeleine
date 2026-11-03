"""Senses: one module per channel (adapter -> embedder -> similarity).

Importing this package (or any of its modules) is cheap: torch, open_clip, transformers and
sentence-transformers are imported lazily on the first ``embed`` call of the embedder that
needs them. See ``docs/channels.md``.
"""

from madeleine.channels.audio import AudioAdapter, ClapEmbedder, make_audio_channel
from madeleine.channels.biometric import BiometricAdapter, BiometricEncoder, make_biometric_channel
from madeleine.channels.geo import GeoAdapter, GeoEmbedder, make_geo_channel
from madeleine.channels.hashing import HashEmbedder
from madeleine.channels.image import ImageAdapter, OpenClipEmbedder, make_image_channel
from madeleine.channels.registry import (
    BUILTIN_CHANNELS,
    DefaultChannelRegistry,
    default_registry,
    hash_registry,
)
from madeleine.channels.similarity import cosine01, l2_normalize, rescaled_cosine
from madeleine.channels.smell import SmellAdapter
from madeleine.channels.text import SentenceTransformerEmbedder, TextAdapter, make_text_channel
from madeleine.channels.time import TimeAdapter, TimeEncoder, make_time_channel

__all__ = [
    "BUILTIN_CHANNELS",
    "AudioAdapter",
    "BiometricAdapter",
    "BiometricEncoder",
    "ClapEmbedder",
    "DefaultChannelRegistry",
    "GeoAdapter",
    "GeoEmbedder",
    "HashEmbedder",
    "ImageAdapter",
    "OpenClipEmbedder",
    "SentenceTransformerEmbedder",
    "SmellAdapter",
    "TextAdapter",
    "TimeAdapter",
    "TimeEncoder",
    "cosine01",
    "default_registry",
    "hash_registry",
    "l2_normalize",
    "make_audio_channel",
    "make_biometric_channel",
    "make_geo_channel",
    "make_image_channel",
    "make_text_channel",
    "make_time_channel",
    "rescaled_cosine",
]
