"""Channel registries: :class:`DefaultChannelRegistry`, :func:`default_registry`,
:func:`hash_registry`.

* :func:`default_registry` builds the senses listed in ``config.channels`` with their real
  backends. Optional channels whose pip extra is missing (``image``, ``audio``) are skipped
  with a logged warning - availability is checked with ``importlib.util.find_spec``, so
  nothing heavy is imported. ``text`` always registers (it falls back to the hash
  embedder under ``text_backend="auto"``). ``smell`` is skipped with a warning (future
  work). Unknown names raise :class:`~madeleine.core.types.ChannelNotAvailableError`.
* :func:`hash_registry` backs every ML channel with
  :class:`~madeleine.channels.hashing.HashEmbedder` and uses the real deterministic
  encoders for time/geo/biometric. It never downloads anything (tests, demo, evals).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable, Iterator

from madeleine.channels._extras import install_hint, missing_modules
from madeleine.channels.audio import make_audio_channel
from madeleine.channels.biometric import make_biometric_channel
from madeleine.channels.geo import make_geo_channel
from madeleine.channels.hashing import HashEmbedder
from madeleine.channels.image import make_image_channel
from madeleine.channels.smell import SMELL_UNAVAILABLE
from madeleine.channels.text import make_text_channel
from madeleine.channels.time import make_time_channel
from madeleine.core.types import (
    AUDIO,
    BIOMETRIC,
    GEO,
    IMAGE,
    SMELL,
    TEXT,
    TIME,
    Channel,
    ChannelName,
    ChannelNotAvailableError,
    ChannelSpec,
    MadeleineConfig,
)

__all__ = [
    "BUILTIN_CHANNELS",
    "DefaultChannelRegistry",
    "default_registry",
    "hash_registry",
]

_log = logging.getLogger(__name__)

BUILTIN_CHANNELS: tuple[ChannelName, ...] = (TEXT, TIME, GEO, BIOMETRIC, IMAGE, AUDIO)
"""Channels madeleine knows how to build (``smell`` is reserved, not buildable)."""


class DefaultChannelRegistry:
    """Thread-safe, in-memory :class:`~madeleine.core.types.ChannelRegistry`."""

    def __init__(self, channels: Iterable[Channel] = ()) -> None:
        self._channels: dict[ChannelName, Channel] = {}
        self._lock = threading.Lock()
        for channel in channels:
            self.register(channel)

    def register(self, channel: Channel, *, replace: bool = False) -> None:
        """Add ``channel``; ``ValueError`` if its name exists and ``replace`` is False."""
        if not isinstance(channel, Channel):
            raise TypeError(f"expected a Channel, got {type(channel).__name__}")
        with self._lock:
            if channel.name in self._channels and not replace:
                raise ValueError(f"channel {channel.name!r} is already registered")
            self._channels[channel.name] = channel

    def unregister(self, name: ChannelName) -> bool:
        """Remove a channel; returns whether it was registered."""
        with self._lock:
            return self._channels.pop(name, None) is not None

    def get(self, name: ChannelName) -> Channel:
        with self._lock:
            channel = self._channels.get(name)
            known = sorted(self._channels)
        if channel is None:
            if name == SMELL:
                raise ChannelNotAvailableError(SMELL_UNAVAILABLE)
            hint = ""
            if name == IMAGE:
                hint = f"; {install_hint(IMAGE, 'image')}"
            elif name == AUDIO:
                hint = f"; {install_hint(AUDIO, 'audio')}"
            raise ChannelNotAvailableError(
                f"channel {name!r} is not available (registered: {', '.join(known) or 'none'})"
                f"{hint}"
            )
        return channel

    def names(self) -> tuple[ChannelName, ...]:
        with self._lock:
            return tuple(sorted(self._channels))

    def specs(self) -> dict[ChannelName, ChannelSpec]:
        """Registered channel specs by name."""
        with self._lock:
            return {name: ch.spec for name, ch in sorted(self._channels.items())}

    def __contains__(self, name: object) -> bool:
        with self._lock:
            return name in self._channels

    def __iter__(self) -> Iterator[Channel]:
        with self._lock:
            channels = [self._channels[n] for n in sorted(self._channels)]
        return iter(channels)

    def __len__(self) -> int:
        with self._lock:
            return len(self._channels)

    def __repr__(self) -> str:
        return f"DefaultChannelRegistry({list(self.names())})"


def _optional(extra: str, name: ChannelName, build: Callable[[], Channel]) -> Channel | None:
    missing = missing_modules(extra)
    if missing:
        _log.warning(
            "skipping channel %r: missing %s; %s",
            name,
            ", ".join(missing),
            install_hint(name, extra),
        )
        return None
    return build()


def _unknown(name: ChannelName) -> ChannelNotAvailableError:
    return ChannelNotAvailableError(
        f"unknown channel {name!r}; built-in channels: {', '.join(BUILTIN_CHANNELS)}"
    )


def default_registry(config: MadeleineConfig | None = None) -> DefaultChannelRegistry:
    """Registry of ``config.channels`` with real backends (models load lazily on first use).

    The ``image``/``audio`` adapters get ``config.allow_local_files`` (default False: media
    must be sent as bytes).
    """
    cfg = config or MadeleineConfig()
    registry = DefaultChannelRegistry()
    for name in dict.fromkeys(cfg.channels):
        channel: Channel | None
        if name == TEXT:
            channel = make_text_channel(cfg)
        elif name == TIME:
            channel = make_time_channel()
        elif name == GEO:
            channel = make_geo_channel()
        elif name == BIOMETRIC:
            channel = make_biometric_channel()
        elif name == IMAGE:
            channel = _optional(
                "image",
                IMAGE,
                lambda: make_image_channel(cfg, allow_local_files=cfg.allow_local_files),
            )
        elif name == AUDIO:
            channel = _optional(
                "audio",
                AUDIO,
                lambda: make_audio_channel(cfg, allow_local_files=cfg.allow_local_files),
            )
        elif name == SMELL:
            _log.warning("skipping channel 'smell': %s", SMELL_UNAVAILABLE)
            channel = None
        else:
            raise _unknown(name)
        if channel is not None:
            registry.register(channel)
    return registry


def hash_registry(
    channels: Iterable[ChannelName] = BUILTIN_CHANNELS,
    dim: int = 256,
    *,
    allow_local_files: bool = False,
) -> DefaultChannelRegistry:
    """Offline registry that never downloads anything (tests, demo, evals).

    * ``text`` uses ``HashEmbedder(dim)`` with
      :data:`~madeleine.channels.text.hash_text_similarity` (``rescaled_cosine(0.10, 0.45)``,
      the lexical hash calibration).
    * ``image``/``audio`` use ``HashEmbedder(dim)`` with ``cosine01``; their adapters get
      ``allow_local_files`` (default False).
    * ``time``/``geo``/``biometric`` use their real deterministic encoders.
    """
    registry = DefaultChannelRegistry()
    for name in dict.fromkeys(channels):
        if name == TEXT:
            channel = make_text_channel(embedder=HashEmbedder(TEXT, dim))
        elif name == IMAGE:
            channel = make_image_channel(
                embedder=HashEmbedder(IMAGE, dim), allow_local_files=allow_local_files
            )
        elif name == AUDIO:
            channel = make_audio_channel(
                embedder=HashEmbedder(AUDIO, dim), allow_local_files=allow_local_files
            )
        elif name == TIME:
            channel = make_time_channel()
        elif name == GEO:
            channel = make_geo_channel()
        elif name == BIOMETRIC:
            channel = make_biometric_channel()
        elif name == SMELL:
            raise ChannelNotAvailableError(SMELL_UNAVAILABLE)
        else:
            raise _unknown(name)
        registry.register(channel)
    return registry
