"""Private helpers for media channels (image, audio): URIs, MIME sniffing, local reads.

Policy:

* madeleine never fetches remote media. ``http(s)://`` and other non-file schemes are
  rejected with :class:`~madeleine.core.types.ObservationError`.
* Local files (``file://`` URIs, bare paths, Windows drive paths) are a **server-side
  capability, off by default** (``MadeleineConfig.allow_local_files``). The media adapters
  enforce it themselves (:func:`normalize_file_uri`, :func:`check_uri_label`), so no
  front-end (MCP, HTTP, webhooks) can bypass it. The refusal is decided *before* any
  filesystem access and its message never depends on whether the file exists, so it is not
  a file-existence oracle.
* Files are capped at :data:`MAX_MEDIA_BYTES`, like in-memory payloads.
"""

from __future__ import annotations

import mimetypes
import re
from pathlib import Path
from typing import Final
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from madeleine.core.types import ChannelName, Observation, ObservationError

MAX_MEDIA_BYTES: Final = 32 * 1024 * 1024
"""Largest accepted media payload, in memory or on disk (32 MiB)."""

_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")

_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
)

_AUDIO_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"fLaC", "audio/flac"),
    (b"OggS", "audio/ogg"),
    (b"ID3", "audio/mpeg"),
)


def sniff_image_mime(data: bytes) -> str | None:
    """Best-effort MIME type of image bytes from their magic number."""
    for magic, mime in _IMAGE_MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def sniff_audio_mime(data: bytes) -> str | None:
    """Best-effort MIME type of audio bytes from their magic number."""
    for magic, mime in _AUDIO_MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data[:4] == b"FORM" and data[8:12] in (b"AIFF", b"AIFC"):
        return "audio/aiff"
    if len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0:
        return "audio/mpeg"
    return None


def base_mime(mime_type: str | None) -> str | None:
    """``"audio/L16; rate=16000"`` -> ``"audio/l16"``."""
    if mime_type is None:
        return None
    return mime_type.split(";", 1)[0].strip().lower() or None


def mime_params(mime_type: str | None) -> dict[str, str]:
    """``"audio/L16; rate=16000; channels=1"`` -> ``{"rate": "16000", "channels": "1"}``."""
    if not mime_type or ";" not in mime_type:
        return {}
    out: dict[str, str] = {}
    for part in mime_type.split(";")[1:]:
        key, sep, value = part.partition("=")
        if sep:
            out[key.strip().lower()] = value.strip().strip('"')
    return out


def local_path(uri: str, channel: ChannelName) -> Path:
    """Resolve a ``file://`` URI or a plain path to a :class:`Path` (no I/O).

    Raises :class:`ObservationError` for remote schemes: madeleine does not fetch URLs.
    """
    text = uri.strip()
    if not text:
        raise ObservationError(f"{channel}: empty uri")
    if _WINDOWS_DRIVE.match(text):
        return Path(text)
    if "://" in text or text.startswith("file:"):
        parsed = urlparse(text)
        if parsed.scheme.lower() != "file":
            raise ObservationError(
                f"{channel}: only file:// URIs or local paths are supported; refusing to fetch "
                f"{parsed.scheme}:// (madeleine never downloads media)"
            )
        if parsed.netloc not in ("", "localhost"):
            raise ObservationError(f"{channel}: remote file host {parsed.netloc!r} not supported")
        return Path(url2pathname(unquote(parsed.path)))
    return Path(text).expanduser()


def local_files_disabled(channel: ChannelName) -> ObservationError:
    """The refusal for a local file reference when local files are disabled.

    Constant per channel: it never echoes the path nor depends on the file existing.
    """
    return ObservationError(
        f"{channel}: local file paths / file:// URIs are disabled on this server; send the "
        "media bytes instead (data, or data_b64 over MCP/HTTP). The server operator can "
        "enable local files with 'serve --allow-file-uris' or MADELEINE_ALLOW_LOCAL_FILES=1"
    )


def check_uri_label(uri: str | None, channel: ChannelName, *, allow_local_files: bool) -> None:
    """Policy check for a ``uri`` sent *alongside bytes* (a label, never read).

    A remote URI is kept as a plain label. A local path is still a reference to the
    server's filesystem (it would be persisted with the memory), so it obeys the policy.
    """
    if uri is None or allow_local_files:
        return
    try:
        local_path(uri, channel)
    except ObservationError:
        return  # remote or malformed: not a local file reference
    raise local_files_disabled(channel)


def _check_size(path: Path, channel: ChannelName, max_bytes: int) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ObservationError(f"{channel}: cannot access {str(path)!r}") from exc
    if size > max_bytes:
        raise ObservationError(f"{channel}: file is larger than {max_bytes} bytes")


def normalize_file_uri(
    uri: str, channel: ChannelName, *, allow_local_files: bool, max_bytes: int = MAX_MEDIA_BYTES
) -> tuple[str, Path]:
    """Validate a local media reference; return ``(file:// uri, absolute path)``.

    Order matters: remote schemes are rejected, then the local-file policy is applied
    **before any filesystem access** (no existence oracle), then the file must exist and
    be at most ``max_bytes``.
    """
    path = local_path(uri, channel)
    if not allow_local_files:
        raise local_files_disabled(channel)
    try:
        resolved = path.resolve()
        is_file = resolved.is_file()
    except OSError as exc:  # pragma: no cover - exotic filesystems
        raise ObservationError(f"{channel}: cannot access {uri!r}: {exc}") from exc
    if not is_file:
        raise ObservationError(f"{channel}: file not found: {uri!r}")
    _check_size(resolved, channel, max_bytes)
    return resolved.as_uri(), resolved


def checked_local_file(uri: str, channel: ChannelName, *, max_bytes: int = MAX_MEDIA_BYTES) -> Path:
    """Path of an observation's ``uri`` for an embedder to read, refusing oversized files.

    Embedders only see uri-only observations that an adapter authorised (the engine always
    normalises signals through the channel's adapter).
    """
    path = local_path(uri, channel)
    _check_size(path, channel, max_bytes)
    return path


def guess_mime_from_path(path: Path) -> str | None:
    mime, _ = mimetypes.guess_type(path.name)
    return mime


def read_media_bytes(observation: Observation) -> bytes:
    """Bytes of a media observation: ``data`` if present, else the local file at ``uri``."""
    if observation.data is not None:
        return observation.data
    if observation.uri is None:
        raise ObservationError(f"{observation.channel}: observation has neither data nor uri")
    path = checked_local_file(observation.uri, observation.channel)
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ObservationError(f"{observation.channel}: cannot read {observation.uri!r}") from exc


def coerce_bytes(value: object) -> bytes | None:
    """Return ``value`` as ``bytes`` if it is bytes-like, else ``None``."""
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray | memoryview):
        return bytes(value)
    return None
