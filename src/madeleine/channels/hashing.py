"""``HashEmbedder``: a deterministic, offline embedder for any channel (tests, demo, CI).

No model, no download, no numpy: vectors come from BLAKE2b (never Python's ``hash()``,
which is salted per process), so they are identical across processes and machines.

* **Text** (``Observation.text``): signed feature hashing of lower-cased word unigrams
  (stop words down-weighted) and word bigrams, optionally character trigrams, then
  L2-normalised. Texts that share words get a high cosine; texts that share none are
  near-orthogonal. It captures *lexical* overlap only ("car" and "automobile" are
  unrelated), which is enough for the offline demo and the synthetic benchmark.
* **Media** (``data`` bytes, or a ``uri``): a pseudo-random unit vector seeded by the
  content hash. Identical content -> identical vector; different content -> near-orthogonal
  (cosine ~ N(0, 1/dim)). A ``uri`` is hashed by its **string**: the hash embedder never
  reads files, so it cannot be used to read or fingerprint a file the adapter did not
  authorise. (Bytes and a path to the same file therefore get different vectors; the
  adapters normalise paths to one absolute ``file://`` URI, so two references to the
  same file agree.)
* **Anything else** (``numbers``/``timestamp``): content vector of a canonical encoding.

``model_id`` is ``"hash-v1/dim=<dim>"`` (``+char3`` with character trigrams). Pair it with
:func:`~madeleine.channels.similarity.cosine01`.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
import unicodedata
from collections.abc import Sequence
from typing import Final

from madeleine.channels.similarity import l2_norm, l2_normalize
from madeleine.core.types import TEXT, ChannelName, Observation, ObservationError, Vector

__all__ = ["HASH_VERSION", "HashEmbedder"]

HASH_VERSION: Final = "hash-v1"

_PERSON: Final = b"madeleine-hash1"  # BLAKE2b personalisation (<= 16 bytes)
_TOKEN_RE: Final = re.compile(r"\w+")

UNIGRAM_WEIGHT: Final = 1.0
STOPWORD_WEIGHT: Final = 0.25
BIGRAM_WEIGHT: Final = 0.5
CHAR_TRIGRAM_WEIGHT: Final = 0.25

# Small English stop-word list: these words still count, but at STOPWORD_WEIGHT, so that
# unrelated sentences do not look similar merely because both contain "the" and "of".
_STOPWORDS: Final = frozenset(
    re.split(
        r"\s+",
        """a an and are as at be been but by for from had has have he her his i if in into is it
        its me my no not of on or our she so that the their them then there they this to was we
        were what when which who will with you your""",
    )
)


def _digest(data: bytes, size: int = 8) -> bytes:
    return hashlib.blake2b(data, digest_size=size, person=_PERSON).digest()


def tokenize(text: str) -> list[str]:
    """Lower-cased (NFKC + casefold) word tokens."""
    return _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).casefold())


class HashEmbedder:
    """Deterministic feature-hashing embedder usable for any channel.

    Args:
        channel: the channel it embeds for (``text``, ``image``, ``audio``...).
        dim: vector length (default 256).
        char_ngrams: also hash character trigrams of each word (more robust to typos
            and inflections, slightly less discriminative). Changes ``model_id``.
    """

    def __init__(
        self, channel: ChannelName = TEXT, dim: int = 256, *, char_ngrams: bool = False
    ) -> None:
        if dim < 2:
            raise ValueError("HashEmbedder dim must be >= 2")
        self._channel = channel
        self._dim = dim
        self._char_ngrams = char_ngrams

    def __repr__(self) -> str:
        return f"HashEmbedder(channel={self._channel!r}, dim={self._dim})"

    @property
    def channel(self) -> ChannelName:
        return self._channel

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        suffix = "+char3" if self._char_ngrams else ""
        return f"{HASH_VERSION}/dim={self._dim}{suffix}"

    # -- public ------------------------------------------------------------------------
    def embed(self, observation: Observation) -> Vector:
        if observation.channel != self._channel:
            raise ObservationError(
                f"HashEmbedder for {self._channel!r} got an observation for {observation.channel!r}"
            )
        if observation.data is not None:
            return self.content_vector(b"data:" + observation.data)
        if observation.text is not None:
            return self.text_vector(observation.text)
        if observation.uri is not None:
            return self.content_vector(self._uri_payload(observation))
        if observation.numbers or observation.timestamp is not None:
            canonical = {
                "numbers": {k: float(v) for k, v in sorted(observation.numbers.items())},
                "timestamp": observation.timestamp.isoformat() if observation.timestamp else None,
            }
            return self.content_vector(b"obs:" + json.dumps(canonical).encode())
        raise ObservationError(f"{self._channel}: empty observation, nothing to embed")

    def embed_many(self, observations: Sequence[Observation]) -> list[Vector]:
        return [self.embed(o) for o in observations]

    # -- building blocks (public for tests and custom channels) --------------------------
    def text_vector(self, text: str) -> Vector:
        """Signed feature-hashing vector of ``text`` (L2-normalised, never zero)."""
        acc = [0.0] * self._dim
        for feature, weight in self._features(text):
            h = int.from_bytes(_digest(feature.encode("utf-8")), "little")
            index = h % self._dim
            sign = 1.0 if (h >> 63) & 1 else -1.0
            acc[index] += sign * weight
        if l2_norm(acc) == 0.0:
            # No word tokens (e.g. only punctuation/emoji) or exact cancellation.
            return self.content_vector(b"text:" + text.encode("utf-8"))
        return l2_normalize(acc)

    def content_vector(self, payload: bytes) -> Vector:
        """Pseudo-random unit vector seeded by ``payload`` (identical in -> identical out)."""
        seed = hashlib.blake2b(payload, digest_size=32, person=_PERSON).digest()
        values: list[float] = []
        block = 0
        while len(values) < self._dim:
            chunk = _digest(seed + block.to_bytes(4, "little"), 64)
            for i in range(0, 64, 2):
                values.append(int.from_bytes(chunk[i : i + 2], "little") / 32767.5 - 1.0)
            block += 1
        vec = l2_normalize(values[: self._dim])
        return vec if l2_norm(vec) > 0 else (1.0,) + (0.0,) * (self._dim - 1)

    # -- internals -----------------------------------------------------------------------
    def _features(self, text: str) -> list[tuple[str, float]]:
        tokens = tokenize(text)
        feats: list[tuple[str, float]] = []
        for tok in tokens:
            feats.append(("u:" + tok, STOPWORD_WEIGHT if tok in _STOPWORDS else UNIGRAM_WEIGHT))
        for left, right in itertools.pairwise(tokens):
            if left in _STOPWORDS and right in _STOPWORDS:
                continue
            feats.append((f"b:{left} {right}", BIGRAM_WEIGHT))
        if self._char_ngrams:
            for tok in tokens:
                padded = f"#{tok}#"
                for i in range(len(padded) - 2):
                    feats.append(("c:" + padded[i : i + 3], CHAR_TRIGRAM_WEIGHT))
        return feats

    @staticmethod
    def _uri_payload(observation: Observation) -> bytes:
        # Never read the file (see the module docstring): hash the reference itself.
        return b"uri:" + (observation.uri or "").encode("utf-8")
