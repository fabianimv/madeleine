"""Senses: one module per channel (adapter -> embedder -> similarity).

Importing this package is cheap: heavy ML libraries are imported lazily on first use.
"""

from madeleine.channels.hashing import HashEmbedder
from madeleine.channels.similarity import cosine01, l2_normalize, rescaled_cosine

__all__ = ["HashEmbedder", "cosine01", "l2_normalize", "rescaled_cosine"]
