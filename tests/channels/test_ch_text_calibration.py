from __future__ import annotations

from madeleine.channels.registry import hash_registry
from madeleine.core.types import TEXT, Observation


def _sim(a: str, b: str) -> float:
    ch = hash_registry().get(TEXT)
    va = ch.embedder.embed(Observation(TEXT, text=a))
    vb = ch.embedder.embed(Observation(TEXT, text=b))
    return ch.similarity(va, vb)


def test_hash_text_similarity_is_calibrated() -> None:
    assert _sim("lime-blossom tea in the kitchen", "lime-blossom tea in the kitchen") == 1.0
    assert _sim("rain on a tin roof", "stock market quarterly report") == 0.0
    assert _sim("tea and a madeleine in grandmother's kitchen", "grandmother's tea") > 0.5
    assert "rescaled from [0.1, 0.45]" in hash_registry().get(TEXT).spec.metric
