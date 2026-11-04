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


def test_default_sentence_transformer_model_is_calibrated_without_loading() -> None:
    from madeleine.channels.text import (
        MINILM_TEXT_COSINE_RANGE,
        SentenceTransformerEmbedder,
        make_text_channel,
    )

    ch = make_text_channel(
        embedder=SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    )
    lo, hi = MINILM_TEXT_COSINE_RANGE
    assert f"rescaled from [{lo:g}, {hi:g}]" in ch.spec.metric
    other = make_text_channel(embedder=SentenceTransformerEmbedder("some/other-model", dim=768))
    assert other.spec.metric == "max(0, cosine)"


def test_bench_cli_reports_missing_text_extra(monkeypatch: object) -> None:
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is not None:
        import pytest

        pytest.skip("sentence-transformers is installed")
    from madeleine.evals.cli import main

    assert main(["--memories", "5", "--cues", "3", "--text-backend", "sentence-transformers"]) == 1
