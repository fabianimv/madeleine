"""Tests for similarity helpers and the HashEmbedder."""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from madeleine.channels.hashing import HashEmbedder
from madeleine.channels.similarity import cosine, cosine01, l2_normalize, rescaled_cosine
from madeleine.core.types import TEXT, Observation

finite = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
vectors = st.lists(finite, min_size=3, max_size=3).filter(lambda v: math.hypot(*v) > 1e-3)


def test_l2_normalize_unit_and_zero_safe() -> None:
    v = l2_normalize([3.0, 4.0])
    assert v == pytest.approx((0.6, 0.8))
    assert l2_normalize([0.0, 0.0]) == (0.0, 0.0)


def test_cosine_basics() -> None:
    assert cosine((1.0, 0.0), (0.0, 1.0)) == 0.0
    assert cosine((1.0, 0.0), (-1.0, 0.0)) == -1.0
    assert cosine01((1.0, 0.0), (-1.0, 0.0)) == 0.0
    assert cosine01((0.0, 0.0), (1.0, 0.0)) == 0.0
    with pytest.raises(ValueError, match="length mismatch"):
        cosine((1.0,), (1.0, 0.0))


@given(vectors, vectors)
def test_cosine01_contract(a: list[float], b: list[float]) -> None:
    ta, tb = tuple(a), tuple(b)
    s = cosine01(ta, tb)
    assert 0.0 <= s <= 1.0
    assert s == cosine01(tb, ta)
    assert cosine01(ta, ta) == pytest.approx(1.0, abs=1e-6)


def test_rescaled_cosine() -> None:
    sim = rescaled_cosine(0.5, 1.0)
    assert sim((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert sim((1.0, 0.0), (0.0, 1.0)) == 0.0
    c = math.cos(math.radians(41.4096221))  # cos = 0.75
    assert sim((1.0, 0.0), (c, math.sqrt(1 - c * c))) == pytest.approx(0.5, abs=1e-6)
    with pytest.raises(ValueError, match="lo < hi"):
        rescaled_cosine(0.9, 0.5)
    with pytest.raises(ValueError, match="lo < hi"):
        rescaled_cosine(0.0, 1.5)


def test_hash_text_lexical_similarity() -> None:
    emb = HashEmbedder(TEXT)
    a = emb.embed(Observation(TEXT, text="rain on a tin roof"))
    b = emb.embed(Observation(TEXT, text="tin roof in the rain"))
    c = emb.embed(Observation(TEXT, text="stock market report"))
    assert cosine01(a, b) > 0.5
    assert cosine01(a, b) > cosine01(a, c) + 0.3
    assert emb.model_id == "hash-v1/dim=256"


def test_hash_text_never_zero_and_case_insensitive() -> None:
    emb = HashEmbedder(TEXT, dim=32)
    v = emb.embed(Observation(TEXT, text="!!!"))
    assert math.isclose(math.fsum(x * x for x in v), 1.0)
    assert emb.embed(Observation(TEXT, text="Hello World")) == emb.embed(
        Observation(TEXT, text="hello world")
    )


def test_hash_content_vectors() -> None:
    emb = HashEmbedder("image", dim=256)
    a = emb.embed(Observation("image", data=b"abc", mime_type="image/png"))
    assert a == emb.embed(Observation("image", data=b"abc"))
    b = emb.embed(Observation("image", data=b"abd"))
    assert abs(cosine(a, b)) < 0.3


def test_hash_char_ngrams_model_id() -> None:
    emb = HashEmbedder(TEXT, char_ngrams=True)
    assert emb.model_id == "hash-v1/dim=256+char3"
    a = emb.embed(Observation(TEXT, text="walking"))
    b = emb.embed(Observation(TEXT, text="walked"))
    plain = HashEmbedder(TEXT)
    pa = plain.embed(Observation(TEXT, text="walking"))
    pb = plain.embed(Observation(TEXT, text="walked"))
    assert cosine01(a, b) > 0.1
    assert cosine01(a, b) > cosine01(pa, pb)
