"""Registries and backend selection (offline: missing extras are simulated)."""

from __future__ import annotations

import importlib.util
import logging
import sys
from importlib.machinery import ModuleSpec

import pytest

from madeleine.channels import text as text_module
from madeleine.channels.hashing import HashEmbedder
from madeleine.channels.registry import DefaultChannelRegistry, default_registry, hash_registry
from madeleine.channels.text import SentenceTransformerEmbedder, make_text_channel
from madeleine.core.types import (
    AUDIO,
    BIOMETRIC,
    GEO,
    IMAGE,
    SMELL,
    TEXT,
    TIME,
    ChannelNotAvailableError,
    ChannelRegistry,
    MadeleineConfig,
    Observation,
)

_real_find_spec = importlib.util.find_spec


def _hide(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    def fake(name: str, package: str | None = None) -> ModuleSpec | None:
        if name.split(".")[0] in names:
            return None
        return _real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


def _pretend_installed(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    def fake(name: str, package: str | None = None) -> ModuleSpec | None:
        if name.split(".")[0] in names:
            return ModuleSpec(name, None)
        return _real_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


@pytest.fixture
def no_st(monkeypatch: pytest.MonkeyPatch) -> None:
    """sentence-transformers is neither findable nor importable."""
    _hide(monkeypatch, "sentence_transformers")
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setattr(text_module, "_auto_fallback_warned", False)


def test_register_get_names_contains() -> None:
    reg = DefaultChannelRegistry()
    assert isinstance(reg, ChannelRegistry)
    ch = hash_registry((GEO,)).get(GEO)
    reg.register(ch)
    assert reg.names() == (GEO,)
    assert GEO in reg
    assert len(reg) == 1
    assert reg.get(GEO) is ch
    with pytest.raises(ValueError, match="already registered"):
        reg.register(ch)
    reg.register(ch, replace=True)
    assert list(reg) == [ch]
    assert reg.specs() == {GEO: ch.spec}
    with pytest.raises(ChannelNotAvailableError, match=r"'text' is not available.*geo"):
        reg.get(TEXT)
    with pytest.raises(ChannelNotAvailableError, match=r"madeleine-mcp\[image\]"):
        reg.get(IMAGE)
    with pytest.raises(ChannelNotAvailableError, match="future work"):
        reg.get(SMELL)
    assert reg.unregister(GEO)
    assert not reg.unregister(GEO)
    with pytest.raises(TypeError):
        reg.register("geo")  # type: ignore[arg-type]


def test_hash_registry_rejects_unknown_and_smell() -> None:
    with pytest.raises(ChannelNotAvailableError, match="unknown channel 'touch'"):
        hash_registry(("touch",))
    with pytest.raises(ChannelNotAvailableError, match="future work"):
        hash_registry((SMELL,))


def test_default_registry_skips_missing_extras(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, no_st: None
) -> None:
    _hide(
        monkeypatch,
        "open_clip",
        "torch",
        "PIL",
        "transformers",
        "soundfile",
        "sentence_transformers",
    )
    with caplog.at_level(logging.WARNING, logger="madeleine.channels"):
        reg = default_registry(MadeleineConfig(db_path=":memory:"))
    assert reg.names() == tuple(sorted((TEXT, TIME, GEO, BIOMETRIC)))
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "skipping channel 'image'" in messages
    assert "madeleine-mcp[image]" in messages
    assert "skipping channel 'audio'" in messages
    assert "madeleine-mcp[audio]" in messages
    assert "madeleine-mcp[text]" in messages  # auto text backend fell back to hash
    assert isinstance(reg.get(TEXT).embedder, HashEmbedder)


def test_default_registry_registers_extras_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    """When extras look installed, image/audio register without importing anything heavy."""
    _pretend_installed(monkeypatch, "open_clip", "torch", "PIL", "transformers", "soundfile")
    reg = default_registry(MadeleineConfig(text_backend="hash"))
    assert reg.get(IMAGE).spec.model_id == "open_clip/ViT-B-32/laion2b_s34b_b79k"
    assert reg.get(IMAGE).spec.dim == 512
    assert reg.get(IMAGE).spec.extra == "image"
    assert reg.get(AUDIO).spec.model_id == "laion/clap-htsat-unfused"
    assert reg.get(AUDIO).spec.dim == 512
    assert reg.get(AUDIO).spec.extra == "audio"


def test_default_registry_channel_selection(caplog: pytest.LogCaptureFixture) -> None:
    cfg = MadeleineConfig(channels=(GEO, TIME, GEO), text_backend="hash")
    assert default_registry(cfg).names() == (GEO, TIME)
    with caplog.at_level(logging.WARNING, logger="madeleine.channels"):
        reg = default_registry(MadeleineConfig(channels=(GEO, SMELL)))
    assert reg.names() == (GEO,)
    assert "smell" in caplog.text
    with pytest.raises(ChannelNotAvailableError, match="unknown channel 'touch'"):
        default_registry(MadeleineConfig(channels=("touch",)))


def test_forced_sentence_transformers_without_package(no_st: None) -> None:
    ch = make_text_channel(MadeleineConfig(text_backend="sentence-transformers"))
    assert isinstance(ch.embedder, SentenceTransformerEmbedder)
    assert ch.spec.dim == 384  # known without loading
    assert ch.spec.model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert ch.spec.extra == "text"
    with pytest.raises(ChannelNotAvailableError, match=r"pip install 'madeleine-mcp\[text\]'"):
        ch.embedder.embed(Observation(TEXT, text="hello"))


def test_auto_backend_warns_once(no_st: None, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="madeleine.channels"):
        first = make_text_channel(MadeleineConfig())
        second = make_text_channel(MadeleineConfig())
    assert isinstance(first.embedder, HashEmbedder)
    assert isinstance(second.embedder, HashEmbedder)
    assert sum("madeleine-mcp[text]" in r.getMessage() for r in caplog.records) == 1
    assert first.spec.model_id == "hash-v1/dim=256"
    assert first.spec.extra is None


def test_auto_backend_prefers_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    _pretend_installed(monkeypatch, "sentence_transformers")
    ch = make_text_channel(MadeleineConfig(text_model="all-mpnet-base-v2"))
    assert isinstance(ch.embedder, SentenceTransformerEmbedder)
    assert ch.spec.dim == 768
    assert not ch.embedder.loaded


def test_hash_backend_forced() -> None:
    ch = make_text_channel(MadeleineConfig(text_backend="hash"), hash_dim=64)
    assert isinstance(ch.embedder, HashEmbedder)
    assert ch.spec.dim == 64


def test_unknown_st_model_dim_needs_package(no_st: None) -> None:
    emb = SentenceTransformerEmbedder("someone/custom-model")
    with pytest.raises(ChannelNotAvailableError, match=r"madeleine-mcp\[text\]"):
        _ = emb.dim
    assert SentenceTransformerEmbedder("someone/custom-model", dim=100).dim == 100
