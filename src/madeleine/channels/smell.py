"""The ``smell`` channel: reserved for future olfactory sensors (not available in v0).

:class:`SmellAdapter` exists so that the name is claimed and callers get a clear error;
it is **not** registered by :func:`~madeleine.channels.registry.default_registry`.

Sketch of a real electronic-nose channel (future work):

* An e-nose is an array of gas sensors (metal-oxide, electrochemical, PID...) each
  responding to a family of volatile organic compounds (VOCs). A reading is a vector of
  sensor resistances / concentrations, e.g. ``{"mox_1": 12.3, "mox_2": 4.1, ...,
  "humidity": 41.0, "temp_c": 22.5}``.
* The adapter would map it to ``Observation(SMELL, numbers={...})``: baseline-correct each
  sensor (``r / r0`` against clean air), log-scale, and drop humidity/temperature after
  using them for compensation.
* The embedder would turn the compensated sensor pattern into a fixed-length vector -
  for raw arrays an L2-normalised pattern (odour identity is in the *shape* of the
  response more than its magnitude), or a learned odour embedding (e.g. a model trained to
  map molecules / VOC spectra to perceptual odour descriptors).
* Similarity: ``max(0, cosine)`` on the normalised pattern, ``solo_trigger=True`` - smell
  is the canonical Proustian cue.
"""

from __future__ import annotations

from typing import NoReturn

from madeleine.core.types import (
    SMELL,
    ChannelName,
    ChannelNotAvailableError,
    Observation,
    RawSignal,
)

__all__ = ["SMELL_UNAVAILABLE", "SmellAdapter", "make_smell_channel"]

SMELL_UNAVAILABLE = (
    "channel 'smell' is not available: olfactory sensors (e-noses) are future work in "
    "madeleine. You can register your own Channel named 'smell' (see docs/channels.md)."
)


class SmellAdapter:
    """Placeholder adapter: :meth:`normalize` always raises ``ChannelNotAvailableError``."""

    @property
    def channel(self) -> ChannelName:
        return SMELL

    def normalize(self, signal: RawSignal) -> Observation:
        raise ChannelNotAvailableError(SMELL_UNAVAILABLE)


def make_smell_channel() -> NoReturn:
    """Always raises :class:`ChannelNotAvailableError` (there is no smell channel yet)."""
    raise ChannelNotAvailableError(SMELL_UNAVAILABLE)
