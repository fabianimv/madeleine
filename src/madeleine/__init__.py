"""madeleine: involuntary, cue-triggered memory for AI agents.

Importing this package is cheap: no models are loaded and no network calls are made.
"""

from madeleine.core.types import (
    Cue,
    MadeleineConfig,
    Memory,
    Observation,
    RawSignal,
    RecallConfig,
    RecallMode,
    RecallQuery,
    Recollection,
    ScoredMemory,
    SensoryContext,
    SensoryFingerprint,
)

__version__ = "0.1.1"

__all__ = [
    "Cue",
    "MadeleineConfig",
    "Memory",
    "Observation",
    "RawSignal",
    "RecallConfig",
    "RecallMode",
    "RecallQuery",
    "Recollection",
    "ScoredMemory",
    "SensoryContext",
    "SensoryFingerprint",
    "__version__",
]
