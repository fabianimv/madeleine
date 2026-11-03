"""Import hygiene: importing every channels module pulls in no heavy ML library."""

from __future__ import annotations

import json
import subprocess
import sys

HEAVY = ("torch", "open_clip", "transformers", "sentence_transformers", "PIL", "soundfile")
MODULES = (
    "madeleine.channels",
    "madeleine.channels.audio",
    "madeleine.channels.biometric",
    "madeleine.channels.geo",
    "madeleine.channels.hashing",
    "madeleine.channels.image",
    "madeleine.channels.registry",
    "madeleine.channels.similarity",
    "madeleine.channels.smell",
    "madeleine.channels.text",
    "madeleine.channels.time",
)


def test_channels_import_no_heavy_modules() -> None:
    code = (
        "import importlib, json, sys\n"
        f"for m in {MODULES!r}: importlib.import_module(m)\n"
        "from madeleine.channels import default_registry, hash_registry\n"
        "from madeleine.core.types import MadeleineConfig\n"
        "default_registry(MadeleineConfig(text_backend='auto'))\n"
        "hash_registry()\n"
        f"print(json.dumps(sorted(m for m in {HEAVY!r} if m in sys.modules)))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []
