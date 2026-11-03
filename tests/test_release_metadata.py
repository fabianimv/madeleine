"""Release metadata must agree, or PyPI / MCP Registry publishing fails after upload."""

from __future__ import annotations

import json
import re
from pathlib import Path

import madeleine

ROOT = Path(__file__).resolve().parent.parent


def test_server_json_matches_package_and_readme() -> None:
    server = json.loads((ROOT / "server.json").read_text())
    name = server["name"]
    # The registry grants io.github.<repository_owner>/* from the GitHub OIDC token and
    # matches case-sensitively; GitHub's canonical login is lower-case.
    assert name == "io.github.fabianimv/madeleine"
    assert server["version"] == madeleine.__version__
    assert server["packages"][0]["version"] == madeleine.__version__
    assert server["packages"][0]["identifier"] == "madeleine-mcp"
    readme = (ROOT / "README.md").read_text()
    # Ownership token: followed by a boundary (space, newline, tag or comment close).
    assert re.search(rf"mcp-name: {re.escape(name)}(\s|-->|<)", readme)
