"""Paths into the vendored soft_walker env, shared by its tests."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SOFT_ROOT = SRC_ROOT / "envs" / "soft_walker"
SCRIPTS = SOFT_ROOT / "scripts"
# Command-line tests use the restored tools in the integrated package layout.
needs_scripts = pytest.mark.skipif(
    not SCRIPTS.is_dir(), reason="soft_walker command-line tools are unavailable"
)
