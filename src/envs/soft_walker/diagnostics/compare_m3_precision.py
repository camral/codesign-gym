"""Cross-precision AD and forward comparison; independent float64 FD is required."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if __package__ in (None, ""):
    # Direct-file execution must select this checkout before package imports.
    _source_root = str(Path(__file__).resolve().parents[3])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)

from envs.soft_walker.diagnostics._precision_comparison import compare_numerics
from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
)

if TYPE_CHECKING:
    from envs.soft_walker.engine.types import (
        PathLike,
    )

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def compare(root: PathLike) -> Report:
    """Compare existing evidence under the recorded precision/device protocols; do not rerun simulation."""
    root = Path(root)
    report = compare_numerics(root, "m3_numerics")
    (root / "m3_precision_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root")
    a = p.parse_args()
    report = compare(a.root)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
