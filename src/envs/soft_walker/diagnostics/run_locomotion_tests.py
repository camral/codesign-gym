#!/usr/bin/env python3
"""Run only locomotion infrastructure tests, recording a ZIP even on failure.

This does NOT run expensive gait optimization or the historical validation suite.
Run scripts/locomotion.py separately for source-identifiable physical experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Direct-file execution must select this checkout before package imports.
    _source_root = str(Path(__file__).resolve().parents[3])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)

from envs.soft_walker.diagnostics._focused_tests import run_focused_tests
from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
    require_source_checkout,
)

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def write_json(path: Path, value: Any) -> None:
    """Write a JSON report, retaining the command's array/scalar encoding and error behavior."""
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def main() -> int:
    """Command entry point. Run only locomotion infrastructure tests, recording a ZIP even on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtypes", nargs="+", choices=("float64", "float32"), default=["float64", "float32"]
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Maximum seconds per precision; not a numerical tolerance",
    )
    parser.add_argument(
        "--offline-spaces",
        action="store_true",
        help="Sandbox development only; does not validate Gymnasium",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    output = args.out.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("--out must be a fresh directory")
    try:
        require_source_checkout(REPO_ROOT)
    except ValueError as exc:
        parser.error(str(exc))
    output.mkdir(parents=True, exist_ok=True)
    return run_focused_tests(
        args,
        output,
        root=REPO_ROOT,
        src_root=SRC_ROOT,
        selection=["tests/soft_walker/test_locomotion.py"],
        schema="locomotion_focused_tests_v1",
        write_json=write_json,
        stage_environment=lambda stage: {},
    )


if __name__ == "__main__":
    raise SystemExit(main())
