#!/usr/bin/env python3
"""Isolated precision/device worker; writes stage status even on failure."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Direct-file execution must select this checkout before package imports.
    _source_root = str(Path(__file__).resolve().parents[3])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)

from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
)

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def main() -> int:
    """Command entry point. Isolated precision/device worker; writes stage status even on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--mode", choices=("parity", "ladder", "audit"), required=True)
    parser.add_argument("--dim", type=int, choices=(2, 3), default=2)
    parser.add_argument(
        "--law", choices=("active_strain", "active_stress"), default="active_stress"
    )
    parser.add_argument("--terrain", choices=("plane", "slope", "box"), default="plane")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["JAX_ENABLE_X64"] = "true" if args.dtype == "float64" else "false"
    # JAX supplies an independent CPU oracle even when Warp is evaluated on GPU.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    start = time.perf_counter()
    status: Report = dict(arguments=vars(args) | {"out": str(args.out)}, success=False)
    result: object
    try:
        from envs.soft_walker.diagnostics.warp.checks import directional_ladder, parity_case

        if args.mode == "parity":
            result = parity_case(
                args.dim,
                args.dtype,
                args.law,
                args.device,
                args.out,
                terrain=args.terrain,
                repeats=args.repeats,
            )
        elif args.mode == "ladder":
            result = directional_ladder(args.dtype, args.device, args.out)
        else:
            from envs.soft_walker.diagnostics.warp.audit import audit_episode

            result = audit_episode(args.dtype, args.out)
        status["success"] = True
        print(json.dumps(result, indent=2), flush=True)
    except Exception:
        status["exception"] = traceback.format_exc()
        print(status["exception"], file=sys.stderr, flush=True)
    finally:
        status["seconds"] = time.perf_counter() - start
        (args.out / "status.json").write_text(json.dumps(status, indent=2))
    return 0 if status["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
