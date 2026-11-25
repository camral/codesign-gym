#!/usr/bin/env python3
"""Compare complete source-matched evidence; never treat failed runs as parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

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

# The followup protocol measures an L2 norm, with its original precision limits.
FLOAT32_TOLERANCE = (3e-3, 3e-7)
FLOAT64_TOLERANCE = (2e-7, 2e-11)
RELATIVE_NORM_FLOOR = 1e-30


def protocol(manifest: Report) -> Report:
    """Compare numerical selections while allowing device and collection-only options."""
    arguments = manifest.get("arguments", {})
    skip_solids = arguments.get("skip_solids", False)
    return {
        "dtypes": sorted(arguments.get("dtypes", [])),
        "physics": [] if skip_solids else sorted(arguments.get("physics", [])),
        "skip_solids": skip_solids,
        "skip_mpm": arguments.get("skip_mpm", False),
        "skip_spring": arguments.get("skip_spring", False),
        "offline_spaces": arguments.get("offline_spaces", manifest.get("offline_spaces", False)),
    }


def verify(root: Path) -> Report:
    """Check an evidence bundle's manifest, hashes, and required success fields."""
    files = json.loads((root / "file_hashes.json").read_text())
    for name, digest in files.items():
        p = (root / name).resolve()
        if not p.is_relative_to(root.resolve()):
            raise ValueError("Unsafe evidence path")
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Corrupt evidence: {name}")
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if not all(summary.get(k) for k in ("success", "complete", "source_unchanged")):
        raise ValueError(f"Incomplete or failed collection: {root}")
    return manifest


def main() -> int:
    """Command entry point. Compare complete source-matched evidence; never treat failed runs as parity."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cpu", type=Path)
    p.add_argument("cuda", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result: Report = {"passed": False, "comparisons": {}}
    try:
        a, b = verify(args.cpu), verify(args.cuda)
        if a["source_sha256"] != b["source_sha256"]:
            raise ValueError("Source mismatch")
        if protocol(a) != protocol(b):
            raise ValueError("Numerical protocol selections differ")
        cpu_files = {file.relative_to(args.cpu) for file in args.cpu.rglob("parity.npz")}
        cuda_files = {file.relative_to(args.cuda) for file in args.cuda.rglob("parity.npz")}
        if not cpu_files or cpu_files != cuda_files:
            raise ValueError("Saved parity fixture selections differ or are empty")
        for rel in sorted(cpu_files):
            file, other = args.cuda / rel, args.cpu / rel
            with np.load(file, allow_pickle=False) as x, np.load(other, allow_pickle=False) as y:
                if set(x.files) != set(y.files):
                    raise ValueError(f"Array schema mismatch: {rel}")
                for k in x.files:
                    xx, yy = x[k], y[k]
                    if xx.shape != yy.shape or xx.dtype != yy.dtype:
                        raise ValueError(f"Shape/dtype mismatch: {rel}/{k}")
                    if not np.all(np.isfinite(xx)) or not np.all(np.isfinite(yy)):
                        raise ValueError("Nonfinite parity data")
                    error = float(np.linalg.norm(xx.astype("float64") - yy.astype("float64")))
                    ref = float(np.linalg.norm(yy.astype("float64")))
                    tol, absolute = (
                        FLOAT32_TOLERANCE if xx.dtype == np.float32 else FLOAT64_TOLERANCE
                    )
                    passed = bool(error <= tol * ref + absolute)
                    result["comparisons"][str(rel) + "/" + k] = {
                        "passed": passed,
                        "relative_l2": error / max(ref, RELATIVE_NORM_FLOOR),
                    }
        if not result["comparisons"]:
            raise ValueError("No comparable numerical arrays")
        result["passed"] = all(v["passed"] for v in result["comparisons"].values())
    except Exception as exc:
        result["error"] = str(exc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
