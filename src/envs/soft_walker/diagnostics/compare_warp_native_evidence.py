#!/usr/bin/env python3
"""Compare native CPU/CUDA arrays; reject changed source or failed collections."""

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


def verify(root: Path) -> Report:
    """Check an evidence bundle's manifest, hashes, and required success fields."""
    for line in (root / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Unsafe checksum path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Checksum mismatch: {path}")
    summary = json.loads((root / "summary.json").read_text())
    if not summary["success"]:
        raise ValueError(f"{root}: validation collection was not successful")
    return json.loads((root / "manifest.json").read_text())


def main() -> int:
    """Command entry point. Compare native CPU/CUDA arrays; reject changed source or failed collections."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    report: Report = dict(success=False, comparisons=[])
    try:
        left, right = verify(a.left), verify(a.right)
        if left["source_hashes"] != right["source_hashes"]:
            raise ValueError("Source identity differs")
        files = sorted(a.left.glob("*/arrays.npz"))
        if not files or set(p.relative_to(a.left) for p in files) != set(
            p.relative_to(a.right) for p in a.right.glob("*/arrays.npz")
        ):
            raise ValueError("Saved parity fixture selections differ or are empty")
        for path in files:
            relative = path.relative_to(a.left)
            with (
                np.load(path, allow_pickle=False) as left_arrays,
                np.load(a.right / relative, allow_pickle=False) as right_arrays,
            ):
                if set(left_arrays.files) != set(right_arrays.files):
                    raise ValueError(f"{relative}: array keys differ")
                for key in left_arrays.files:
                    if not key.startswith("native_"):
                        continue
                    x, y = left_arrays[key], right_arrays[key]
                    tolerance = 3e-3 if x.dtype == np.float32 else 3e-7
                    absolute = 3e-7 if x.dtype == np.float32 else 2e-9
                    good = (
                        x.shape == y.shape
                        and x.dtype == y.dtype
                        and np.all(np.isfinite(x))
                        and np.all(np.isfinite(y))
                    )
                    good = good and (
                        np.linalg.norm(x - y) <= tolerance * np.linalg.norm(x) + 1e-24
                        if key.startswith("native_gradient_")
                        else np.allclose(x, y, rtol=tolerance, atol=absolute)
                    )
                    report["comparisons"].append(
                        dict(
                            fixture=str(relative),
                            key=key,
                            passed=bool(good),
                            relative_l2=float(np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-30))
                            if x.shape == y.shape
                            else None,
                            rtol=tolerance,
                            atol=absolute,
                        )
                    )
        report["success"] = bool(report["comparisons"]) and all(
            row["passed"] for row in report["comparisons"]
        )
    except Exception as error:
        report["exception"] = repr(error)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
