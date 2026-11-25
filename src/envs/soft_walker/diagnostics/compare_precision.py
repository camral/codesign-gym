"""Float32-versus-float64 arrays/AD on the same device and source revision.

A separate gate is important when float32 finite differences reach a subtraction
noise floor before satisfying a tight relative threshold on a tiny objective.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from envs.soft_walker.engine.types import (
        PathLike,
    )

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def compare(root: PathLike) -> Report:
    """Compare existing evidence under the recorded precision/device protocols; do not rerun simulation."""
    root = Path(root)
    rows = []
    failures: list[str] = []
    p64 = root / "float64" / "numerics" / "reference_arrays.npz"
    p32 = root / "float32" / "numerics" / "reference_arrays.npz"
    if not p64.exists() or not p32.exists():
        return {"passed": False, "failures": ["missing_precision_arrays"], "results": []}
    with np.load(p64, allow_pickle=False) as x, np.load(p32, allow_pickle=False) as y:
        if set(x.files) != set(y.files):
            failures.append("array_key_mismatch")
        for key in sorted(set(x.files) & set(y.files)):
            if key.endswith("max_residual"):
                continue
            a, b = x[key].astype(np.float64), y[key].astype(np.float64)
            # Componentwise tolerances plus reported norms. Small objective
            # gradients are not compared with a blanket O(1e-3) absolute floor.
            rtol = 3e-3 if "gradient" in key else 3e-4
            atol = 3e-8 if "gradient" in key or "objective" in key else 3e-6
            ok = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            ok = bool(ok and np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=False))
            row: Report = {"array": key, "passed": ok, "rtol": rtol, "atol": atol}
            if a.shape == b.shape:
                row["max_abs_error"] = float(np.max(np.abs(a - b)))
                row["relative_l2_error"] = float(
                    np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-15)
                )
            rows.append(row)
            if not ok:
                failures.append(key)
    return {
        "passed": not failures,
        "failures": failures,
        "results": rows,
        "meaning": "same-device mixed-precision consistency, NOT independent analytic correctness or cross-device parity",
    }


def main() -> int:
    """Command entry point. Float32-versus-float64 arrays/AD on the same device and source revision."""
    p = argparse.ArgumentParser()
    p.add_argument("root")
    a = p.parse_args()
    report = compare(a.root)
    out = Path(a.root) / "precision_comparison.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
