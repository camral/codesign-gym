"""Shared M2/M3 cross-precision acceptance and independent ladder checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

Report = dict[str, Any]


def compare_numerics(root: Path, directory: str) -> Report:
    """Compare existing evidence under the recorded precision/device protocols; do not rerun simulation."""
    root = Path(root)
    failures: list[str] = []
    rows = []
    files = [root / d / directory / "reference_arrays.npz" for d in ("float64", "float32")]
    if not all(p.exists() for p in files):
        failures.append("missing_reference_arrays")
    else:
        for d in ("float64", "float32"):
            p = root / d / directory / "summary.json"
            if not p.exists() or not json.loads(p.read_text()).get("passed"):
                failures.append(d + ":numerics_not_passed")
        ladders = {
            d: json.loads((root / d / directory / "gradient_ladders.json").read_text())["reports"]
            for d in ("float64", "float32")
        }
        reference_ladders = {r["name"]: r for r in ladders["float64"]}
        for case in ladders["float32"]:
            if case.get("cross_precision_required"):
                ref = reference_ladders.get(case["name"], {})
                if not ref.get("passed") or not ref.get("second_order_window"):
                    failures.append(case["name"] + ":missing_float64_order_evidence")
        with np.load(files[0], allow_pickle=False) as x, np.load(files[1], allow_pickle=False) as y:
            if set(x.files) != set(y.files):
                failures.append("array_key_mismatch")
            for k in sorted(set(x.files) & set(y.files)):
                # Relative L2 is meaningful for a gradient vector, while a small
                # absolute allowance handles components expected near zero.
                rtol = 2e-3 if k.endswith("gradient") else 1e-3
                atol = 2e-6
                a, b = x[k], y[k]
                shape = a.shape == b.shape
                finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
                diff = np.linalg.norm(a - b) if shape else float("inf")
                ref = np.linalg.norm(a)
                limit = atol * np.sqrt(a.size) + rtol * ref
                passed = shape and finite and diff <= limit
                rows.append(
                    dict(
                        array=k,
                        passed=bool(passed),
                        relative_l2_error=float(diff / max(ref, 1e-15)),
                        absolute_l2_error=float(diff),
                        limit=float(limit),
                        rtol=rtol,
                        atol=atol,
                    )
                )
                if not passed:
                    failures.append(k)
    report: Report = dict(
        passed=not failures,
        failures=failures,
        results=rows,
        scope="Float32 AD vs independently FD-validated float64 AD; not a replacement for native-precision ladders",
    )
    return report
