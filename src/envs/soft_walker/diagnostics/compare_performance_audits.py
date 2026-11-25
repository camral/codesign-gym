"""Compare saved source-identical focused performance evidence across devices."""

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

# This protocol compares elements, with the original focused-audit limits.
FLOAT32_TOLERANCE = (2e-3, 3e-6)
FLOAT64_TOLERANCE = (1e-6, 1e-10)
RELATIVE_NORM_FLOOR = 1e-12


def load(root: Path) -> tuple[Report, list[str]]:
    """Load the audit manifest and result records from an existing evidence directory."""
    manifest = json.loads((root / "manifest.json").read_text())
    failures: list[str] = []
    if manifest.get("schema") != "codesign_performance_audit_v1":
        failures.append("unexpected evidence schema")
    if not manifest.get("passed") or not manifest.get("completed"):
        failures.append("selected stages did not all complete successfully")
    for line in (root / "SHA256SUMS").read_text().splitlines():
        checksum, name = line.split("  ", 1)
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            failures.append("unsafe checksum path: " + name)
            continue
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            failures.append("checksum: " + name)
    return manifest, failures


def compare(left: Path, right: Path) -> Report:
    """Compare existing evidence under the recorded precision/device protocols; do not rerun simulation."""
    lm, le = load(left)
    rm, re = load(right)
    errors = ["left: " + x for x in le] + ["right: " + x for x in re]
    if lm["source_hashes"] != rm["source_hashes"]:
        errors.append("source hashes differ")
    lf = {p.relative_to(left).as_posix(): p for p in left.rglob("references.npz")}
    rf = {p.relative_to(right).as_posix(): p for p in right.rglob("references.npz")}
    if set(lf) != set(rf) or not lf:
        errors.append("reference stage coverage differs or is empty")
    rows = []
    for name in sorted(set(lf) & set(rf)):
        with np.load(lf[name], allow_pickle=False) as a, np.load(rf[name], allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                errors.append(name + ": array names differ")
            for key in sorted(set(a.files) & set(b.files)):
                x, y = a[key], b[key]
                rtol, atol = FLOAT32_TOLERANCE if x.dtype == np.float32 else FLOAT64_TOLERANCE
                good = (
                    x.shape == y.shape
                    and x.dtype == y.dtype
                    and np.all(np.isfinite(x))
                    and np.all(np.isfinite(y))
                )
                relative, maximum = None, None
                if good:
                    diff = x.astype(np.float64) - y.astype(np.float64)
                    relative = float(
                        np.linalg.norm(diff.ravel())
                        / max(np.linalg.norm(x.astype(np.float64).ravel()), RELATIVE_NORM_FLOOR)
                    )
                    maximum = float(np.max(np.abs(diff), initial=0))
                    good = bool(np.allclose(x, y, rtol=rtol, atol=atol))
                rows.append(
                    dict(
                        stage=name,
                        array=key,
                        passed=bool(good),
                        relative_l2=relative,
                        max_absolute=maximum,
                        rtol=rtol,
                        atol=atol,
                    )
                )
                if not good:
                    errors.append(name + ": " + key)
    return {
        "passed": not errors,
        "failures": errors,
        "comparisons": rows,
        "scope": "Numerical parity only. Timing differences are recorded separately, not treated as correctness failures.",
    }


def main() -> int:
    """Command entry point. Compare saved source-identical focused performance evidence across devices."""
    p = argparse.ArgumentParser()
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = compare(args.left, args.right)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Numerical parity passed={result['passed']}; {len(result['comparisons'])} arrays")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
