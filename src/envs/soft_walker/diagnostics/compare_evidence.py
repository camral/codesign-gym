"""Compare saved deterministic CPU/CUDA numerical arrays, never timings alone."""

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


def main() -> int:
    """Command entry point. Compare saved deterministic CPU/CUDA numerical arrays, never timings alone."""
    p = argparse.ArgumentParser()
    p.add_argument("reference")
    p.add_argument("candidate")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ref = Path(a.reference)
    cand = Path(a.candidate)
    rm = json.loads((ref / "manifest.json").read_text())
    cm = json.loads((cand / "manifest.json").read_text())
    results = []
    failures: list[str] = []
    for label, m in (("reference", rm), ("candidate", cm)):
        if not m.get("completed") or any(s["returncode"] != 0 for s in m.get("stages", [])):
            failures.append(label + ":validation_not_successful")
    if set(rm["dtypes"]) != set(cm["dtypes"]):
        failures.append("dtype_set_mismatch")
    if (
        rm.get("selected_stages") != cm.get("selected_stages")
        or rm.get("unit_filter") != cm.get("unit_filter")
        or rm.get("case_filter") != cm.get("case_filter")
    ):
        failures.append("test_selection_mismatch")
    if rm["source_sha256"] != cm["source_sha256"]:
        failures.append("source_hash_mismatch")
    for label, root in [("reference", ref), ("candidate", cand)]:
        hashes = root / "SHA256SUMS"
        if not hashes.exists():
            failures.append(label + ":checksums_missing")
        else:
            for checksum_row in hashes.read_text().splitlines():
                expected, name = checksum_row.split("  ", 1)
                path = (root / name).resolve()
                if (
                    root.resolve() not in path.parents
                    or not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected
                ):
                    failures.append(label + ":checksum:" + name)
    stages = {
        s["stage"]
        for s in rm["stages"] + cm["stages"]
        if s["stage"] in ("numerics", "m2_numerics", "m3_numerics")
    }
    for dtype in sorted(set(rm["dtypes"]) & set(cm["dtypes"])):
        for stage in sorted(stages):
            rp = ref / dtype / stage / "reference_arrays.npz"
            cp = cand / dtype / stage / "reference_arrays.npz"
            if not rp.exists() or not cp.exists():
                failures.append(dtype + ":" + stage + ":missing_reference_arrays")
                continue
            with np.load(rp, allow_pickle=False) as x, np.load(cp, allow_pickle=False) as y:
                if set(x.files) != set(y.files):
                    failures.append(dtype + ":" + stage + ":array_key_mismatch")
                    continue
                for k in x.files:
                    if k.endswith("max_residual"):
                        continue
                    rtol, atol = (3e-4, 3e-6) if dtype == "float32" else (2e-6, 2e-9)
                    shape_match = x[k].shape == y[k].shape
                    passed = shape_match and np.allclose(
                        x[k], y[k], rtol=rtol, atol=atol, equal_nan=False
                    )
                    row: Report = {
                        "dtype": dtype,
                        "stage": stage,
                        "array": k,
                        "passed": bool(passed),
                        "rtol": rtol,
                        "atol": atol,
                    }
                    if shape_match:
                        row["max_abs_error"] = float(np.max(np.abs(x[k] - y[k])))
                        row["relative_l2_error"] = float(
                            np.linalg.norm(x[k] - y[k]) / max(np.linalg.norm(x[k]), 1e-15)
                        )
                    results.append(row)
                    if not passed:
                        failures.append(dtype + ":" + stage + ":" + k)
    if not results:
        failures.append("no_arrays_compared")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"passed": not failures, "failures": failures, "results": results}, indent=2)
        + "\n"
    )
    print(out.read_text())
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
