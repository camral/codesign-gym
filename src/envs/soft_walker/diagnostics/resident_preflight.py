#!/usr/bin/env python3
"""One bounded capture probe per fresh process, before expensive physics."""

from __future__ import annotations

import argparse
import faulthandler
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

ROOT = REPO_ROOT
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/envs/soft_walker/scripts")]

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def main() -> int:
    """Command entry point. One bounded capture probe per fresh process, before expensive physics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe", required=True, choices=("control", "backsolve", "implicit_gradient")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from envs.soft_walker.shared import environment_record, write_json, write_npz

    result: Report = {"passed": False, "completed": False, "probe": args.probe}
    start = time.perf_counter()
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        import jax
        import numpy as np

        from envs.soft_walker.diagnostics.residency.probes import capture_probe
        from envs.soft_walker.engine.execution import ResidentPolicy, compile_program

        dtype = np.dtype(os.environ.get("CODESIGN_TEST_DTYPE", "float64"))
        device = os.environ.get("CODESIGN_RESIDENT_DEVICE", "cpu")
        jax.config.update("jax_default_matmul_precision", "highest")
        write_json(args.out / "environment.json", environment_record(ROOT))
        raw, reference, initial = capture_probe(args.probe, dtype)
        program = compile_program(
            raw,
            (initial,),
            policy=ResidentPolicy(device),
            out=args.out / "executable",
            name=args.probe,
        )
        inputs = program.stage(np.asarray(0.23, dtype))
        # Change both input values and control-flow branch/counts, not only
        # addresses. This catches stale parameters after graph updates.
        alternate = program.stage(np.asarray(-0.41, dtype))
        expected = reference(np.asarray(0.23, dtype))
        alternate_expected = reference(np.asarray(-0.41, dtype))
        _actual = program.certify(
            *inputs,
            alternate_arguments=alternate,
            expected_host=expected,
            alternate_expected_host=alternate_expected,
            detailed_trace=args.trace,
        )
        for index, p in enumerate((0.23, -0.41, 0.71, 0.23)):
            values = program.stage(np.asarray(p, dtype))
            got = program(*values)
            want = reference(np.asarray(p, dtype))
            for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
                np.testing.assert_allclose(
                    a,
                    b,
                    rtol=2e-5 if dtype == np.float32 else 5e-10,
                    atol=2e-7 if dtype == np.float32 else 5e-12,
                )
            write_npz(
                args.out / f"replay_{index}.npz",
                parameter=np.asarray(p, dtype),
                **{f"actual_{i}": x for i, x in enumerate(jax.tree.leaves(got))},
                **{f"expected_{i}": x for i, x in enumerate(jax.tree.leaves(want))},
            )
        program.benchmark(*inputs, repeats=3)
        result.update(passed=True, cuda_residency_demonstrated=device == "cuda")
    except BaseException as error:
        result.update(exception=repr(error), traceback=traceback.format_exc())
        print(result["traceback"], file=sys.stderr, flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
        result.update(completed=True, wall_seconds=time.perf_counter() - start)
        write_json(args.out / "result.json", result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
