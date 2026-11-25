#!/usr/bin/env python3
"""Whole-episode parity, AD/FD, and residency; stop at the first failing phase.

The ordinary-compilation paths are independent diagnostics, not runtime
fallbacks. The new fused solver is selected explicitly and compared with the
legacy incremental solver at identical physical/numerical tolerances.
"""

from __future__ import annotations

import argparse
import faulthandler
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    require_source_checkout,
)

if TYPE_CHECKING:
    from envs.soft_walker.diagnostics.host_types import NumericalTolerance


ROOT = REPO_ROOT
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "tests/soft_walker"),
    str(ROOT / "src/envs/soft_walker/scripts"),
]

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def main() -> int:
    """Command entry point. Whole-episode parity, AD/FD, and residency; stop at the first failing phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--driver", required=True, choices=("spring", "mpm2", "mpm3", "fem", "beam")
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    try:
        require_source_checkout(ROOT)
    except ValueError as exc:
        parser.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=True)
    from envs.soft_walker.shared import environment_record, write_json, write_npz

    status: Report = {
        "passed": False,
        "completed": False,
        "phases": [],
        "resident_gmres_method": "fused",
        "legacy_gmres_method": "incremental",
    }
    environments = []
    start = time.perf_counter()
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)

    def phase(name: str) -> None:
        status["phase"] = name
        status["phases"].append({"name": name, "elapsed_s": time.perf_counter() - start})
        write_json(args.out / "result.json", status)
        print(f"phase={name}", flush=True)

    try:
        import jax
        import numpy as np
        from test_resident_execution import physical_case

        from envs.soft_walker.engine.execution import ResidentPolicy, compile_program
        from envs.soft_walker.shared import difference_ladder

        dtype = np.dtype(os.environ.get("CODESIGN_TEST_DTYPE", "float64"))
        device = os.environ.get("CODESIGN_RESIDENT_DEVICE", "cpu")
        jax.config.update("jax_default_matmul_precision", "highest")
        write_json(args.out / "environment.json", environment_record(ROOT))
        phase("build_explicit_fixtures")
        _, _, theta, env, embedding, program = physical_case(
            args.driver, dtype, gmres_method="fused"
        )
        environments.append(env)
        old_loss, old_run, old_theta, old_env, _, _ = physical_case(
            args.driver, dtype, gmres_method="incremental"
        )
        environments.append(old_env)
        np.testing.assert_array_equal(theta, old_theta)
        write_json(
            args.out / "fixture.json",
            {
                "resident_config": asdict(env.config),
                "legacy_config": asdict(old_env.config),
                "only_intended_solver_change": "incremental -> fused with array back substitution",
            },
        )
        policy = ResidentPolicy(device)
        raw_loss = lambda p: program.loss(embedding(p), p)
        raw_record = lambda p: program.record(embedding(p), p)
        raw_gradient = jax.value_and_grad(raw_loss)
        alternate_theta = theta * np.asarray(0.999, dtype)
        # Explicitly disable command buffers for independent numerical controls.
        # These calls are not accepted as production GPU-resident execution.
        ordinary_options = {"xla_gpu_enable_command_buffer": ""} if device == "cuda" else {}
        ordinary = lambda f: jax.jit(f, compiler_options=ordinary_options)
        tolerance: NumericalTolerance = (
            dict(rtol=1e-8, atol=1e-10) if dtype == np.float64 else dict(rtol=5e-4, atol=1e-6)
        )

        phase("ordinary_forward_reference")
        legacy_value = ordinary(old_loss)(theta)
        normal_loss = ordinary(raw_loss)
        expected_value = normal_loss(theta)
        alternate_value = normal_loss(alternate_theta)
        jax.block_until_ready((legacy_value, expected_value, alternate_value))
        np.testing.assert_allclose(expected_value, legacy_value, **tolerance)
        if not np.isfinite(np.asarray(expected_value)).all():
            raise ValueError("Ordinary reference is nonfinite before resident compilation")
        write_npz(
            args.out / "forward_reference.npz",
            theta=theta,
            legacy=legacy_value,
            fused=expected_value,
            alternate=alternate_value,
        )

        phase("resident_forward_compile_and_certify")
        forward = compile_program(
            raw_loss, (theta,), policy=policy, out=args.out / "loss", name="whole_loss"
        )
        inputs, alternate = forward.stage(theta), forward.stage(alternate_theta)
        forward.certify(
            *inputs,
            alternate_arguments=alternate,
            detailed_trace=args.trace,
            expected_host=jax.device_get(expected_value),
            alternate_expected_host=jax.device_get(alternate_value),
        )

        phase("ordinary_record_reference")
        normal_record = ordinary(raw_record)
        expected_record = normal_record(theta)
        alternate_record = normal_record(alternate_theta)
        original = ordinary(old_run)(theta)
        jax.block_until_ready((expected_record, alternate_record, original))
        assert bool(expected_record["valid"])
        for a, b in zip(
            jax.tree.leaves(expected_record["state"]), jax.tree.leaves(original["state"])
        ):
            np.testing.assert_allclose(a, b, **tolerance)

        phase("resident_record_compile_and_certify")
        recording = compile_program(
            raw_record, (theta,), policy=policy, out=args.out / "record", name="whole_record"
        )
        result = recording.certify(
            *inputs,
            alternate_arguments=alternate,
            detailed_trace=args.trace,
            expected_host=jax.device_get(expected_record),
            alternate_expected_host=jax.device_get(alternate_record),
        )

        phase("ordinary_gradient_reference")
        normal_gradient = ordinary(raw_gradient)
        expected_gradient = normal_gradient(theta)
        alternate_gradient = normal_gradient(alternate_theta)
        reference = ordinary(jax.value_and_grad(old_loss))(theta)
        jax.block_until_ready((expected_gradient, alternate_gradient, reference))
        for a, b in zip(expected_gradient, reference):
            np.testing.assert_allclose(a, b, **tolerance)
        write_npz(
            args.out / "ordinary_gradient_reference.npz",
            theta=theta,
            legacy_loss=reference[0],
            legacy_gradient=reference[1],
            fused_loss=expected_gradient[0],
            fused_gradient=expected_gradient[1],
        )

        phase("resident_gradient_compile_and_certify")
        derivative = compile_program(
            raw_gradient,
            (theta,),
            policy=policy,
            out=args.out / "gradient",
            name="whole_value_and_gradient",
        )
        actual = derivative.certify(
            *inputs,
            alternate_arguments=alternate,
            detailed_trace=args.trace,
            expected_host=jax.device_get(expected_gradient),
            alternate_expected_host=jax.device_get(alternate_gradient),
        )
        for a, b in zip(actual, reference):
            np.testing.assert_allclose(a, b, **tolerance)
        write_npz(
            args.out / "numerical_reference.npz",
            theta=theta,
            loss=actual[0],
            gradient=actual[1],
            legacy_loss=reference[0],
            legacy_gradient=reference[1],
            sample_positions=result["history"]["samples"]["position"],
            sample_indices=result["history"]["frame_indices"],
            metrics=result["history"]["metrics"],
        )

        phase("whole_episode_central_difference_ladder")
        ladder = difference_ladder(
            lambda p: forward(*forward.stage(np.asarray(p, dtype))),
            lambda p: derivative(*derivative.stage(np.asarray(p, dtype))),
            np.asarray(theta),
            seed=7,
        )
        write_json(args.out / "gradient_ladder.json", ladder)
        if not ladder["agreement_passed"]:
            raise AssertionError("Whole-episode finite-difference agreement failed")
        # Do not call an unobserved plateau O(h^2). All rows and orders remain.
        if dtype == np.float64 and not ladder["second_order_observed"]:
            raise AssertionError("Float64 ladder lacks a second-order convergence window")

        phase("whole_program_timing")
        for executable in (forward, derivative, recording):
            executable.benchmark(*inputs, repeats=3)
        status.update(
            passed=True,
            cuda_residency_demonstrated=device == "cuda",
            scope="three action intervals plus settling, not a full gait",
        )
    except BaseException as error:
        status.update(exception=repr(error), traceback=traceback.format_exc())
        print(status["traceback"], file=sys.stderr, flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
        for env in environments:
            env.close()
        status.update(completed=True, wall_seconds=time.perf_counter() - start)
        write_json(args.out / "result.json", status)
    return 0 if status["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
