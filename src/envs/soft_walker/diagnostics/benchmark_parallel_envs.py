"""Synchronized native API throughput with serial/parallel numerical parity.

This measures complete vector step calls, including NumPy observations. It is
not a solver-only benchmark or a claim about fused multi-body CUDA kernels.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import warp as wp

if __package__ in (None, ""):
    # Direct-file execution must select this checkout before package imports.
    _source_root = str(Path(__file__).resolve().parents[3])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)

from envs.soft_walker.engine import make_vec
from envs.soft_walker.engine.warp_native.model import uniform_body
from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
from envs.soft_walker.shared import REPO_ROOT as REPO_ROOT, SRC_ROOT as SRC_ROOT

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from envs.soft_walker.engine.types import (
        Array,
    )
    from envs.soft_walker.engine.warp_native.episode import Episode
    from envs.soft_walker.engine.warp_native.model import Bundle, DeviceModel
    from envs.soft_walker.engine.warp_native.solid_episode import SolidEpisode
    from envs.soft_walker.engine.warp_native.solid_model import SolidBundle, SolidDeviceModel

IDS = {
    "spring": "SoftSpring2D-v0",
    "beam": "SoftBeam3D-v0",
    "fem": "SoftHexFEM3D-v0",
    "mpm2": "SoftMPM2D-v0",
    "mpm3": "SoftMPM3D-v0",
}

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def body_for(kind: str, dtype: DTypeLike) -> Bundle | SolidBundle:
    """Build the small resolved physical body used for an isolated batching comparison."""
    body: Bundle | SolidBundle
    if kind.startswith("mpm"):
        body = uniform_body(int(kind[-1]), precision=dtype)
    else:
        shape = (1, 2, 1) if kind == "beam" else None
        body = regular_solid(kind, precision=dtype, shape=shape)
    body.config = replace(body.config, gravity=0.0, contact_enabled=False, dt=1e-4)
    return body


def check_result(result: tuple[Any, ...]) -> None:
    """Reject nonfinite or failed transitions before including their timings in a throughput result."""
    if result[2].any() or result[3].any():
        raise RuntimeError("Benchmark lane failed or reached its horizon; timings are invalid")
    if not all(np.all(np.isfinite(value)) for value in result[:2]):
        raise RuntimeError("Nonfinite benchmark observation or reward")


def measure(
    kind: str, count: int, mode: str, args: argparse.Namespace, commands: Array
) -> tuple[Report, list[tuple[Any, ...]]]:
    """Time the requested batch execution mode and retain its transition evidence."""
    body = body_for(kind, args.dtype)
    construction_start = time.perf_counter()
    env, fields = make_vec(
        IDS[kind],
        backend="warp",
        num_envs=count,
        execution=mode,
        bundle=body,
        dtype=args.dtype,
        device=args.device,
        substeps=args.substeps,
        horizon=2 + args.steps * args.repeats,
    )
    # Distinct material values ensure the benchmark exercises independent lane
    # storage, not repeated references to one physical model.
    key = "density" if kind.startswith("mpm") else "mass"
    fields[key] *= np.linspace(0.9, 1.1, count, dtype=args.dtype).reshape(
        (count,) + (1,) * (fields[key].ndim - 1)
    )
    try:
        _, rejected, _ = env.reset(embodiment=fields)
        if rejected.any():
            raise RuntimeError("Benchmark reset rejected a lane")
        device = cast("DeviceModel | SolidDeviceModel", env.envs[0].owner).device
        wp.synchronize_device(device)
        reset_s = time.perf_counter() - construction_start
        start = time.perf_counter()
        for lane in env.envs:
            cast("Episode | SolidEpisode", lane._episode).compile()
        wp.synchronize_device(device)
        compile_s = time.perf_counter() - start
        # Also excludes first executor binding, command uploads, and cold caches.
        start = time.perf_counter()
        check_result(env.step(commands[0]))
        wp.synchronize_device(device)
        warmup_s = time.perf_counter() - start
        durations, transitions = [], []
        batch_execution_seconds = []
        solver_work = []
        for repeat in range(args.repeats):
            wp.synchronize_device(device)
            start = time.perf_counter()
            outputs = []
            for tick in range(args.steps):
                outputs.append(env.step(commands[1 + repeat * args.steps + tick]))
            wp.synchronize_device(device)
            durations.append(time.perf_counter() - start)
            # Verify after the measured synchronized API interval.
            for result in outputs:
                check_result(result)
                transitions.append(tuple(np.array(value, copy=True) for value in result[:4]))
                first_info = result[4]["per_env"][0]
                if "batch_execution_seconds" in first_info:
                    batch_execution_seconds.append(float(first_info["batch_execution_seconds"]))
                for info in result[4]["per_env"]:
                    numerics = info["numerics"]
                    solver_work.append(
                        {
                            key: int(cast("int", numerics.get(key, 0)))
                            for key in ("iterations", "linear_solves", "backtracks")
                        }
                    )
        work_summary = {
            key: dict(
                total=sum(row[key] for row in solver_work),
                minimum=min(row[key] for row in solver_work),
                maximum=max(row[key] for row in solver_work),
                nonzero_lane_steps=sum(row[key] > 0 for row in solver_work),
            )
            for key in ("iterations", "linear_solves", "backtracks")
        }
        median = float(np.median(durations))
        row: Report = dict(
            physics=kind,
            num_envs=count,
            execution=mode,
            effective_execution="parallel" if device.is_cuda and mode != "serial" else "serial",
            precision=args.dtype,
            device=str(device),
            model_sites=len(body.fields["reference"]),
            dt=body.config.dt,
            substeps=args.substeps,
            timed_steps_per_repeat=args.steps,
            repeats=args.repeats,
            reset_seconds=reset_s,
            compilation_seconds=compile_s,
            warmup_seconds=warmup_s,
            repeat_seconds=durations,
            median_batch_step_seconds=median / args.steps,
            median_env_steps_per_second=count * args.steps / median,
            median_env_physics_steps_per_second=count * args.steps * args.substeps / median,
            batch_graph_execution_seconds=batch_execution_seconds,
            solver_work_at_final_substep=work_summary,
            actuated_linear_solves_observed=work_summary["linear_solves"]["total"] > 0,
            solver_tolerances=dict(
                atol64=body.config.atol64,
                atol32=body.config.atol32,
                linear_rtol64=body.config.linear_rtol64,
                linear_rtol32=body.config.linear_rtol32,
            ),
        )
        return row, transitions
    finally:
        env.close()


def main() -> None:
    """Command entry point. Synchronized native API throughput with serial/parallel numerical parity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--physics", nargs="+", choices=tuple(IDS), default=["spring"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("runs/parallel_envs"))
    args = parser.parse_args()
    if min(*args.batch_sizes, args.steps, args.substeps, args.repeats) < 1:
        parser.error("Counts must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    wp.init()
    device = wp.get_device(args.device)
    modes = ("serial", "parallel") if device.is_cuda else ("serial", "auto")
    report: Report = dict(
        schema_version=1,
        benchmark="native vector API parallel lane throughput",
        python=platform.python_version(),
        numpy=np.__version__,
        warp=wp.__version__,
        device=str(device),
        device_name=device.name,
        scope="Warm forward API step; includes host NumPy observations and metrics. "
        "Reset, CUDA graph capture, compilation, and one warmup step excluded. "
        "Small bodies without terrain contact; not locomotion or gradient throughput.",
        rows=[],
        comparisons=[],
        complete=False,
    )
    path = args.out / "parallel_envs.json"
    rtol, atol = (6e-4, 3e-6) if args.dtype == "float32" else (2e-8, 2e-11)
    for kind in args.physics:
        channels = body_for(kind, args.dtype).channels
        for count in args.batch_sizes:
            steps = 1 + args.steps * args.repeats
            phase = np.arange(steps)[:, None, None] * 0.11 + np.arange(count)[None, :, None] * 0.23
            phase = phase + np.arange(channels)[None, None, :] * 0.41
            commands = np.asarray(0.16 * np.cos(phase), dtype=args.dtype)
            reference = None
            pair = []
            for mode in modes:
                row, transitions = measure(kind, count, mode, args, commands)
                if reference is None:
                    reference = transitions
                else:
                    for observed, expected in zip(transitions, reference):
                        np.testing.assert_allclose(observed[0], expected[0], rtol=rtol, atol=atol)
                        np.testing.assert_allclose(observed[1], expected[1], rtol=rtol, atol=atol)
                        np.testing.assert_array_equal(observed[2], expected[2])
                        np.testing.assert_array_equal(observed[3], expected[3])
                row["serial_trajectory_parity"] = True
                pair.append(row)
                report["rows"].append(row)
                path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(
                    f"{kind} batch={count} {mode}: "
                    f"{row['median_batch_step_seconds'] * 1000:.3f} ms/batch, "
                    f"{row['median_env_steps_per_second']:.2f} env-steps/s",
                    flush=True,
                )
            report["comparisons"].append(
                dict(
                    physics=kind,
                    num_envs=count,
                    parallel_over_serial_speedup=pair[0]["median_batch_step_seconds"]
                    / pair[1]["median_batch_step_seconds"],
                    comparison_mode=modes[1],
                    parity=True,
                )
            )
    report["complete"] = True
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
