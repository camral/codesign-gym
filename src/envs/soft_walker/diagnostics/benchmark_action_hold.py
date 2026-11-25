"""Compare repeated native API steps with an equal resident action hold.

Both paths use identical physical timesteps, models, solver tolerances, action
histories, warmup duration, and total simulated time. Only the number of host
API boundaries changes. Timings include observations and reward reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import warp as wp

if __package__ in (None, ""):
    # Direct-file execution must select this checkout before package imports.
    _source_root = str(Path(__file__).resolve().parents[3])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)

import envs.soft_walker.engine as engine
from envs.soft_walker.engine import make_vec
from envs.soft_walker.engine.warp_native.model import uniform_body
from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
from envs.soft_walker.shared import REPO_ROOT as REPO_ROOT, SRC_ROOT as SRC_ROOT, diagnostic_sources

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from envs.soft_walker.engine.types import (
        Array,
    )
    from envs.soft_walker.engine.warp_native.episode import Episode
    from envs.soft_walker.engine.warp_native.model import Bundle, DeviceModel
    from envs.soft_walker.engine.warp_native.solid_episode import SolidEpisode
    from envs.soft_walker.engine.warp_native.solid_model import SolidBundle, SolidDeviceModel
    from envs.soft_walker.engine.warp_native.vector_env import NativeVectorEnv

IDS = {
    "spring": "SoftSpring2D-v0",
    "beam": "SoftBeam3D-v0",
    "fem": "SoftHexFEM3D-v0",
    "mpm2": "SoftMPM2D-v0",
    "mpm3": "SoftMPM3D-v0",
}
SHAPES = {
    "small": {
        "spring": (2, 1),
        "beam": (1, 2, 1),
        "fem": (2, 1, 1),
        "mpm2": (1, 1),
        "mpm3": (1, 1, 1),
    },
    "larger": {
        "spring": (4, 3),
        "beam": (1, 8, 1),
        "fem": (2, 2, 2),
        "mpm2": (4, 3),
        "mpm3": (3, 2, 2),
    },
}

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def make_body(kind: str, preset: str, dtype: DTypeLike, dt: float) -> Bundle | SolidBundle:
    """Build a regular, gravity-free benchmark body with sufficient interpolation support."""
    body: Bundle | SolidBundle
    shape = SHAPES[preset][kind]
    if kind.startswith("mpm"):
        body = uniform_body(int(kind[-1]), cells=shape, particles_per_axis=2, precision=dtype)
        # Include the complete quadratic interpolation support of larger
        # reference bodies, with space for the small commanded deformation.
        body.config = replace(body.config, grid_shape=tuple(max(8, n + 6) for n in shape))
    else:
        body = regular_solid(kind, shape=shape, precision=dtype)
    body.config = replace(body.config, dt=dt, gravity=0.0, contact_enabled=False)
    return body.validate()


def command_schedule(
    periods: int, repeats: int, count: int, channels: int, dtype: DTypeLike, amplitude: float
) -> Array:
    """Generate identical nonzero held commands for paired timing runs, including warmup."""
    # Index zero is a full held-action warmup, identical in the two paths.
    phase = (
        np.arange(1 + periods * repeats)[:, None, None] * 0.11
        + np.arange(count)[None, :, None] * 0.23
        + np.arange(channels)[None, None, :] * 0.41
    )
    return np.asarray(amplitude * np.cos(phase), dtype=dtype)


def source_manifest() -> Report:
    """Capture source digests and Git provenance separately from benchmark timing."""
    source = Path(engine.__file__).resolve().parent
    files = (
        sorted(source.rglob("*.py")) + diagnostic_sources(REPO_ROOT) + [Path(__file__).resolve()]
    )
    hashes = {
        str(path.relative_to(REPO_ROOT)).replace("\\", "/"): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in files
    }
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    git = ["git", "-c", f"safe.directory={REPO_ROOT.as_posix()}"]

    def read_git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                git + list(args), cwd=REPO_ROOT, capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return dict(
        sha256=digest,
        files=hashes,
        git_head=read_git("rev-parse", "HEAD"),
        git_status=read_git("status", "--short", "--untracked-files=no"),
    )


def prepare(
    kind: str, count: int, method: str, args: argparse.Namespace, body: Bundle | SolidBundle
) -> tuple[NativeVectorEnv, Report]:
    """Create/reset one benchmark path and compile its graphs outside the measured interval."""
    kwargs = (
        dict(substeps=1) if method == "api_each_step" else dict(physics_steps_per_action=args.hold)
    )
    # Horizon counts API calls. Neither path should truncate during warmup or
    # the matched physical trajectory, despite their different API cadences.
    calls = args.hold if method == "api_each_step" else 1
    horizon = (1 + args.periods * args.repeats) * calls + 1
    start = time.perf_counter()
    env, fields = make_vec(
        IDS[kind],
        backend="warp",
        num_envs=count,
        execution=args.execution,
        bundle=body,
        dtype=args.dtype,
        device=args.device,
        horizon=horizon,
        energy_weight=0.05,
        **kwargs,
    )
    try:
        key = "density" if kind.startswith("mpm") else "mass"
        scales = (
            np.linspace(0.9, 1.1, count, dtype=args.dtype)
            if count > 1
            else np.ones(1, dtype=args.dtype)
        )
        fields[key] *= scales.reshape((count,) + (1,) * (fields[key].ndim - 1))
        _, rejected, _ = env.reset(embodiment=fields)
        if rejected.any():
            raise RuntimeError(
                f"{method}: reset rejected lanes {np.flatnonzero(rejected).tolist()}"
            )
        device = cast("DeviceModel | SolidDeviceModel", env.envs[0].owner).device
        wp.synchronize_device(device)
        reset_s = time.perf_counter() - start
        start = time.perf_counter()
        for lane in env.envs:
            cast("Episode | SolidEpisode", lane._episode).compile()
        wp.synchronize_device(device)
        return env, dict(
            method=method,
            api_calls_per_action=calls,
            reset_seconds=reset_s,
            compilation_seconds=time.perf_counter() - start,
            repeat_seconds=[],
            action_boundary_traces=[],
        )
    except BaseException:
        env.close()
        raise


def run_periods(
    env: NativeVectorEnv, commands: Array, calls: int, hold: int, start_period: int, dt: float
) -> tuple[float, list[Report]]:
    """Time API work, then summarize/validate the returned transition arrays."""
    device = cast("DeviceModel | SolidDeviceModel", env.envs[0].owner).device
    wp.synchronize_device(device)
    start = time.perf_counter()
    periods = []
    for command in commands:
        results = []
        for _ in range(calls):
            result = env.step(command)
            # Stop at the first failure; never let an autoreset conceal a
            # failed intermediate microstep in the baseline trajectory.
            if result[2].any() or result[3].any():
                raise RuntimeError(
                    "A lane terminated/truncated before the matched physical horizon"
                )
            results.append(result)
        periods.append(results)
    wp.synchronize_device(device)
    seconds = time.perf_counter() - start
    traces = []
    for index, results in enumerate(periods):
        final = results[-1]
        rewards = np.sum([result[1] for result in results], axis=0)
        if not np.all(np.isfinite(final[0])) or not np.all(np.isfinite(rewards)):
            raise RuntimeError("Nonfinite observation or accumulated reward")
        infos = final[4]["per_env"]
        # Whole holds only expose their final internal-step solver counters;
        # do not compare this with the baseline's sum over all microsteps.
        traces.append(
            dict(
                period=start_period + index,
                physical_time_seconds=(start_period + index + 1) * hold * dt,
                observation=np.array(final[0], copy=True),
                reward=rewards,
                terminated=np.array(final[2], copy=True),
                truncated=np.array(final[3], copy=True),
                displacement=np.asarray([info["metrics"]["displacement"] for info in infos]),
                positive_work=np.asarray([info["metrics"]["positive_work"] for info in infos]),
                final_internal_step_iterations=np.asarray(
                    [info["numerics"]["iterations"] for info in infos]
                ),
                final_internal_step_linear_solves=np.asarray(
                    [info["numerics"]["linear_solves"] for info in infos]
                ),
            )
        )
    return seconds, traces


def compare_traces(reference: Array, observed: list[Report], dtype: DTypeLike) -> Report:
    """Assert matching observations, accumulated rewards, and failure flags at shared physical times."""
    if len(reference) != len(observed):
        raise AssertionError("Compared paths contain different numbers of held-action boundaries")
    rtol, atol = (8e-4, 5e-6) if dtype == "float32" else (3e-8, 3e-11)
    worst = dict(observation=0.0, reward=0.0, displacement=0.0, positive_work=0.0)
    for baseline, held in zip(reference, observed):
        assert baseline["period"] == held["period"]
        assert baseline["physical_time_seconds"] == held["physical_time_seconds"]
        for key in worst:
            np.testing.assert_allclose(
                held[key],
                baseline[key],
                rtol=rtol,
                atol=atol,
                err_msg=f"{key} differs at physical time {held['physical_time_seconds']}",
            )
            worst[key] = max(worst[key], float(np.max(np.abs(held[key] - baseline[key]))))
        for key in ("terminated", "truncated"):
            np.testing.assert_array_equal(held[key], baseline[key])
    return dict(
        passed=True,
        compared_boundaries=len(reference),
        rtol=rtol,
        atol=atol,
        max_absolute_difference=worst,
        observations="All native observation components compared; no time component excluded",
    )


def save_traces(path: Path, traces: list[Report]) -> None:
    """Save action-boundary arrays to a compressed NPZ for independent parity review."""
    names = (
        "period",
        "physical_time_seconds",
        "observation",
        "reward",
        "terminated",
        "truncated",
        "displacement",
        "positive_work",
        "final_internal_step_iterations",
        "final_internal_step_linear_solves",
    )
    np.savez_compressed(
        path, **{key: np.asarray([trace[key] for trace in traces]) for key in names}
    )


def run_case(kind: str, count: int, args: argparse.Namespace) -> Report:
    """Measure one model/batch pair at equal physical duration and verify boundary parity."""
    body = make_body(kind, args.preset, args.dtype, args.dt)
    commands = command_schedule(
        args.periods, args.repeats, count, body.channels, args.dtype, args.amplitude
    )
    methods = ("api_each_step", "resident_hold")
    envs, measurements = {}, {}
    try:
        for method in methods:
            envs[method], measurements[method] = prepare(kind, count, method, args, body)
        for method in methods:
            row = measurements[method]
            row["warmup_seconds"], row["warmup_trace"] = run_periods(
                envs[method], commands[:1], row["api_calls_per_action"], args.hold, 0, args.dt
            )
        warmup_parity = compare_traces(
            measurements[methods[0]]["warmup_trace"],
            measurements[methods[1]]["warmup_trace"],
            args.dtype,
        )
        measurement_order = []
        repeat_parity = []
        for repeat in range(args.repeats):
            # Alternate ordering to reduce systematic clock/thermal bias.
            order = methods if repeat % 2 == 0 else methods[::-1]
            measurement_order.append(list(order))
            offset = 1 + repeat * args.periods
            recent = {}
            for method in order:
                row = measurements[method]
                seconds, traces = run_periods(
                    envs[method],
                    commands[offset : offset + args.periods],
                    row["api_calls_per_action"],
                    args.hold,
                    offset,
                    args.dt,
                )
                row["repeat_seconds"].append(seconds)
                row["action_boundary_traces"].extend(traces)
                recent[method] = traces
            repeat_parity.append(compare_traces(recent[methods[0]], recent[methods[1]], args.dtype))
        rows = []
        physical_steps = args.periods * args.hold
        for method in methods:
            row = measurements[method]
            traces = row.pop("action_boundary_traces")
            row.pop("warmup_trace")
            median = float(np.median(row["repeat_seconds"]))
            row.update(
                warmup_physics_steps_per_env=args.hold,
                measured_physics_steps_per_env_per_repeat=physical_steps,
                measured_simulated_seconds_per_env_per_repeat=physical_steps * args.dt,
                total_measured_api_calls=args.periods * args.repeats * row["api_calls_per_action"],
                median_batch_action_seconds=median / args.periods,
                median_physics_steps_per_second=count * physical_steps / median,
                median_real_time_factor_per_env=physical_steps * args.dt / median,
                final_internal_step_linear_solves_total=int(
                    sum(np.sum(t["final_internal_step_linear_solves"]) for t in traces)
                ),
                final_internal_step_iterations_total=int(
                    sum(np.sum(t["final_internal_step_iterations"]) for t in traces)
                ),
                accumulated_rewards=np.sum([t["reward"] for t in traces], axis=0).tolist(),
                final_physical_time_seconds=traces[-1]["physical_time_seconds"],
            )
            if args.trace:
                name = f"{kind}_batch{count}_{method}_boundaries.npz"
                save_traces(args.out / name, traces)
                row["action_boundary_trace"] = name
            rows.append(row)
        baseline, resident = rows
        paired = np.asarray(baseline["repeat_seconds"]) / np.asarray(resident["repeat_seconds"])
        return dict(
            physics=kind,
            preset=args.preset,
            cell_shape=SHAPES[args.preset][kind],
            num_envs=count,
            model_sites=len(body.fields["reference"]),
            channels=body.channels,
            solver_capacity=cast(
                "Episode | SolidEpisode", envs[methods[0]].envs[0]._episode
            ).step.capacity,
            execution=args.execution,
            effective_execution="parallel"
            if cast("DeviceModel | SolidDeviceModel", envs[methods[0]].envs[0].owner).device.is_cuda
            and args.execution != "serial"
            else "serial",
            dt=args.dt,
            physics_steps_per_action=args.hold,
            control_period_seconds=args.dt * args.hold,
            periods_per_repeat=args.periods,
            repeats=args.repeats,
            measured_physical_horizon_per_env_seconds=args.periods
            * args.repeats
            * args.hold
            * args.dt,
            total_physical_horizon_including_warmup_seconds=(1 + args.periods * args.repeats)
            * args.hold
            * args.dt,
            body_config=asdict(body.config),
            warmup_parity=warmup_parity,
            repeat_parity=repeat_parity,
            measurement_order=measurement_order,
            rows=rows,
            paired_speedups=paired.tolist(),
            median_paired_speedup=float(np.median(paired)),
            ratio_of_median_times=float(
                np.median(baseline["repeat_seconds"]) / np.median(resident["repeat_seconds"])
            ),
        )
    finally:
        for env in envs.values():
            env.close()


def main() -> int:
    """Command entry point. Compare repeated native API steps with an equal resident action hold."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--physics", nargs="+", choices=(*IDS, "all"), default=["all"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 16])
    parser.add_argument("--preset", choices=tuple(SHAPES), default="larger")
    parser.add_argument("--execution", choices=("auto", "serial", "parallel"), default="auto")
    parser.add_argument("--hold", type=int, default=10)
    parser.add_argument("--periods", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--amplitude", type=float, default=0.16)
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Save matched action-boundary state/reward traces; not a kernel profiler",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/action_hold"))
    args = parser.parse_args()
    if min(*args.batch_sizes, args.hold, args.periods, args.repeats) < 1:
        parser.error("Batch sizes, hold, periods, and repeats must be positive")
    if (
        not np.isfinite(args.dt)
        or args.dt <= 0
        or not np.isfinite(args.amplitude)
        or not 0 < args.amplitude <= 1
    ):
        parser.error("dt must be positive and finite; amplitude must be in (0, 1]")
    args.out.mkdir(parents=True, exist_ok=True)
    wp.init()
    device = wp.get_device(args.device)
    selected = list(IDS) if "all" in args.physics else list(dict.fromkeys(args.physics))
    report: Report = dict(
        schema_version=1,
        benchmark="Matched native action-hold API boundary amortization",
        python=platform.python_version(),
        numpy=np.__version__,
        warp=wp.__version__,
        device=str(device),
        device_name=device.name,
        dtype=args.dtype,
        source=source_manifest(),
        scope="Warm synchronized forward vector API work, including observations and reward reporting. "
        "Baseline repeats each command at substeps=1; candidate holds it inside one resident episode. "
        "Both have identical physical timestep, command sequence, warmup, and physical horizon. "
        "Setup/compilation/warmup, parity validation, and trace serialization excluded. "
        "Transition termination checks and API-loop bookkeeping included. "
        "Gravity and contact disabled; ordinary regular bodies, not walking or backward-gradient performance.",
        counter_scope="Solver counters compare only the final internal physics step at each shared action boundary.",
        cases=[],
        failures=[],
        complete=False,
    )
    path = args.out / "action_hold.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for kind in selected:
        for count in args.batch_sizes:
            try:
                case = run_case(kind, count, args)
                report["cases"].append(case)
                before, after = case["rows"]
                print(
                    f"{kind} batch={count} sites={case['model_sites']}: "
                    f"{before['median_batch_action_seconds'] * 1000:.3f} -> "
                    f"{after['median_batch_action_seconds'] * 1000:.3f} ms/action; "
                    f"{case['median_paired_speedup']:.3f}x paired speedup; parity passed",
                    flush=True,
                )
            except Exception as exc:
                report["failures"].append(
                    dict(
                        physics=kind,
                        num_envs=count,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                )
                print(f"{kind} batch={count}: FAILED: {exc}", flush=True)
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["complete"] = True
    report["passed"] = not report["failures"]
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(path, flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
