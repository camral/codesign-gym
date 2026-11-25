#!/usr/bin/env python3
"""Build/export/replay native Spring2D, Beam3D and HexFEM3D experiments.

Only export-demo imports JAX. Simulation and its whole implicit adjoint are
native Warp. A recorded animation is a diagnostic, not a walking qualification.
"""

from __future__ import annotations

import argparse

# Resolve integrated imports independently of the caller's working directory.
import sys as _script_sys
from pathlib import Path, Path as _ScriptPath
from typing import TYPE_CHECKING, Any, cast

_source_root = str(_ScriptPath(__file__).resolve().parents[3])
if _source_root not in _script_sys.path:
    _script_sys.path.insert(0, _source_root)
_script_dir = str(_ScriptPath(__file__).resolve().parent)
if _script_dir not in _script_sys.path:
    _script_sys.path.insert(0, _script_dir)
import hashlib
import json
import os
import shutil
import time
import traceback
from dataclasses import asdict

import numpy as np

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


def write(path: PathLike, data: Any) -> None:
    """Serialize evidence to the requested path using this command's existing JSON encoder."""
    Path(path).write_text(json.dumps(data, indent=2, default=str) + "\n")


def source_hashes(root: Path) -> dict[str, str]:
    """Hash the source files used to reproduce this command's evidence."""
    files = [p for folder in (ENGINE_ROOT, SCRIPTS_ROOT) for p in sorted(folder.rglob("*.py"))]
    files.append(SOFT_ROOT / "shared.py")
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def parser() -> argparse.ArgumentParser:
    """Build the command-line parser without constructing a simulator or running physics."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["make-smoke", "export-demo", "replay"])
    p.add_argument("--physics", choices=["spring", "beam", "fem"], default="spring")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--commands", type=Path)
    p.add_argument("--demo-config", type=Path)
    p.add_argument("--controller", type=Path)
    p.add_argument("--shape", nargs="+", type=int)
    p.add_argument("--steps", type=int, default=16, help="control intervals for make-smoke only")
    p.add_argument("--dt", type=float, default=0.0005, help="physical step for make-smoke only")
    p.add_argument("--cycles", type=int, default=1, help="export-demo duration only")
    p.add_argument("--amplitude", type=float, default=0.4)
    p.add_argument("--lift", type=float, default=0.5)
    p.add_argument(
        "--gait", choices=["alternating", "crawl", "trot", "bound"], default="alternating"
    )
    p.add_argument("--substeps", type=int)
    p.add_argument("--settling-steps", type=int)
    p.add_argument("--checkpoint-stride", type=int, default=16)
    p.add_argument("--memory-mib", type=int, default=512)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--gradient", action="store_true")
    p.add_argument("--animation", action="store_true")
    p.add_argument("--audit", action="store_true")
    p.add_argument("--length-ref", type=float, default=0.2)
    p.add_argument("--work-weight", type=float, default=0.0)
    p.add_argument("--velocity-weight", type=float, default=0.0)
    return p


def export_demo(args: argparse.Namespace) -> Report:
    """Assemble and export the demo into a native resolved bundle, without differentiating through the export."""
    os.environ["JAX_ENABLE_X64"] = "1" if args.dtype == "float64" else "0"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp

    from envs.soft_walker.engine.physics import beam, fem, spring
    from envs.soft_walker.engine.warp_native.solid_conversion import from_jax
    from envs.soft_walker.presets.gaits import seed_controller
    from envs.soft_walker.presets.locomotion import DemoConfig, build_demo
    from envs.soft_walker.shared import periodic_action

    cfg = (
        DemoConfig(**json.loads(args.demo_config.read_text()))
        if args.demo_config
        else DemoConfig(
            driver=args.physics,
            dtype=args.dtype,
            cycles=args.cycles,
            gmres_method="fused",
            spring_solver="gmres",
        )
    )
    from envs.soft_walker.engine.drivers.task import BodyConfig

    if cfg.driver != args.physics or cfg.dtype != args.dtype:
        raise ValueError("Demo-config driver/precision must match the explicit command")
    if cfg.actuation_model != "active_strain":
        raise ValueError(
            "These JAX Lagrangian models implement active strain, not the MPM active-stress law"
        )
    env, design = build_demo(cfg, device="cpu")
    try:
        _, model, errors = env._prepare(design)
        if errors:
            raise ValueError(errors)
        module = {"spring": spring, "beam": beam, "fem": fem}[args.physics]
        initial = module.initial_state(model)
        dynamics = env.config if args.physics == "spring" else cast(BodyConfig, env.config).dynamics
        bundle = from_jax(
            args.physics,
            cast(spring.Model | beam.Model | fem.Model, model),
            initial,
            dynamics,
            dt=cfg.control_dt / cfg.substeps,
        )
        if args.controller:
            data = np.load(args.controller, allow_pickle=False)
            if isinstance(data, np.lib.npyio.NpzFile):
                try:
                    key = next((k for k in ("controller", "parameters") if k in data.files), None)
                    if key is None:
                        raise ValueError("Controller NPZ needs controller or parameters")
                    coefficients = data[key].copy()
                finally:
                    data.close()
            else:
                coefficients = data
        else:
            coefficients = seed_controller(
                cfg.dim, amplitude=args.amplitude, lift=args.lift, gait=args.gait, dtype=cfg.dtype
            )
        times = jnp.arange(cfg.steps, dtype=cfg.dtype) * cfg.control_dt
        commands = np.asarray(
            jax.vmap(lambda t: periodic_action(jnp.asarray(coefficients), t, cfg.period))(times)
        )
        if commands.shape[1] != bundle.channels:
            raise ValueError("Controller channels differ from decoded embodiment")
        bundle.save(args.out / "model.npz")
        np.save(args.out / "commands.npy", commands)
        np.save(args.out / "controller.npy", coefficients)
        protocol: Report = dict(
            substeps=cfg.substeps,
            settling_steps=round(cfg.settle_seconds / cfg.control_dt) * cfg.substeps,
            model="resolved original JAX model; native backward Euler",
            geometry_derivatives="resolved arrays, not the original decoder",
            demo_config=asdict(cfg),
            qualification="unvalidated native gait seed",
        )
        write(args.out / "protocol.json", protocol)
        return dict(
            exported=True,
            nodes=len(bundle.fields["mass"]),
            interactions=len(bundle.fields["entities"]),
            physical_steps=protocol["settling_steps"] + len(commands) * cfg.substeps,
        )
    finally:
        env.close()


def make_smoke(args: argparse.Namespace) -> Report:
    """Build a small resolved body/control fixture for the native backend smoke test."""
    from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
    from envs.soft_walker.engine.warp_native.solid_model import SolidConfig

    d = 2 if args.physics == "spring" else 3
    shape = args.shape or ((2, 1) if d == 2 else (1, 2, 1) if args.physics == "beam" else (2, 1, 1))
    cfg = SolidConfig(
        physics=args.physics,
        grid_shape=(4,) * d,
        origin=(0.0,) * d,
        dt=args.dt,
        gravity=9.81,
        contact_enabled=True,
        contact_stiffness=1.0 if args.physics == "spring" else 1e6,
        linear_work_reuse=True,
    )
    bundle = regular_solid(args.physics, shape=tuple(shape), precision=args.dtype, config=cfg)
    t = np.arange(args.steps) * args.dt
    commands = np.stack(
        [
            np.tanh(args.amplitude * np.sin(2 * np.pi * (4 * t + c / 2)))
            for c in range(bundle.channels)
        ],
        axis=1,
    ).astype(args.dtype)
    bundle.save(args.out / "model.npz")
    np.save(args.out / "commands.npy", commands)
    write(
        args.out / "protocol.json",
        dict(
            substeps=1, settling_steps=0, qualification="actuated block smoke case, not locomotion"
        ),
    )
    return dict(
        created=True, physics=args.physics, nodes=len(bundle.fields["mass"]), steps=args.steps
    )


def replay(args: argparse.Namespace) -> Report:
    """Execute the selected saved/native design and record its trajectory, status and timings."""
    import warp as wp

    from envs.soft_walker.engine.warp_native.solid_episode import SolidEpisode
    from envs.soft_walker.engine.warp_native.solid_model import SolidBundle, SolidDeviceModel

    if args.bundle is None:
        raise ValueError("replay requires --bundle")
    bundle = SolidBundle.load(args.bundle)
    if bundle.config.physics != args.physics or bundle.precision != args.dtype:
        raise ValueError("Bundle physics/precision must match; no silent conversion")
    commands = np.load(args.commands or args.bundle.parent / "commands.npy", allow_pickle=False)
    protocol_path = args.bundle.parent / "protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    substeps = args.substeps if args.substeps is not None else protocol.get("substeps")
    settling = (
        args.settling_steps if args.settling_steps is not None else protocol.get("settling_steps")
    )
    if substeps is None or settling is None:
        raise ValueError("Provide protocol.json or explicit --substeps and --settling-steps")
    owner = SolidDeviceModel(bundle, args.device)
    episode = SolidEpisode(
        owner,
        commands,
        substeps=substeps,
        settling_steps=settling,
        checkpoint_stride=args.checkpoint_stride,
        memory_mib=args.memory_mib,
        length_ref=args.length_ref,
        work_weight=args.work_weight,
        velocity_weight=args.velocity_weight,
    )
    first = episode.run(args.gradient)
    if first["status"] != 0 or (args.gradient and first["adjoint_status"] != 0):
        write(args.out / "first_failure.json", first)
        np.savez_compressed(args.out / "failed_recording.npz", **episode.recording())
        raise RuntimeError("First whole episode failed; no valid gait/gradient is claimed")
    samples = [episode.run(args.gradient) for _ in range(args.repeats)]
    if any(s["status"] != 0 or (args.gradient and s["adjoint_status"] != 0) for s in samples):
        raise RuntimeError("A repeated whole episode failed")
    recording = episode.recording()
    np.savez_compressed(args.out / "trajectory.npz", **recording)
    bundle.save(args.out / "model.npz")
    np.save(args.out / "commands.npy", commands)
    write(args.out / "protocol.json", dict(protocol, substeps=substeps, settling_steps=settling))
    if args.gradient:
        g = episode.gradients()
        arrays = {"commands": g["commands"]}
        arrays.update(
            {f"{group}_{k}": v for group in ("model", "initial") for k, v in g[group].items()}
        )
        np.savez_compressed(args.out / "gradients.npz", **arrays)
    audit = None
    if args.audit:
        if not owner.device.is_cuda:
            raise ValueError("--audit requires CUDA")
        from envs.soft_walker.engine.execution.cupti import DriverAudit

        stream = wp.get_stream(owner.device)
        wp.synchronize_stream(stream)
        with DriverAudit() as observer:
            wp.capture_launch(
                episode.graphs["gradient" if args.gradient else "forward"], stream=stream
            )
            wp.synchronize_stream(stream)
        audit = observer.report()
        write(args.out / "submission_audit.json", audit)
        if not audit["passed"]:
            raise RuntimeError("Whole-program CUDA submission audit failed")
    if args.animation:
        from PIL import Image

        from envs.soft_walker.engine.warp_native.solid_visualization import render_state

        frames = [
            Image.fromarray(
                render_state(
                    x,
                    bundle.fields["reference"],
                    bundle.fields["entities"],
                    label=f"{args.physics}: actual simulation t={t:.4f}s; unqualified gait",
                )
            )
            for x, t in zip(recording["x"], recording["physical_time"])
        ]
        if len(frames) > 1:
            durations = np.maximum(
                20, np.rint(1000 * np.diff(recording["physical_time"])).astype(int)
            ).tolist()
            frames[0].save(
                args.out / "replay.gif",
                save_all=True,
                append_images=frames[1:],
                duration=durations + [durations[-1]],
                loop=0,
            )
        else:
            frames[0].save(args.out / "replay.png")
        write(
            args.out / "frames.json",
            dict(physical_timestamps=recording["physical_time"].tolist(), interpolated=False),
        )
    return dict(
        first=first,
        warm_samples=samples,
        warm_median_seconds=float(np.median([s["execution_seconds"] for s in samples])),
        diagnostics=episode.step.diagnostics(),
        audit=audit,
        qualification="Numerically valid replay; walking/airborne criteria were not evaluated",
        objective="mass-mean negative post-settling displacement / length_ref + velocity_weight*mean(v²) + work_weight*positive_work",
    )


def main() -> int:
    """Command entry point. Build/export/replay native Spring2D, Beam3D and HexFEM3D experiments."""
    args = parser().parse_args()
    if min(args.steps, args.cycles, args.repeats, args.checkpoint_stride, args.memory_mib) < 1:
        raise ValueError("Counts must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    root = REPO_ROOT
    write(
        args.out / "manifest.json",
        dict(
            arguments=vars(args),
            source_sha256=source_hashes(root),
            offline_spaces=os.environ.get("CODESIGN_OFFLINE_SPACES") == "1",
        ),
    )
    start = time.perf_counter()
    result: Report = {"success": False}
    try:
        result.update(
            {"make-smoke": make_smoke, "export-demo": export_demo, "replay": replay}[args.mode](
                args
            )
        )
        result["success"] = True
    except BaseException:
        result["exception"] = traceback.format_exc()
        print(result["exception"], flush=True)
    finally:
        result["total_seconds"] = time.perf_counter() - start
        write(args.out / "result.json", result)
        sums = {
            str(p.relative_to(args.out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.out.rglob("*"))
            if p.is_file()
        }
        write(args.out / "file_hashes.json", sums)
        shutil.make_archive(str(args.out), "zip", args.out.parent, args.out.name)
        print(json.dumps(result, indent=2, default=str), flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
