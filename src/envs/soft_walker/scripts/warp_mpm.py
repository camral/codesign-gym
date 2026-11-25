#!/usr/bin/env python3
"""Native MPM bundle creation, replay, recording, and resolved-field gradients.

Only ``export-demo`` imports JAX, and only for construction/controller sampling.
Native replay never calls the JAX simulator. The diagnostic objective is recorded
explicitly; it is NOT the existing hop/travel benchmark objective.
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
import sys
import time
import traceback
from dataclasses import asdict, replace

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
        Array,
    )
    from envs.soft_walker.engine.warp_native.model import Bundle
    from envs.soft_walker.engine.warp_native.solid_model import SolidBundle

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def parse_args() -> argparse.Namespace:
    """Parse the existing command-line options and reject incompatible combinations."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("make-smoke", "export-demo", "replay"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--commands", type=Path)
    p.add_argument("--demo-config", type=Path, help="DemoConfig JSON for explicit export")
    p.add_argument(
        "--controller",
        type=Path,
        help="Saved Fourier coefficients .npy or .npz[controller/parameters]",
    )
    p.add_argument("--driver", choices=("mpm2", "mpm3"), default="mpm2")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--cycles", type=int, default=1)
    p.add_argument("--substeps", type=int)
    p.add_argument("--settling-steps", type=int)
    p.add_argument("--checkpoint-stride", type=int, default=32)
    p.add_argument("--memory-mib", type=int, default=512)
    p.add_argument("--gradient", action="store_true")
    p.add_argument(
        "--krylov-method", choices=("legacy", "adaptive", "adaptive_dynamic"), default=None
    )
    p.add_argument("--compact-nodes", type=int, default=None)
    p.add_argument("--linear-work-reuse", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--block-preconditioner", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--cache-tangent", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--reuse-replay-roots", action="store_true")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--animation", action="store_true")
    p.add_argument("--amplitude", type=float, default=0.8)
    p.add_argument("--lift", type=float, default=0.5)
    p.add_argument("--gait", choices=("alternating", "crawl", "pronk"), default="alternating")
    p.add_argument("--length-ref", type=float, default=0.2)
    p.add_argument("--work-weight", type=float, default=0.0)
    p.add_argument("--velocity-weight", type=float, default=0.0)
    a = p.parse_args()
    if min(a.steps, a.cycles, a.repeats, a.checkpoint_stride, a.memory_mib) < 1:
        p.error("Counts and capacity must be positive")
    if a.mode == "replay" and a.bundle is None:
        p.error("replay requires --bundle")
    return a


def coefficients(path: Path) -> Array:
    """Read a saved controller coefficient array without allowing pickle payloads."""
    import numpy as np

    value = np.load(path, allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        try:
            for key in ("controller", "parameters"):
                if key in value.files:
                    return value[key].copy()
            raise ValueError("Controller archive needs controller or parameters array")
        finally:
            value.close()
    return value


def export_demo(args: argparse.Namespace) -> Report:
    """Assemble and export the demo into a native resolved bundle, without differentiating through the export."""
    import jax
    import jax.numpy as j
    import numpy as np

    from envs.soft_walker.engine.drivers.task import BodyConfig
    from envs.soft_walker.engine.physics import mpm
    from envs.soft_walker.engine.warp_native.mpm_conversion import from_jax
    from envs.soft_walker.presets.gaits import seed_controller
    from envs.soft_walker.presets.locomotion import DemoConfig, build_demo
    from envs.soft_walker.shared import periodic_action

    config = (
        DemoConfig(**json.loads(args.demo_config.read_text()))
        if args.demo_config
        else DemoConfig(
            driver=args.driver, dtype=args.dtype, cycles=args.cycles, gmres_method="fused"
        )
    )
    if config.driver not in ("mpm2", "mpm3"):
        raise ValueError("Only MPM demos can be exported")
    env, embodiment = build_demo(config, device="cpu")
    try:
        # Preparation validates the embodiment but intentionally does not run
        # settling: it must happen in the native whole-episode computation.
        _, model, errors = env._prepare(embodiment)
        if errors:
            raise ValueError(errors)
        model = cast(mpm.Model, model)
        state = mpm.initial_state(model)
        bundle = from_jax(
            model,
            state,
            env._grid,
            cast(BodyConfig, env.config).dynamics,
            dt=config.control_dt / config.substeps,
        )
        params = (
            coefficients(args.controller)
            if args.controller
            else seed_controller(
                config.dim,
                amplitude=args.amplitude,
                lift=args.lift,
                gait=args.gait,
                dtype=config.dtype,
            )
        )
        if params.shape[0] != bundle.channels:
            raise ValueError("Controller channel count disagrees with exported body")
        times = j.arange(config.steps, dtype=config.dtype) * config.control_dt
        commands = np.asarray(
            jax.vmap(lambda t: periodic_action(j.asarray(params), t, config.period))(times)
        )
        bundle.save(args.out / "model.npz")
        np.save(args.out / "commands.npy", commands)
        np.save(args.out / "controller.npy", params)
        protocol: Report = dict(
            substeps=config.substeps,
            settling_steps=round(config.settle_seconds / config.control_dt) * config.substeps,
            demo_config=asdict(config),
            embodiment={k: np.asarray(v).tolist() for k, v in embodiment.items()},
            kind="resolved forward-physics export; no JAX stepping",
            initial_state="unsettled; native episode includes settling",
            objective="Native diagnostic objective only; existing hop/travel objective is NOT translated",
            geometry_derivatives="Resolved arrays only; decoder derivatives are not exported",
        )
        (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2))
        return dict(
            exported=True,
            particles=len(bundle.fields["alpha"]),
            channels=bundle.channels,
            physical_steps=protocol["settling_steps"] + len(commands) * config.substeps,
            note="Construction/export only: this command does not certify a native gait",
        )
    finally:
        env.close()


def preview(recording: Report, bundle: Bundle | SolidBundle, path: Path) -> None:
    """Render saved physical samples with display-only geometry; do not rerun the solver."""
    import numpy as np
    from PIL import Image, ImageDraw

    positions = recording["x"]
    active = bundle.fields["alpha"] > 0
    positions = positions[:, active]
    dim = bundle.config.dim

    # Stable oblique projection; no camera tracking or invented intermediate states.
    def project(x: Array) -> Array:
        return np.stack(
            (
                x[..., 0] + (0.32 * x[..., 2] if dim == 3 else 0),
                x[..., 1] + (0.12 * x[..., 2] if dim == 3 else 0),
            ),
            axis=-1,
        )

    points = project(positions)
    lo = points.min(axis=(0, 1))
    hi = points.max(axis=(0, 1))
    ground = bundle.config.contact_height
    lo[1] = min(lo[1], ground)
    extent = np.maximum(hi - lo, 0.01)
    scale = min(680 / extent[0], 400 / extent[1])

    def pixel(p: Array) -> Array:
        return (int(40 + (p[0] - lo[0]) * scale), int(460 - (p[1] - lo[1]) * scale))

    frames = []
    for i, cloud in enumerate(points):
        image = Image.new("RGB", (760, 520), "white")
        draw = ImageDraw.Draw(image)
        draw.text(
            (20, 12),
            f"Actual native Warp state | t={recording['physical_time'][i]:.5f} s",
            fill="black",
        )
        draw.text((20, 30), "Diagnostic projection; not a certified gait", fill="black")
        if bundle.terrain is None:
            y = pixel(np.array([lo[0], ground]))[1]
            draw.line((20, y, 740, y), fill="black", width=2)
        for p in cloud:
            x, y = pixel(p)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(35, 110, 160))
        frames.append(image)
    times = recording["physical_time"]
    delays = [max(10, round(1000 * t)) for t in np.diff(times)]
    delays.append(delays[-1] if delays else 100)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=delays, loop=0)
    path.with_suffix(".frames.json").write_text(
        json.dumps(
            dict(
                physical_time=times.tolist(),
                stored_state_indices=list(range(len(times))),
                display_delay_ms=delays,
                note="GIF time quantization may slow tiny steps; physical timestamps are authoritative",
            ),
            indent=2,
        )
    )


def main() -> int:
    """Command entry point. Native MPM bundle creation, replay, recording, and resolved-field gradients."""
    args = parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["JAX_ENABLE_X64"] = "true" if args.dtype == "float64" else "false"
    os.environ["JAX_PLATFORMS"] = "cpu"
    status: Report = dict(
        success=False, arguments=vars(args) | {"out": str(args.out)}, started=time.time()
    )
    root = REPO_ROOT
    tracked = list((ENGINE_ROOT / "warp_native").glob("*.py")) + [
        Path(__file__).resolve(),
        SOFT_ROOT / "shared.py",
    ]
    status["source_hashes"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked
    }

    def phase(name: str) -> None:
        (args.out / "phase.json").write_text(
            json.dumps(dict(phase=name, utc=time.time()), indent=2)
        )
        print(name, flush=True)

    try:
        if args.mode == "export-demo":
            phase("Constructing and exporting existing JAX demo, without simulation")
            status["result"] = export_demo(args)
        else:
            import numpy as np

            from envs.soft_walker.engine.warp_native.episode import Episode
            from envs.soft_walker.engine.warp_native.model import Bundle, DeviceModel, uniform_body

            if args.mode == "make-smoke":
                bundle = uniform_body(int(args.driver[-1]), precision=args.dtype)
                commands = np.full((args.steps, 1), 0.2, args.dtype)
                bundle.save(args.out / "model.npz")
                np.save(args.out / "commands.npy", commands)
                (args.out / "protocol.json").write_text(
                    json.dumps(
                        dict(
                            substeps=1,
                            settling_steps=0,
                            note="One-cell material smoke test, not a robot gait",
                        ),
                        indent=2,
                    )
                )
                status["result"] = dict(created=True, particles=len(bundle.fields["alpha"]))
            else:
                bundle = Bundle.load(args.bundle)
                changes = {}
                if args.krylov_method is not None:
                    changes["krylov_method"] = args.krylov_method
                if args.cache_tangent is not None:
                    changes["cache_tangent"] = args.cache_tangent
                for field in ("compact_nodes", "linear_work_reuse", "block_preconditioner"):
                    value = getattr(args, field)
                    if value is not None:
                        changes[field] = value
                if changes:
                    bundle.config = replace(bundle.config, **changes)
                if str(bundle.fields["reference"].dtype) != args.dtype:
                    raise ValueError(
                        "Bundle dtype differs; export/create the requested precision explicitly"
                    )
                command_path = args.commands or args.bundle.parent / "commands.npy"
                commands = np.load(command_path, allow_pickle=False)
                protocol_path = args.bundle.parent / "protocol.json"
                protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
                substeps = (
                    args.substeps if args.substeps is not None else protocol.get("substeps", 1)
                )
                settling = (
                    args.settling_steps
                    if args.settling_steps is not None
                    else protocol.get("settling_steps", 0)
                )
                phase("Allocating native model and block checkpoint storage")
                episode = Episode(
                    DeviceModel(bundle, args.device),
                    commands,
                    substeps=substeps,
                    settling_steps=settling,
                    checkpoint_stride=args.checkpoint_stride,
                    memory_mib=args.memory_mib,
                    length_ref=args.length_ref,
                    velocity_weight=args.velocity_weight,
                    work_weight=args.work_weight,
                    reuse_replay_roots=args.reuse_replay_roots,
                )
                (args.out / "resolved_config.json").write_text(
                    json.dumps(
                        dict(
                            config=bundle.config.to_dict(),
                            substeps=substeps,
                            settling_steps=settling,
                            checkpoint_stride=episode.stride,
                            objective="massmean(-dot(direction, final_x - initial_x)/length_ref + velocity_weight*|final_v|^2) + work_weight*positive_work",
                            direction="world +x",
                            length_ref=args.length_ref,
                            velocity_weight=args.velocity_weight,
                            work_weight=args.work_weight,
                            settling_included_in_loss=True,
                        ),
                        indent=2,
                    )
                )
                phase("Compiling/capturing whole executable and first complete execution")
                first = episode.run(gradient=args.gradient)
                if first["status"] != 0:
                    raise RuntimeError(f"First physical execution failed: {first}")
                phase("Synchronized whole-executable repetitions")
                samples = [episode.run(gradient=args.gradient) for _ in range(args.repeats)]
                if any(s["status"] != 0 for s in samples):
                    raise RuntimeError("A repeated execution failed")
                recording = episode.recording()
                np.savez_compressed(args.out / "trajectory.npz", **recording)
                if args.gradient:
                    gradient = episode.gradients()
                    arrays = {"commands": gradient["commands"]}
                    arrays.update({f"model_{k}": v for k, v in gradient["model"].items()})
                    arrays.update({f"initial_{k}": v for k, v in gradient["initial"].items()})
                    np.savez_compressed(args.out / "gradients.npz", **arrays)
                if args.animation:
                    phase("Rendering actual checkpoint states after execution")
                    preview(recording, bundle, args.out / "replay.gif")
                status["result"] = dict(
                    first=first,
                    repetitions=samples,
                    solver=episode.step.diagnostics(),
                    commands_sha256=hashlib.sha256(command_path.read_bytes()).hexdigest(),
                    bundle_sha256=hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
                    qualification="Not a gait certification; diagnostic objective, fixed resolved topology",
                    cuda_residency="Requires separate --audit validation; no certificate inferred from JIT or graph construction",
                )
        status["success"] = True
    except BaseException:
        status["exception"] = traceback.format_exc()
        print(status["exception"], file=sys.stderr, flush=True)
    finally:
        status["seconds"] = time.time() - status["started"]
        (args.out / "status.json").write_text(json.dumps(status, indent=2, default=str))
        checks = {
            str(p.relative_to(args.out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.out.rglob("*"))
            if p.is_file()
        }
        (args.out / "SHA256SUMS").write_text(
            "".join(f"{value}  {key}\n" for key, value in checks.items())
        )
        shutil.make_archive(str(args.out), "zip", root_dir=args.out.parent, base_dir=args.out.name)
        print("Archive:", str(args.out) + ".zip", flush=True)
    return 0 if status["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
