#!/usr/bin/env python3
"""One isolated precision/device performance experiment; no historical test sweep.

Warm whole-episode timings exclude compilation, auditing, and final downloads.
Ablations keep physics, tolerance and maximum iteration budgets fixed. CUDA
submission auditing is separate from timed calls and must not overlap Nsight.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, cast

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
    from pathlib import Path

    from envs.soft_walker.engine.physics import mpm
    from envs.soft_walker.engine.types import (
        Array,
        HostArray,
        PathLike,
    )
    from envs.soft_walker.engine.warp_native.audit_types import (
        NativeSubmissionReport,
        ResolvedParameters,
    )
    from envs.soft_walker.engine.warp_native.episode import Episode
    from envs.soft_walker.engine.warp_native.model import Bundle

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def write(path: PathLike, value: Any) -> None:
    """Serialize evidence to the requested path using this command's existing JSON encoder."""
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse the existing command-line options and reject incompatible combinations."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("native", "jax", "ladder"), default="native")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    p.add_argument("--dim", choices=(2, 3), type=int, default=2)
    p.add_argument("--cells", type=int, default=1)
    p.add_argument(
        "--compact-nodes",
        type=int,
        default=0,
        help="Explicit live-node capacity for compact/all variants; overflow fails",
    )
    p.add_argument("--steps", type=int)
    p.add_argument("--settling", type=int)
    p.add_argument("--substeps", type=int)
    p.add_argument("--checkpoint-stride", type=int, default=16)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--commands", type=Path)
    p.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "legacy",
            "adaptive",
            "cached",
            "optimized",
            "dynamic",
            "dynamic_cached",
            "dynamic_optimized",
            "linear_reuse",
            "compact",
            "compact_reuse",
            "block",
            "all",
        ),
        default=["adaptive", "optimized"],
    )
    p.add_argument("--gradient", action="store_true")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--audit", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if any(v in ("compact", "compact_reuse", "all") for v in a.variants) and a.compact_nodes < 1:
        p.error("compact variants require --compact-nodes > 0")
    a = resolve_protocol(a)
    if min(a.cells, a.steps, a.substeps, a.repeats, a.checkpoint_stride) < 1 or a.settling < 0:
        p.error("Invalid counts")
    if a.audit and a.device != "cuda":
        p.error("--audit requires CUDA")
    if a.commands and not a.bundle:
        p.error("--commands requires --bundle")
    return a


def resolve_protocol(args: argparse.Namespace) -> argparse.Namespace:
    """Keep exported duration/settling intact unless explicitly overridden.

    Reading a command-table shape is host-side experiment preparation, never
    part of simulation. A missing protocol must not silently remove settling.
    """
    if args.bundle is not None:
        import numpy as np

        path = args.commands or args.bundle.parent / "commands.npy"
        commands = np.load(path, allow_pickle=False, mmap_mode="r")
        if commands.ndim != 2:
            raise ValueError("Expected commands[interval, channel]")
        args.steps = len(commands) if args.steps is None else args.steps
        protocol_path = args.bundle.parent / "protocol.json"
        protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
        if args.settling is None:
            args.settling = protocol.get("settling_steps", protocol.get("settling"))
        if args.substeps is None:
            args.substeps = protocol.get("substeps")
        if args.settling is None or args.substeps is None:
            raise ValueError("Bundle needs protocol.json or explicit --settling and --substeps")
        if args.steps != len(commands):
            raise ValueError("--steps must match the whole command table; no implicit truncation")
    else:
        args.steps = 3 if args.steps is None else args.steps
        args.settling = 1 if args.settling is None else args.settling
        args.substeps = 1 if args.substeps is None else args.substeps
    return args


def fixture(args: argparse.Namespace) -> tuple[Bundle, HostArray]:
    """Construct the benchmark model and commands from the resolved physical protocol."""
    import numpy as np

    from envs.soft_walker.diagnostics.warp.checks import contact_fixture
    from envs.soft_walker.engine.warp_native.model import Bundle, uniform_body

    if args.bundle:
        bundle = Bundle.load(args.bundle)
        if bundle.config.dim != args.dim:
            raise ValueError("Bundle dimension differs from --dim")
        command_path = args.commands or args.bundle.parent / "commands.npy"
        commands = np.load(command_path, allow_pickle=False)
        if str(bundle.fields["reference"].dtype) != args.dtype:
            raise ValueError("Bundle precision differs; no silent precision conversion")
        if len(commands) != args.steps:
            raise ValueError(
                "--steps must equal the supplied command-table length; no silent truncation"
            )
    else:
        bundle = contact_fixture(args.dim, args.dtype)
        if args.cells != 1:
            large = uniform_body(args.dim, cells=(args.cells,) * args.dim, precision=args.dtype)
            bundle.fields = large.fields
            bundle.fields["alpha"].fill(0.8)
            bundle.config = replace(bundle.config, grid_shape=(max(8, args.cells + 6),) * args.dim)
            n, d = len(bundle.fields["alpha"]), args.dim
            bundle.initial = dict(
                x=bundle.fields["reference"].copy(),
                v=np.zeros((n, d), args.dtype),
                F=np.tile(np.eye(d, dtype=args.dtype), (n, 1, 1)),
                C=np.zeros((n, d, d), args.dtype),
                actuator=np.full(1, 0.02, args.dtype),
                ledger=np.zeros(4, args.dtype),
                status=np.zeros(1, np.int32),
            )
            bundle.initial["v"][:, 0] = 0.02
        commands = np.resize(np.array([0.4, -0.2, 0.15], args.dtype), (args.steps, bundle.channels))
    bundle.validate()
    if bundle.initial is None:
        raise ValueError("The paired experiment requires an explicit resolved initial state")
    return bundle, commands


def scalar_arrays(episode: Episode, gradient: Array) -> dict[str, HostArray]:
    """Download the selected native result/gradient arrays for evidence comparison."""
    arrays = {"loss": episode.loss.numpy()}
    for key in ("x", "v", "F", "C", "actuator", "ledger"):
        arrays["state_" + key] = getattr(episode.final, key).numpy()
    if gradient:
        g = episode.gradients()
        arrays["gradient_commands"] = g["commands"]
        for group in ("initial", "model"):
            arrays.update({f"gradient_{group}_{key}": value for key, value in g[group].items()})
    return arrays


def native(args: argparse.Namespace, phase: Callable[..., Any]) -> dict[str, Report]:
    """Run the native Warp benchmark path and record its individual phase measurements."""
    import numpy as np
    import warp as wp

    from envs.soft_walker.diagnostics.warp.checks import _validate_execution, compare_array
    from envs.soft_walker.engine.warp_native.episode import Episode
    from envs.soft_walker.engine.warp_native.model import DeviceModel

    bundle, commands = fixture(args)
    bundle.save(args.out / "model.npz")
    np.save(args.out / "commands.npy", commands)
    write(
        args.out / "protocol.json",
        dict(
            substeps=args.substeps,
            settling_steps=args.settling,
            checkpoint_stride=args.checkpoint_stride,
            length_ref=0.2,
            velocity_weight=0.1,
            work_weight=0.05,
            objective="massmean(-accumulated_dx/.2 + .1*|v_final|^2) + .05*positive_work",
            settling_included=True,
            qualification="resolved-physics performance fixture, not a gait score",
        ),
    )
    options = dict(
        legacy=("legacy", False, False),
        adaptive=("adaptive", False, False),
        cached=("adaptive", True, False),
        optimized=("adaptive", True, True),
        dynamic=("adaptive_dynamic", False, False),
        dynamic_cached=("adaptive_dynamic", True, False),
        dynamic_optimized=("adaptive_dynamic", True, True),
        linear_reuse=("adaptive", True, True),
        compact=("adaptive", True, True),
        compact_reuse=("adaptive", True, True),
        block=("adaptive", True, True),
        all=("adaptive", True, True),
    )
    reference = None
    results = {}
    for variant in args.variants:
        method, cache, roots = options[variant]
        destination = args.out / variant
        destination.mkdir()
        cfg = replace(
            bundle.config,
            krylov_method=method,
            cache_tangent=cache,
            compact_nodes=args.compact_nodes
            if variant in ("compact", "compact_reuse", "all")
            else 0,
            linear_work_reuse=variant in ("linear_reuse", "compact_reuse", "block", "all"),
            block_preconditioner=variant in ("block", "all"),
        )
        candidate = replace(bundle, config=cfg)
        phase(variant + ": allocate")
        start = time.perf_counter()
        episode = Episode(
            DeviceModel(candidate, args.device),
            commands,
            substeps=args.substeps,
            settling_steps=args.settling,
            checkpoint_stride=args.checkpoint_stride,
            velocity_weight=0.1,
            work_weight=0.05,
            reuse_replay_roots=roots,
        )
        result: Report = dict(
            config=cfg.to_dict(),
            allocation_seconds=time.perf_counter() - start,
            particles=episode.owner.n,
            device=str(episode.device),
            cuda_validated=False,
            whole_program_timings={},
        )
        for gradient in [False, True] if args.gradient else [False]:
            name = "value_gradient" if gradient else "forward"
            phase(variant + ": compile/warm " + name)
            first = episode.run(gradient=gradient)
            _validate_execution(first, gradient=gradient, phase="First " + name)
            warm = episode.run(gradient=gradient)
            _validate_execution(warm, gradient=gradient, phase="Warm " + name)
            if episode.device.is_cuda:
                wp.capture_debug_dot_print(
                    episode.graphs["gradient" if gradient else "forward"],
                    str(destination / (name + ".dot")),
                )
            phase(variant + ": measure " + name)
            samples = []
            for repeat in range(args.repeats):
                sample = episode.run(gradient=gradient)
                _validate_execution(sample, gradient=gradient, phase=f"Timed {name} {repeat}")
                samples.append(sample)
            seconds = [sample["execution_seconds"] for sample in samples]
            row: Report = dict(
                first=first,
                samples=samples,
                median_seconds=float(np.median(seconds)),
                p10=float(np.percentile(seconds, 10)),
                p90=float(np.percentile(seconds, 90)),
                solver=episode.step.diagnostics(),
            )
            arrays = scalar_arrays(episode, gradient)
            np.savez_compressed(destination / (name + ".npz"), **arrays)
            if args.audit:
                phase(variant + ": audit " + name)
                from envs.soft_walker.engine.execution.cupti import DriverAudit

                observer = DriverAudit()
                stream = wp.get_stream(episode.device)
                wp.synchronize_stream(stream)
                with observer:
                    wp.capture_launch(
                        episode.graphs["gradient" if gradient else "forward"], stream=stream
                    )
                    wp.synchronize_stream(stream)
                audit = cast("NativeSubmissionReport", observer.report())
                after = scalar_arrays(episode, gradient)
                audit["numerical_comparisons"] = {
                    key: compare_array(after[key], value, name=key, precision=args.dtype)
                    for key, value in arrays.items()
                }
                write(destination / (name + "_audit.json"), audit)
                if not audit["passed"]:
                    raise RuntimeError(f"{variant} {name} residency audit failed")
                result["cuda_validated"] = True
            result["whole_program_timings"][name] = row
            write(destination / "result.json", result)
        if reference is None:
            reference = arrays
            result["reference_variant"] = variant
        else:
            result["comparisons"] = {
                key: compare_array(value, reference[key], name=key, precision=args.dtype)
                for key, value in arrays.items()
            }
        write(destination / "result.json", result)
        results[variant] = result
    return results


def jax_program(
    bundle: Bundle, substeps: int, settling: int, stride: int
) -> Callable[[ResolvedParameters], tuple[Array, dict[str, Array]]]:
    """Same resolved fields, initial state, command loss, and full derivative set.

    Checkpoint fixed-size time blocks; do not store an entire position history.
    Padding in the final block is an identity operation, not an extra timestep.
    """
    import jax
    import jax.numpy as j

    from envs.soft_walker.engine.physics import mpm
    from envs.soft_walker.engine.warp_native.mpm_conversion import to_jax

    m, initial, grid, config = to_jax(bundle)

    def objective(p: ResolvedParameters) -> tuple[Array, dict[str, Array]]:
        fields, seed = p["model"], p["initial"]
        cloud = m.cloud._replace(
            reference=fields["reference"],
            volume=fields["volume"],
            face_offset=fields["face_offset"],
        )
        model = m._replace(
            cloud=cloud,
            **{
                key: fields[key]
                for key in (
                    "alpha",
                    "young",
                    "poisson",
                    "density",
                    "strength",
                    "direction",
                    "contact_area",
                )
            },
            mass=fields["alpha"] * fields["density"] * fields["volume"],
        )
        state = initial._replace(
            **{key: seed[key] for key in ("x", "v", "F", "C", "actuator")},
            work_signed=seed["ledger"][0],
            work_positive=seed["ledger"][1],
            dissipated=seed["ledger"][2],
            time=seed["ledger"][3],
        )
        actions = j.concatenate(
            (
                j.zeros((settling, bundle.channels), model.alpha.dtype),
                j.repeat(p["commands"], substeps, axis=0),
            )
        )
        steps = actions.shape[0]
        blocks = (steps + stride - 1) // stride
        padded = j.pad(actions, ((0, blocks * stride - steps), (0, 0))).reshape(
            blocks, stride, bundle.channels
        )
        indices = j.arange(blocks * stride).reshape(blocks, stride)

        def one(
            carry: tuple[mpm.State, Array], data: tuple[Array, Array]
        ) -> tuple[tuple[mpm.State, Array], None]:
            s, travel = carry
            action, index = data

            def physical(_: None) -> tuple[mpm.State, Array]:
                result, diag = mpm.trial(
                    model, s, action, j.asarray(bundle.config.dt, model.alpha.dtype), grid, config
                )
                dx = j.where(model.active[:, None], bundle.config.dt * result.v, 0.0)
                return result, travel + dx

            return jax.lax.cond(index < steps, physical, lambda _: carry, None), None

        def block(
            carry: tuple[mpm.State, Array], data: tuple[Array, Array]
        ) -> tuple[tuple[mpm.State, Array], None]:
            return jax.lax.scan(jax.checkpoint(one), carry, data)[0], None

        (final, travel), _ = jax.lax.scan(
            jax.checkpoint(block), (state, j.zeros_like(state.x)), (padded, indices)
        )
        loss = (
            j.sum(model.mass * (-travel[:, 0] / 0.2 + 0.1 * j.sum(final.v**2, axis=-1)))
            / j.sum(model.mass)
            + 0.05 * final.work_positive
        )
        loss = j.where(final.status == 0, loss, j.nan)
        return loss, {key: getattr(final, key) for key in ("x", "v", "F", "C", "actuator")} | {
            "ledger": j.stack(
                (final.work_signed, final.work_positive, final.dissipated, final.time)
            )
        }

    return objective


def jax_reference(args: argparse.Namespace, phase: Callable[..., Any]) -> Report:
    """Run the JAX reference protocol with the same physical controls and measurement scope."""
    import jax
    import jax.numpy as j
    import numpy as np

    from envs.soft_walker.engine.execution.compiled import compile_program
    from envs.soft_walker.engine.execution.policy import ResidentPolicy

    # Match Warp's full float32 matrix products. JAX's default GPU precision
    # can otherwise reduce mantissa precision in 3D contractions.
    jax.config.update("jax_default_matmul_precision", "highest")
    write(
        args.out / "numerics.json",
        dict(
            matmul_precision=str(jax.config.jax_default_matmul_precision),
            enable_x64=bool(jax.config.jax_enable_x64),
            storage_dtype=args.dtype,
        ),
    )
    bundle, commands = fixture(args)
    p = dict(
        commands=commands,
        model={
            key: value for key, value in bundle.fields.items() if key not in ("kind", "channel")
        },
        initial={
            key: value
            for key, value in cast("dict[str, Array]", bundle.initial).items()
            if key != "status"
        },
    )
    p = jax.tree.map(j.asarray, p)
    objective = jax_program(bundle, args.substeps, args.settling, args.checkpoint_stride)
    results = {}
    for gradient in [False, True] if args.gradient else [False]:
        name = "value_gradient" if gradient else "forward"
        phase("JAX compile/certify " + name)
        f = jax.value_and_grad(objective, has_aux=True) if gradient else objective
        program = compile_program(
            f, (p,), policy=ResidentPolicy(args.device), out=args.out / name, name=name
        )
        arguments = program.stage(p)
        output = program.certify(*arguments)
        phase("JAX measure " + name)
        output = program.benchmark(*arguments, repeats=args.repeats)
        values = jax.device_get(output)
        (loss, state), gradient_values = values if gradient else (values, None)
        arrays = {
            "loss": np.asarray(loss).reshape(1),
            **{"state_" + key: np.asarray(v) for key, v in state.items()},
        }
        if gradient:
            arrays["gradient_commands"] = np.asarray(
                cast(dict[str, Any], gradient_values)["commands"]
            )
            for group in ("model", "initial"):
                arrays.update(
                    {
                        f"gradient_{group}_{key}": np.asarray(value)
                        for key, value in cast(dict[str, Any], gradient_values)[group].items()
                    }
                )
        np.savez_compressed(args.out / (name + ".npz"), **arrays)
        results[name] = json.loads((args.out / name / "timing.json").read_text())
    return results


def ladder(args: argparse.Namespace, phase: Callable[..., Any]) -> Report:
    """Compare native gradients against finite-difference scales and save the agreement evidence."""
    import numpy as np

    from envs.soft_walker.diagnostics.warp.checks import _validate_execution, contact_fixture
    from envs.soft_walker.engine.warp_native.episode import Episode
    from envs.soft_walker.engine.warp_native.model import DeviceModel

    b = contact_fixture(2, args.dtype)
    b.config = replace(b.config, krylov_method="adaptive", cache_tangent=True)
    c = np.array([[0.4], [-0.2], [0.15]], args.dtype)
    d = np.array([[0.1], [-0.07], [0.08]], args.dtype)
    e = Episode(
        DeviceModel(b, args.device),
        c,
        settling_steps=1,
        checkpoint_stride=2,
        velocity_weight=0.1,
        work_weight=0.05,
        reuse_replay_roots=True,
    )
    phase("Ladder center and whole adjoint")
    _validate_execution(e.run(True), gradient=True, phase="Ladder center")
    g = e.gradients()
    ad = float(
        np.sum(g["commands"] * d)
        + np.sum(g["model"]["young"] * b.fields["young"] * 0.8)
        + np.sum(g["model"]["strength"] * b.fields["strength"] * 0.1)
    )

    def value(theta: Array) -> float:
        e.set_commands(c + theta * d)
        e.owner.model.young.assign((b.fields["young"] * np.exp(0.8 * theta)).astype(args.dtype))
        e.owner.model.strength.assign((b.fields["strength"] * (1 + 0.1 * theta)).astype(args.dtype))
        result = e.run()
        _validate_execution(result, gradient=False, phase="Finite difference")
        return result["loss"]

    rows = []
    for h in 0.4 * 0.5 ** np.arange(9):
        phase(f"Ladder h={h}")
        plus, minus = value(float(h)), value(float(-h))
        fd = (plus - minus) / (2 * h)
        rows.append(
            dict(
                h=float(h),
                plus=plus,
                minus=minus,
                central_difference=float(fd),
                ad=ad,
                absolute_error=float(abs(fd - ad)),
                relative_error=float(abs(fd - ad) / max(abs(ad), 1e-30)),
            )
        )
        write(args.out / "ladder_progress.json", rows)
    orders = [
        float(np.log2(a["absolute_error"] / b["absolute_error"]))
        if min(a["absolute_error"], b["absolute_error"]) > 0
        else None
        for a, b in zip(rows, rows[1:])
    ]
    windows = [
        i
        for i in range(len(orders) - 1)
        if all(x is not None and 1.5 <= x <= 2.5 for x in orders[i : i + 2])
    ]
    passed = min(x["relative_error"] for x in rows) < (3e-5 if args.dtype == "float64" else 5e-3)
    passed = passed and (bool(windows) or args.dtype == "float32")
    result: Report = dict(
        passed=passed,
        rows=rows,
        orders=orders,
        second_order_windows=windows,
        qualification="float32 may reach a roundoff plateau; complete short contact/settling rollout",
    )
    write(args.out / "ladder.json", result)
    if not passed:
        raise AssertionError("Whole-episode central difference ladder failed")
    return result


def main() -> int:
    """Command entry point. One isolated precision/device performance experiment; no historical test sweep."""
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["JAX_ENABLE_X64"] = "true" if args.dtype == "float64" else "false"
    os.environ["JAX_PLATFORMS"] = "cuda" if args.mode == "jax" and args.device == "cuda" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    status: Report = dict(success=False, arguments=vars(args), started=time.time())

    def phase(name: str) -> None:
        write(args.out / "phase.json", dict(phase=name, time=time.time()))
        print(name, flush=True)

    try:
        action = dict(native=native, jax=jax_reference, ladder=ladder)[args.mode]
        status["result"] = action(args, phase)
        status["success"] = True
    except BaseException:
        status["exception"] = traceback.format_exc()
        print(status["exception"], file=sys.stderr, flush=True)
    finally:
        status["seconds"] = time.time() - status["started"]
        write(args.out / "status.json", status)
    return 0 if status["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
