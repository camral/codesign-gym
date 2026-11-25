#!/usr/bin/env python3
"""A/B the nested-loop and fused GMRES implementations on one physical interval.

Invoked by run_krylov_audit.py in an isolated, already configured process.
No XLA command-buffer flags are enabled by this script. Traces are separate
from timing, and their graph-capture behavior must be read from captured logs.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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
    source_group_files,
)

if TYPE_CHECKING:
    from pathlib import Path

    from envs.soft_walker.engine.drivers.task import BodyConfig, TaskState
    from envs.soft_walker.engine.physics import mpm
    from envs.soft_walker.engine.physics.common import StepInfo
    from envs.soft_walker.engine.types import (
        Array,
        BodyModel,
        JsonValue,
    )

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT / "src"))

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def write_json(path: Path, value: Any) -> None:
    """Write a JSON report, retaining the command's array/scalar encoding and error behavior."""

    def encode(item: Any) -> JsonValue:
        if hasattr(item, "tolist"):
            return item.tolist()
        if hasattr(item, "__dataclass_fields__"):
            return asdict(item)
        return str(item)

    path.write_text(json.dumps(value, indent=2, default=encode) + "\n")


def validate_gradient_result(
    method: str, value: float, status: int, gradient: Array, reference_value: float
) -> None:
    """Reject failed physical intervals even when rollback gives finite values."""
    import numpy as np

    if int(status) != 0:
        raise RuntimeError(f"{method} gradient interval failed: status={int(status)}")
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        raise RuntimeError(f"{method} nonfinite interval adjoint")
    np.testing.assert_allclose(
        value,
        reference_value,
        rtol=2e-8,
        atol=2e-10,
        err_msg=f"{method} gradient forward value differs from physical reference",
    )


def environment(out: Path, device: str) -> Report:
    """Capture hardware and dependency provenance in the evidence directory."""
    import importlib.metadata

    import jax

    from envs.soft_walker.engine.spaces import USING_GYMNASIUM

    info = {
        "device_requested": device,
        "devices": [str(d) for d in jax.devices()],
        "jax_backend": jax.default_backend(),
        "x64": jax.config.x64_enabled,
        "real_gymnasium": USING_GYMNASIUM,
        "python": sys.version,
        "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
        "sources": {
            p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (
                "src/envs/soft_walker/engine",
                "src/envs/soft_walker/scripts",
                "src/envs/soft_walker/diagnostics",
                "tests/soft_walker",
            )
            for p in source_group_files(ROOT, folder)
        },
        "packages": {},
    }
    for name in (
        "jax",
        "jaxlib",
        "jax-cuda12-plugin",
        "jax-cuda13-plugin",
        "gymnasium",
        "numpy",
        "scipy",
        "diffrax",
        "pytest",
    ):
        try:
            info["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info["packages"][name] = None
    write_json(out / "environment.json", info)
    expected = "gpu" if device == "cuda" else "cpu"
    if jax.default_backend() != expected:
        raise RuntimeError(f"Requested {device}; refusing backend fallback")
    return info


def summarize_trace(path: Path) -> Report:
    """Count full-capture events, avoiding duplicate Perfetto export accounting."""
    from collections import Counter

    events = json.load(gzip.open(path, "rt"))["traceEvents"]
    gpu_pids = {
        e["pid"]
        for e in events
        if e.get("name") == "process_name" and "GPU" in e.get("args", {}).get("name", "")
    }
    host_pids = {
        e["pid"]
        for e in events
        if e.get("name") == "process_name"
        and any(tag in e.get("args", {}).get("name", "").upper() for tag in ("HOST", "CPU"))
    }
    device_events = [e for e in events if e.get("pid") in gpu_pids and e.get("ph") == "X"]
    kernels = [e for e in device_events if "kernel_details" in e.get("args", {})]
    copies = [e for e in device_events if "memcpy_details" in e.get("args", {})]
    return {
        "trace": str(path),
        "scope": "whole capture; timings exclude profiling",
        "kernel_count": len(kernels),
        "copies": dict(Counter(e["name"] for e in copies)),
        "host_while_checks": sum(e.get("name") == "While" for e in events),
        "host_conditional_checks": sum(e.get("name") == "Conditional" for e in events),
        "graph_launches": sum(
            e.get("ph") == "X"
            and e.get("pid") in host_pids
            and bool(re.match(r"^(?:cu|cuda)GraphLaunch(?:_\w+)?(?:\W|$)", e.get("name", "")))
            for e in events
        ),
        "kernel_median_us": (
            float(__import__("numpy").median([e["dur"] for e in kernels])) if kernels else None
        ),
        "profiling_warning": "No command-buffer profiling override is set. Profile counts are not untraced timing or SM utilization.",
    }


def benchmark(args: argparse.Namespace) -> None:
    """Execute the selected physics performance comparisons and save their numerical validity evidence."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from envs.soft_walker.engine.drivers import environment as drivers
    from envs.soft_walker.presets.gaits import seed_controller
    from envs.soft_walker.presets.locomotion import DemoConfig, build_demo
    from envs.soft_walker.shared import periodic_action

    if args.dtype != "float64":
        raise ValueError(
            "The returned full biped fixture is float64; float32 is covered by separate small physical tests"
        )
    demo = DemoConfig(dtype="float64", cycles=6)
    write_json(args.out / "protocol.json", demo)
    timings, results, compiled, configs = {}, {}, {}, {}
    env = None
    try:
        started = time.perf_counter()
        env, emb = build_demo(demo, device=args.device)
        _, rejected, info = env.reset(seed=demo.seed, embodiment=emb)
        write_json(args.out / "reset_incremental.json", info)
        if rejected.any():
            raise RuntimeError("Reference biped settling failed")
        write_json(args.out / "embodiment.json", emb)
        m, initial = cast("BodyModel", env._model), cast("TaskState", env._state)
        body_config = cast("BodyConfig", env.config)
        reference_reset_s = time.perf_counter() - started
        params = jnp.asarray(seed_controller(2, amplitude=-0.8, lift=1.2, dtype="float64"))
        action = periodic_action(params, jnp.asarray(0.0, jnp.float64), demo.period)
        np.savez(
            args.out / "inputs.npz",
            action=np.asarray(action),
            parameters=np.asarray(params),
            # Keys are prefixed: none collide with NumPy's reserved allow_pickle keyword.
            **{f"initial_{k}": np.asarray(v) for k, v in initial.body._asdict().items()},  # type: ignore[arg-type]
        )
        for method in ("incremental", "fused"):
            config = replace(
                body_config,
                dynamics=replace(
                    body_config.dynamics,
                    newton=replace(body_config.dynamics.newton, gmres_method=method),
                ),
            )
            configs[method] = config

            def step(
                model: BodyModel,
                state: TaskState,
                control: Array,
                *,
                _config: BodyConfig = config,
            ) -> tuple[TaskState, tuple[StepInfo, Array]]:
                return drivers.advance(env.kind, model, state, control, _config, env._grid)

            t = time.perf_counter()
            lowered = jax.jit(step).lower(m, initial, action)
            lower_s = time.perf_counter() - t
            t = time.perf_counter()
            executable = lowered.compile()
            compile_s = time.perf_counter() - t
            compiled[method] = executable
            ir = str(lowered.compiler_ir(dialect="stablehlo"))
            with gzip.open(args.out / f"{method}.stablehlo.mlir.gz", "wt") as f:
                f.write(ir)
            with gzip.open(args.out / f"{method}.hlo.txt.gz", "wt") as f:
                f.write(executable.as_text())
            t = time.perf_counter()
            result = jax.block_until_ready(executable(m, initial, action))
            first_s = time.perf_counter() - t
            if int(result[0].status) != 0:
                raise RuntimeError(f"{method} interval failed")
            results[method] = result
            timings[method] = {
                "lower_s": lower_s,
                "compile_or_cache_load_s": compile_s,
                "first_s": first_s,
                "warm_samples_s": [],
                "stablehlo_while_operations": ir.count("stablehlo.while"),
                "solver": result[1][0]._asdict(),
            }
            for _ in range(2):
                jax.block_until_ready(executable(m, initial, action))
            mem = executable.memory_analysis()
            timings[method]["compiler_memory_estimate"] = {
                key: getattr(mem, key, None)
                for key in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes")
            }
        # Alternating/reversed order reduces (does not eliminate) thermal bias.
        for repetition in range(args.repeats):
            order = ("incremental", "fused") if repetition % 2 == 0 else ("fused", "incremental")
            for method in order:
                t = time.perf_counter()
                results[method] = jax.block_until_ready(compiled[method](m, initial, action))
                timings[method]["warm_samples_s"].append(time.perf_counter() - t)
        arrays = {}
        for method, (state, diagnostics) in results.items():
            for key, value in state.body._asdict().items():
                arrays[f"{method}_{key}"] = np.asarray(value)
            arrays[f"{method}_displacement"] = np.asarray(state.displacement)
            arrays[f"{method}_diagnostics"] = np.asarray(diagnostics[0])
            arrays[f"{method}_dt_trace"] = np.asarray(diagnostics[1])
            samples = timings[method]["warm_samples_s"]
            timings[method].update(
                median_s=float(np.median(samples)),
                p10_s=float(np.quantile(samples, 0.1)),
                p90_s=float(np.quantile(samples, 0.9)),
            )
        # Physical state parity excludes iteration counters, which may differ
        # after two-pass orthogonalization/rounding, but never ignores failure.
        for name in (*initial.body._fields, "displacement"):
            np.testing.assert_allclose(
                arrays[f"fused_{name}"], arrays[f"incremental_{name}"], rtol=2e-8, atol=2e-10
            )
        if args.gradient:
            for method in ("incremental", "fused"):
                config = configs[method]

                def loss(
                    model: BodyModel,
                    state: TaskState,
                    control: Array,
                    *,
                    _config: BodyConfig = config,
                ) -> tuple[Array, Array]:
                    new, _ = drivers.advance(env.kind, model, state, control, _config, env._grid)
                    value = (
                        -new.displacement[0] / _config.length_ref
                        + 0.001 * new.body.work_positive / _config.work_ref
                    )
                    return value, new.status

                t = time.perf_counter()
                grad_lowered = jax.jit(jax.value_and_grad(loss, argnums=2, has_aux=True)).lower(
                    m, initial, action
                )
                grad_ir = str(grad_lowered.compiler_ir(dialect="stablehlo"))
                with gzip.open(args.out / f"{method}.gradient.stablehlo.mlir.gz", "wt") as f:
                    f.write(grad_ir)
                if f"{env._grid.count * 2}x{env._grid.count * 2}x" in grad_ir:
                    raise AssertionError("Unexpected global dense tangent in reverse program")
                executable = grad_lowered.compile()
                timings[method]["gradient_compile_s"] = time.perf_counter() - t
                jax.block_until_ready(executable(m, initial, action))
                t = time.perf_counter()
                (value, gradient_status), gradient = jax.block_until_ready(
                    executable(m, initial, action)
                )
                timings[method]["value_and_gradient_s"] = time.perf_counter() - t
                reference_value = (
                    -arrays[f"{method}_displacement"][0] / config.length_ref
                    + 0.001 * arrays[f"{method}_work_positive"] / config.work_ref
                )
                validate_gradient_result(method, value, gradient_status, gradient, reference_value)
                arrays[f"{method}_loss"] = np.asarray(value)
                arrays[f"{method}_gradient"] = np.asarray(gradient)
                arrays[f"{method}_gradient_status"] = np.asarray(gradient_status)
                arrays[f"{method}_forward_reference_loss"] = np.asarray(reference_value)
            np.testing.assert_allclose(
                arrays["fused_gradient"], arrays["incremental_gradient"], rtol=2e-6, atol=2e-9
            )
        np.savez(args.out / "references.npz", **arrays)
        write_json(
            args.out / "benchmark.json",
            {
                "timing_scope": "same frozen settled model/state/action; entire 20ms interval (40 substeps); no API or rendering",
                "timings": timings,
                "reference_reset_s": reference_reset_s,
                "particles": len(m.mass),
                "active_particles": int(np.asarray(m.active).sum()),
                "grid_shape": env._grid.shape,
                "allocated_unknowns": env._grid.count * 2,
                "fused_speedup": timings["incremental"]["median_s"] / timings["fused"]["median_s"],
                "position_max_abs_difference": float(
                    np.max(np.abs(arrays["fused_x"] - arrays["incremental_x"]))
                ),
                "tolerances_unchanged": True,
                "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
            },
        )
        print(json.dumps({method: timings[method]["median_s"] for method in timings}), flush=True)
        # Validate fused settling independently, not just one step from an old root.
        t = time.perf_counter()
        other, other_emb = build_demo(replace(demo, gmres_method="fused"), device=args.device)
        try:
            _, rejected, info = other.reset(seed=demo.seed, embodiment=other_emb)
            write_json(args.out / "reset_fused.json", info)
            if rejected.any():
                raise RuntimeError("Fused biped settling failed")
            other_body = cast("mpm.State", cast("TaskState", other._state).body)
            initial_body = cast("mpm.State", initial.body)
            delta = float(np.max(np.abs(np.asarray(other_body.x) - np.asarray(initial_body.x))))
            np.testing.assert_allclose(other_body.x, initial_body.x, rtol=2e-8, atol=2e-10)
            write_json(
                args.out / "settling_parity.json",
                {
                    "valid": True,
                    "max_position_difference": delta,
                    "fused_reset_s": time.perf_counter() - t,
                },
            )
        finally:
            other.close()
        if args.trace:
            summaries = {}
            for method in ("incremental", "fused"):
                path = args.out / "traces" / method
                with jax.profiler.trace(str(path), create_perfetto_trace=True):
                    with jax.profiler.TraceAnnotation(f"codesign/krylov/{method}"):
                        traced = jax.block_until_ready(compiled[method](m, initial, action))
                if int(traced[0].status) != 0:
                    raise RuntimeError(f"{method} trace call invalid")
                for p in path.rglob("*.trace.json.gz"):
                    summaries[method] = summarize_trace(p)
            write_json(args.out / "trace_summary.json", summaries)
    finally:
        if env is not None:
            env.close()


def numerics(args: argparse.Namespace) -> None:
    """Run precision-specific primal/gradient checks before interpreting solver speed measurements."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    from envs.soft_walker.engine.numerics.solvers import NewtonConfig, linear_solve
    from envs.soft_walker.shared import difference_ladder

    n = 17
    diagonal = jnp.linspace(1, 2, n, dtype=args.dtype)
    theta = jnp.asarray([0.15, 0.3, -0.1], args.dtype)
    ladders = {}
    for method in ("incremental", "fused"):
        config = NewtonConfig(gmres_method=method, preconditioner_side="right")

        def loss(p: Array) -> Array:
            d = jnp.exp(p[0]) * diagonal
            x, _ = linear_solve(
                lambda v: d * v + p[1] * 0.2 * jnp.roll(v, 1),
                (1 + p[2]) * jnp.linspace(0.3, 1, n, dtype=args.dtype),
                d,
                config,
            )
            return 0.5 * jnp.vdot(x, x)

        value = jax.jit(loss)
        vg = jax.jit(jax.value_and_grad(loss))
        ladder = difference_ladder(value, vg, np.asarray(theta))
        ladders[method] = ladder
        write_json(args.out / "ladders.json", ladders)
        if not ladder["agreement_passed"]:
            raise AssertionError(f"{method}: finite-difference disagreement")
        if args.dtype == "float64" and not ladder["second_order_observed"]:
            raise AssertionError(f"{method}: missing second-order window")
    np.testing.assert_allclose(
        ladders["fused"]["gradient"],
        ladders["incremental"]["gradient"],
        rtol=2e-4 if args.dtype == "float32" else 2e-8,
        atol=1e-8,
    )


def main() -> int:
    """Command entry point. A/B the nested-loop and fused GMRES implementations on one physical interval."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("benchmark", "numerics"))
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--dtype", choices=("float32", "float64"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--gradient", action="store_true")
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()
    if args.repeats < 1:
        p.error("repeats must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    status: Report = {"completed": False, "exitcode": 1}
    try:
        import jax

        jax.config.update("jax_default_matmul_precision", "highest")
        environment(args.out, args.device)
        if args.dtype == "float64" and not jax.config.x64_enabled:
            raise RuntimeError("float64 requested but JAX_ENABLE_X64 is false")
        if args.mode == "benchmark":
            benchmark(args)
        else:
            numerics(args)
        status.update(completed=True, exitcode=0)
    except BaseException as exc:
        status.update(error=repr(exc), traceback=traceback.format_exc())
        print(status["traceback"], file=sys.stderr, flush=True)
    finally:
        write_json(args.out / "status.json", status)
    return status["exitcode"]


if __name__ == "__main__":
    raise SystemExit(main())
