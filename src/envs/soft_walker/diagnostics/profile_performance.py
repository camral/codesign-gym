"""Focused runtime experiments; called by run_performance_audit.py.

This is diagnostic infrastructure, not a gait benchmark. Legacy orchestration
is replayed only here to isolate its cost while keeping physics identical.
"""

from __future__ import annotations

import argparse
import cProfile
import gzip
import io
import json
import pstats
import sys
import time
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np

if __package__ in (None, ""):
    # Direct-file execution must select this checkout before package imports.
    _source_root = str(Path(__file__).resolve().parents[3])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)

from envs.soft_walker.engine import make_vec
from envs.soft_walker.engine.drivers.task import BodyConfig
from envs.soft_walker.engine.numerics.solvers import NewtonConfig, linear_solve
from envs.soft_walker.engine.physics.spring_config import SpringConfig
from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
    evidence_write_json as write_json,
    finite_difference_ladder,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import DTypeLike

    from envs.soft_walker.engine.drivers.spring_environment import Spring2DVectorEnv
    from envs.soft_walker.engine.drivers.task import TaskState
    from envs.soft_walker.engine.physics import spring
    from envs.soft_walker.engine.types import (
        Array,
        DriverState,
        HostArray,
        PhysicsModel,
    )

IDS = {
    "spring": "SoftSpring2D-v0",
    "fem": "SoftHexFEM3D-v0",
    "mpm2": "SoftMPM2D-v0",
    "mpm3": "SoftMPM3D-v0",
    "beam": "SoftBeam3D-v0",
}

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]
T = TypeVar("T")


def synchronize(value: T) -> T:
    """Wait for device work in a result pytree and return the same result structure."""
    return jax.block_until_ready(value)


def timed(
    label: str,
    function: Callable[..., Any],
    args: tuple[Any, ...],
    out: Path,
    repeats: int,
    *,
    compiled: bool = False,
) -> tuple[Any, Report, Callable[..., Any]]:
    """Compilation, first execution and synchronized samples are distinct."""
    metadata: Report = {"name": label, "compiled_explicitly": compiled}
    if compiled:
        start = time.perf_counter()
        lowered = jax.jit(function).lower(*args)
        metadata["lowering_s"] = time.perf_counter() - start
        start = time.perf_counter()
        executable = lowered.compile()
        metadata["compile_or_cache_load_s"] = time.perf_counter() - start
        (out / (label + ".hlo.txt")).write_text(executable.as_text())
        try:
            analysis = executable.memory_analysis()
            metadata["compiler_memory_bytes"] = {
                key: getattr(analysis, key)
                for key in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "temp_size_in_bytes",
                    "alias_size_in_bytes",
                )
            }
        except Exception as error:
            metadata["memory_analysis_error"] = repr(error)
        call = lambda: executable(*args)
    else:
        call = lambda: function(*args)

    start = time.perf_counter()
    result = synchronize(call())
    metadata["first_execution_s"] = time.perf_counter() - start
    for _ in range(2):
        synchronize(call())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = synchronize(call())
        samples.append(time.perf_counter() - start)
    metadata.update(
        samples_s=samples,
        median_s=float(np.median(samples)),
        p10_s=float(np.quantile(samples, 0.1)),
        p90_s=float(np.quantile(samples, 0.9)),
    )
    write_json(out / (label + ".json"), metadata)
    return result, metadata, call


def legacy_vector_step(
    vec: Spring2DVectorEnv, actions: Array
) -> tuple[HostArray, HostArray, HostArray, HostArray, Report]:
    """Pre-patch steady-state orchestration, for a frozen nonterminal fixture.

    This copies the old numeric stack/unstack/report path, not its whole reset
    implementation. No production caller uses this reference implementation.
    """
    from envs.soft_walker.engine.drivers.vector_lifecycle import _stack_info

    controls = [e._check_action(a) for e, a in zip(vec.envs, actions)]
    models = jax.tree.map(lambda *x: jnp.stack(x), *[e._model for e in vec.envs])
    states = jax.tree.map(lambda *x: jnp.stack(x), *[e._state for e in vec.envs])
    new, diagnostics = synchronize(vec.envs[0]._advance_batch(models, states, jnp.stack(controls)))
    outputs = []
    for i, env in enumerate(vec.envs):
        before = env._state
        env._state = jax.tree.map(lambda x: x[i], new)
        diag = jax.tree.map(lambda x: x[i], diagnostics)
        env._steps += 1
        terminated = int(env._state.status) != 0
        truncated = env._steps >= env.config.horizon and not terminated
        env._done = terminated or truncated
        obs, mask = env._observe(cast("PhysicsModel", env._model), cast("DriverState", env._state))
        info = env._info(diag, mask)
        info["autoreset"] = False
        reward = float(
            env._reward(
                cast("PhysicsModel", env._model),
                cast("DriverState", before),
                cast("DriverState", env._state),
            )
        )
        outputs.append((np.asarray(obs), reward, terminated, truncated, info))
    obs, rewards, terminations, truncations, infos = zip(*outputs)
    return (
        np.stack(obs),
        np.asarray(rewards),
        np.asarray(terminations),
        np.asarray(truncations),
        _stack_info(infos),
    )


def cprofile(label: str, call: Callable[..., Any], out: Path) -> None:
    """Profile host-side call overhead into a named artifact, keeping it separate from synchronized timing."""
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(3):
        synchronize(call())
    profiler.disable()
    profiler.dump_stats(str(out / (label + ".prof")))
    text = io.StringIO()
    stats = pstats.Stats(profiler, stream=text).sort_stats("cumulative")
    stats.print_stats(60)
    (out / (label + "_host_profile.txt")).write_text(text.getvalue())
    selected = []
    for (filename, line, name), (primitive, total, own, cumulative, _) in stats.stats.items():  # type: ignore[attr-defined]  # CPython pstats exposes this documented data table.
        if name in ("__getitem__", "apply_primitive", "device_put", "device_get") or (
            name == "stack" and "jax" in filename
        ):
            selected.append(
                dict(
                    file=filename,
                    line=line,
                    name=name,
                    calls=total,
                    own_s=own,
                    cumulative_s=cumulative,
                )
            )
    write_json(out / (label + "_dispatch_counts.json"), selected)


def trace_calls(calls: dict[str, Callable[[], Any]], out: Path) -> Report:
    """A bounded warm trace; never include tracing in baseline timing samples."""
    status: Report = {"requested": True, "completed": False}
    try:
        with jax.profiler.trace(str(out / "trace"), create_perfetto_trace=True):
            for name, call in calls.items():
                with jax.profiler.TraceAnnotation("codesign/" + name):
                    synchronize(call())
        status["completed"] = True
        for path in sorted((out / "trace").rglob("*.trace.json.gz")):
            with gzip.open(path, "rt") as stream:
                trace = json.load(stream)
            counts = defaultdict(list)
            for event in trace.get("traceEvents", []):
                if event.get("ph") == "X" and "dur" in event:
                    key = (str(event.get("pid")), event.get("cat", ""), event.get("name", ""))
                    counts[key].append(float(event["dur"]))
            groups = [
                dict(
                    process=key[0],
                    category=key[1],
                    name=key[2],
                    count=len(values),
                    summed_duration_us=sum(values),
                    median_us=float(np.median(values)),
                )
                for key, values in counts.items()
            ]
            groups.sort(key=lambda row: row["summed_duration_us"], reverse=True)
            write_json(
                out / "trace_summary.json",
                {
                    "source": str(path.relative_to(out)),
                    "top_events": groups[:150],
                    "warning": "Nested and overlapping event sums are NOT wall time; inspect stream timelines for kernel gaps and transfers.",
                },
            )
    except Exception as error:
        status["error"] = repr(error)
    write_json(out / "trace_status.json", status)
    return status


def driver_probe(
    kind: str,
    dtype: DTypeLike,
    out: Path,
    repeats: int,
    batch_size: int,
    method: str,
    profile: bool,
    size: tuple[float, ...],
) -> None:
    """Measure a driver's API, compiled transition, and gradient paths with separate timing scopes."""
    dim = 2 if kind in ("spring", "mpm2") else 3
    shape = (
        ((2, 1) if dim == 2 else (2, 1, 1))
        if size == "small"
        else ((4, 2) if dim == 2 else (4, 2, 1))
    )
    args: dict[str, Any] = dict(
        grid_shape=shape,
        size=tuple(0.05 * n for n in shape),
        control_dt=0.001,
        substeps=1,
        settle_steps=0,
        horizon=8,
    )
    cfg: SpringConfig | BodyConfig
    if kind == "spring":
        if method != "incremental":
            raise ValueError("This probe's spring solver is unchanged; use incremental")
        cfg = SpringConfig(**args)
    else:
        cfg = BodyConfig(**args, grid_margin=1)
        cfg = replace(
            cfg,
            dynamics=replace(
                cfg.dynamics, newton=replace(cfg.dynamics.newton, gmres_method=method)
            ),
        )
    vec, base = make_vec(
        IDS[kind],
        num_envs=batch_size,
        dtype=dtype,
        config=cfg,
        max_observation_space_size=12000,
    )
    start = time.perf_counter()
    _, rejected, _ = vec.reset(seed=17, embodiment=base)
    if rejected.any():
        raise AssertionError("Diagnostic embodiment reset rejected")
    reset_s = time.perf_counter() - start
    first = vec.envs[0]
    models = tuple(cast("PhysicsModel", e._model) for e in vec.envs)
    states = tuple(cast("DriverState", e._state) for e in vec.envs)
    actions = np.asarray([[0.24 * (1 - 0.05 * i), -0.17, 0, 0] for i in range(batch_size)], dtype)
    controls = jax.device_put(actions[:, : first.channels], first.device)
    bm = jax.tree.map(lambda *x: jnp.stack(x), *models)
    bs = jax.tree.map(lambda *x: jnp.stack(x), *states)

    def rewind() -> None:
        for env, state in zip(vec.envs, states):
            env._state, env._steps, env._done = state, 0, False

    def old_api() -> tuple[HostArray, HostArray, HostArray, HostArray, Report]:
        rewind()
        return legacy_vector_step(vec, actions)

    def new_api() -> tuple[HostArray, HostArray, HostArray, HostArray, Report]:
        rewind()
        return vec.step(actions)

    results, calls, references = [], {}, {}
    for name, fn, inputs in (
        ("scalar_physics", first._advance, (models[0], states[0], controls[0])),
        ("batch_physics", first._advance_batch, (bm, bs, controls)),
    ):
        result, metadata, call = timed(name, fn, inputs, out, repeats, compiled=True)
        if np.any(np.asarray(result[0].status) != 0):
            raise AssertionError(name + " has a failed physical state")
        metadata["solver"] = result[1]._asdict() if kind == "spring" else result[1][0]._asdict()
        write_json(out / (name + ".json"), metadata)
        results.append(metadata)
        calls[name] = call
        references[name + "_velocity"] = np.asarray(result[0].v)

    for name, api_call in (("legacy_vector_api", old_api), ("batched_vector_api", new_api)):
        result, metadata, call = timed(name, api_call, (), out, repeats)
        if np.any(result[2]):
            raise AssertionError(name + " terminated numerically")
        references[name + "_observation"] = np.asarray(result[0])
        references[name + "_reward"] = np.asarray(result[1])
        results.append(metadata)
        calls[name] = call
        cprofile(name, call, out)
    rtol, atol = (4e-4, 3e-6) if dtype == np.dtype("float32") else (2e-9, 2e-11)
    for suffix in ("observation", "reward"):
        np.testing.assert_allclose(
            references["legacy_vector_api_" + suffix],
            references["batched_vector_api_" + suffix],
            rtol=rtol,
            atol=atol,
        )

    def one_loss(action: Array) -> Array:
        new, _ = first._advance(models[0], states[0], action)
        physical = cast("spring.State", new) if kind == "spring" else cast("TaskState", new).body
        value = 20 * jnp.sum(new.v**2) + 1e5 * physical.work_signed
        return jnp.where(new.status == 0, value, jnp.nan)

    result, metadata, call = timed(
        "scalar_value_and_grad",
        jax.value_and_grad(one_loss),
        (controls[0],),
        out,
        repeats,
        compiled=True,
    )
    if not all(np.all(np.isfinite(x)) for x in jax.tree.leaves(result)):
        raise AssertionError("Nonfinite action gradient")
    references["action_gradient"] = np.asarray(result[1])
    references["diagnostic_value"] = np.asarray(result[0])
    calls["scalar_value_and_grad"] = call
    results.append(metadata)

    # A population of controls shares this frozen design/initial state. Each
    # member has an independent loss and gradient; no batch reduction couples
    # candidates. This tests the outer vmap relevant to gait optimization.
    batch_result, batch_metadata, batch_call = timed(
        "batch_value_and_grad",
        jax.vmap(jax.value_and_grad(one_loss)),
        (controls,),
        out,
        repeats,
        compiled=True,
    )
    if not all(np.all(np.isfinite(x)) for x in jax.tree.leaves(batch_result)):
        raise AssertionError("Nonfinite population action gradient")
    np.testing.assert_allclose(
        np.asarray(batch_result[1][0]),
        references["action_gradient"],
        rtol=rtol,
        atol=atol,
    )
    references["batch_action_gradient"] = np.asarray(batch_result[1])
    references["batch_diagnostic_value"] = np.asarray(batch_result[0])
    results.append(batch_metadata)
    calls["batch_value_and_grad"] = batch_call

    # Check the real driver's selected action direction as well as the
    # independent analytic linear-system ladders in the separate solver stage.
    direction = jnp.asarray([0.7, -0.3], dtype)
    h: HostArray = np.asarray([0.16, 0.08, 0.04, 0.02, 0.01, 0.005, 0.0025], np.float64)
    if dtype == np.dtype("float64"):
        h = np.r_[h, [0.00125, 0.000625, 0.0003125]]
    ladder = finite_difference_ladder(
        jax.jit(one_loss),
        controls[0],
        direction,
        steps=h,
        autodiff=jnp.dot(result[1], direction),
        name=kind + "_action",
    )
    # Convergence order is recorded rather than fabricated on nearly quadratic
    # one-step fixtures. Analytic solver probes require second-order windows.
    allowance = 2e-3 if dtype == np.dtype("float32") else 2e-6
    ladder["passed"] = ladder["best_relative_error"] < allowance
    write_json(out / "action_ladder.json", ladder)
    if not ladder["passed"]:
        raise AssertionError("Action derivative failed central-difference audit")

    trace_status = trace_calls(calls, out) if profile else {"requested": False}
    np.savez_compressed(out / "references.npz", **references)
    active_unknowns = int(np.sum(np.asarray(models[0].active))) * dim
    if kind.startswith("mpm"):
        from envs.soft_walker.engine.physics import mpm

        mpm_state = cast("mpm.State", cast("TaskState", states[0]).body)
        stencil = mpm.stencil(mpm_state.x, first._grid)
        grid_mass, _ = mpm.p2g(models[0].mass, mpm_state.v, mpm_state.C, stencil, first._grid)
        active_unknowns = int(np.count_nonzero(np.asarray(grid_mass))) * dim
    elif kind == "beam":
        active_unknowns *= 2
    summary: Report = {
        "active_unknowns_at_start": active_unknowns,
        "passed": True,
        "driver": kind,
        "method": method,
        "dtype": str(dtype),
        "batch_size": batch_size,
        "config": asdict(cfg),
        "first_reset_s": reset_s,
        "model_sites": len(models[0].mass),
        "allocated_unknowns": first._grid.count * dim
        if getattr(first, "_grid", None) is not None
        else states[0].v.size * (2 if kind == "beam" else 1),
        "benchmarks": results,
        "trace": trace_status,
        "scope": "One frozen, valid, 1 ms diagnostic interval; not a gait or an accuracy-certified timestep.",
        "api_reference": "Old warm orchestration on the same patched numerical kernels; independent uploaded-versus-patched probe also provided in delivery evidence.",
    }
    write_json(out / "summary.json", summary)
    vec.close()


def solver_probe(dtype: DTypeLike, out: Path, repeats: int, profile: bool) -> None:
    """Measure representative implicit-solver primal/adjoint work for the requested precision."""
    n = 128
    index = jnp.arange(n, dtype=dtype)
    rhs = jnp.sin(index * 0.1 + 0.3)
    weights = jnp.cos(index * 0.03)
    theta = jnp.asarray(0.2, dtype)
    references, rows, calls = {}, [], {}
    for method in ("incremental", "batched"):
        cfg = NewtonConfig(gmres_method=method)

        def loss(t: Array) -> Array:
            diagonal = 2 + jnp.exp(t) + 0.002 * index
            op = lambda x: diagonal * x + 0.17 * jnp.roll(x, 1) - 0.08 * jnp.roll(x, -1)
            x, _ = linear_solve(op, rhs, diagonal, cfg)
            return jnp.sum(weights * x**2)

        result, data, call = timed(
            method, jax.value_and_grad(loss), (theta,), out, repeats, compiled=True
        )
        references[method + "_value"] = np.asarray(result[0])
        references[method + "_gradient"] = np.asarray(result[1])
        calls[method] = call
        steps = [0.2 / 2**i for i in range(12)]
        ladder = finite_difference_ladder(
            jax.jit(loss),
            theta,
            jnp.asarray(1, dtype),
            steps=steps,
            autodiff=result[1],
            name=method,
        )
        orders = [r["observed_order"] for r in ladder["rows"]]
        hits = [p is not None and 1.5 < p < 2.5 for p in orders]
        convergence = (
            any(hits)
            if dtype == np.dtype("float32")
            else any(a and b for a, b in zip(hits, hits[1:]))
        )
        accuracy = ladder["best_relative_error"] < (1e-3 if dtype == np.dtype("float32") else 1e-6)
        ladder.update(
            {"passed": bool(convergence and accuracy), "observed_second_order_window": convergence}
        )
        write_json(out / (method + "_ladder.json"), ladder)
        if not ladder["passed"]:
            raise AssertionError(method + " ladder did not converge")
        rows.append(data)
    np.testing.assert_allclose(
        references["incremental_gradient"], references["batched_gradient"], rtol=8e-5, atol=1e-9
    )
    np.savez_compressed(out / "references.npz", **references)
    trace_status = trace_calls(calls, out) if profile else {"requested": False}
    write_json(
        out / "summary.json",
        {
            "passed": True,
            "benchmarks": rows,
            "trace": trace_status,
            "scope": "Independent smooth nonsymmetric linear-system probe, not proof of all-model suitability for batched GMRES.",
        },
    )


def main() -> None:
    """Command entry point. Focused runtime experiments; called by run_performance_audit.py."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--driver", choices=list(IDS) + ["linear"], required=True)
    parser.add_argument("--dtype", choices=["float32", "float64"], required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--method", choices=["incremental", "batched"], default="incremental")
    parser.add_argument("--size", choices=["small", "medium"], default="small")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_default_matmul_precision", "highest")
    dtype = np.dtype(args.dtype)
    if bool(jax.config.jax_enable_x64) != (dtype == np.dtype("float64")):
        raise ValueError("Precision must be configured before importing JAX")
    if args.driver == "linear":
        solver_probe(dtype, args.out, args.repeats, args.trace)
    else:
        driver_probe(
            args.driver,
            dtype,
            args.out,
            args.repeats,
            args.batch_size,
            args.method,
            args.trace,
            args.size,
        )


if __name__ == "__main__":
    main()
