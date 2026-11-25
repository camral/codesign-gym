"""Synchronized cold/warm benchmarks, executable memory estimates and fixtures."""

from __future__ import annotations

import gzip
import resource
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np

from envs.soft_walker.engine import make
from envs.soft_walker.engine.differentiation import bind_rollout
from envs.soft_walker.engine.embodiments.decoders import cast_embedding
from envs.soft_walker.engine.physics import spring
from envs.soft_walker.engine.physics.spring_config import SpringConfig
from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
    evidence_record as record,
    evidence_write_json as write_json,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import DTypeLike

    from envs.soft_walker.diagnostics.host_types import JittedCallable
    from envs.soft_walker.engine.types import (
        Array,
        PathLike,
    )

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]
T = TypeVar("T")


def sync(x: T) -> T:
    """Wait for every device value in the pytree before recording wall-clock time."""
    return jax.block_until_ready(x)


def measured_compile(
    fn: JittedCallable,
    args: tuple[Any, ...],
    out: Path,
    name: str,
    repeats: int = 10,
    dump_ir: bool = False,
    profile: bool = False,
) -> tuple[Any, Report]:
    """Measure compilation separately from synchronized repeated execution and optional IR/profiling output."""
    t = time.perf_counter()
    lowered = fn.lower(*args)
    lower_s = time.perf_counter() - t
    t = time.perf_counter()
    compiled = lowered.compile()
    compile_s = time.perf_counter() - t
    t = time.perf_counter()
    result = sync(compiled(*args))
    first_s = time.perf_counter() - t
    for _ in range(3):
        sync(compiled(*args))
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = sync(compiled(*args))
        times.append(time.perf_counter() - start)
    mem = None
    cost = None
    try:
        m = compiled.memory_analysis()
        if m is not None:
            mem = {
                k: getattr(m, k)
                for k in dir(m)
                if k.endswith("_size_in_bytes") and isinstance(getattr(m, k), (int, float))
            }
    except Exception as exc:
        mem = {"unavailable": repr(exc)}
    try:
        cost = compiled.cost_analysis()
    except Exception as exc:
        cost = {"unavailable": repr(exc)}
    if dump_ir:
        try:
            with gzip.open(out / (name + ".stablehlo.mlir.gz"), "wt") as f:
                f.write(str(lowered.compiler_ir(dialect="stablehlo")))
        except Exception as exc:
            record("ir_dump_failed", name=name, error=repr(exc))
    if profile:
        try:
            with jax.profiler.trace(str(out / (name + "_profile")), create_perfetto_link=False):
                for _ in range(3):
                    sync(compiled(*args))
        except Exception as exc:
            record("profiler_unavailable", name=name, error=repr(exc))
    data: Report = {
        "name": name,
        "lower_s": lower_s,
        "compile_s": compile_s,
        "first_execute_s": first_s,
        "warm_s": times,
        "median_s": float(np.median(times)),
        "p10_s": float(np.quantile(times, 0.1)),
        "p90_s": float(np.quantile(times, 0.9)),
        "min_s": min(times),
        "max_s": max(times),
        "compile_memory_analysis": mem,
        "compiler_cost_analysis": cost,
        "host_ru_maxrss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "note": "ru_maxrss is KiB on Linux; executable memory analysis is not a measured process peak",
    }
    record("benchmark", **data)
    write_json(out / (name + ".json"), data)
    print(
        f"{name}: lower {lower_s:.3f}s, compile {compile_s:.3f}s, warm median {1e3 * np.median(times):.3f}ms",
        flush=True,
    )
    return result, data


def benchmark_main(
    out: PathLike, dtype: DTypeLike, repeats: int = 10, dump_ir: bool = False, profile: bool = False
) -> int:
    """Run this precision-specific benchmark, retaining compile/runtime scope and failure evidence."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    results = []
    failures: list[str] = []
    for shape in ((2, 1), (4, 2), (6, 3)):
        cfg = SpringConfig(grid_shape=shape, settle_steps=1, substeps=10, horizon=20)
        obs_capacity = max(256, 5 + 5 * (shape[0] + 1) * (shape[1] + 1) + 4 * np.prod(shape) + 4)
        e, b = make(dtype=dtype, config=cfg, max_observation_space_size=int(obs_capacity))
        t = time.perf_counter()
        o, rejected, info = e.reset(embodiment=b)
        reset_s = time.perf_counter() - t
        if rejected[0]:
            failures.append(f"reset_{shape}")
            continue
        m, s = cast("spring.Model", e._model), cast("spring.State", e._state)
        a = jax.device_put(np.array([0.2, -0.1], dtype=dtype), e.device)
        name = f"forward_{shape[0]}x{shape[1]}"
        fun = jax.jit(lambda m, s, a: spring.advance(m, s, a, cfg))
        (n, diag), data = measured_compile(
            fun,
            (m, s, a),
            out,
            name,
            repeats,
            dump_ir and shape == (2, 1),
            profile and shape == (4, 2),
        )
        data.update(
            {
                "config": asdict(cfg),
                "nodes": len(m.mass),
                "edges": len(m.length),
                "velocity_unknowns": 2 * len(m.mass),
                "first_reset_s": reset_s,
                "reset_timing": info["timing"],
                "native_channels": e.channels,
                "solver_diagnostics": diag._asdict(),
                "simulated_seconds_per_wall_second": cfg.control_dt / data["median_s"],
            }
        )
        results.append(data)
        if int(n.status) != 0:
            failures.append(name + "_invalid")
        # A short sequence, rather than repeatedly stepping only the same state.
        episode = []
        for k in range(12):
            u = np.asarray([0.25 * np.sin(0.6 * k), 0.2 * np.sin(0.6 * k + 1.2), 0, 0], dtype=dtype)
            start = time.perf_counter()
            o, r, t, tr, inf = e.step(u)
            wall = time.perf_counter() - start
            row: Report = {
                "step": k,
                "wall_api_s": wall,
                "reward": r,
                "terminated": t,
                "truncated": tr,
                **inf,
            }
            episode.append(row)
            if t:
                failures.append(name + "_episode_failed")
                break
        write_json(out / (name + "_episode.json"), episode)
        warm_api = [x["wall_api_s"] for x in episode[1:]]
        data["api_episode_median_s"] = float(np.median(warm_api)) if warm_api else None
        write_json(out / (name + ".json"), data)

        if shape == (4, 2):
            for batch in (4, 16):
                models = jax.tree.map(lambda x: jnp.stack([x] * batch), m)
                states = jax.tree.map(lambda x: jnp.stack([x] * batch), s)
                actions = jnp.stack([a] * batch)
                fun = jax.jit(jax.vmap(lambda m, s, a: spring.advance(m, s, a, cfg)))
                (nn, dd), bd = measured_compile(
                    fun, (models, states, actions), out, f"batch_{batch}", repeats, dump_ir=False
                )
                bd["batch"] = batch
                bd["per_env_median_s"] = bd["median_s"] / batch
                results.append(bd)
                if np.any(np.asarray(nn.status) != 0):
                    failures.append(f"batch_{batch}_invalid")

    # Decode+settle+two-control rollout, and reverse pass to design AND controls.
    cfg = SpringConfig(grid_shape=(2, 1), settle_steps=1, substeps=4, horizon=4)
    e, b = make(dtype=dtype, config=cfg, design_mode="relaxed")
    p = cast_embedding(e.decoder, b)
    a = jax.device_put(np.array([[0.2, -0.1], [0.1, 0.15]], dtype=dtype), e.device)
    roll = bind_rollout(e)

    def loss(values: Array, actions: Array) -> Array:
        r = roll({"values": values}, actions)
        o = r["metrics"]["objective"]
        return o[0] + cfg.energy_weight * o[1]

    for name, fun in (
        ("rollout_forward", jax.jit(loss)),
        ("rollout_value_and_grad", jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))),
    ):
        value, data = measured_compile(fun, (p["values"], a), out, name, repeats, dump_ir, False)
        results.append(data)
        if any(not np.isfinite(x).all() for x in jax.tree.leaves(value)):
            failures.append(name + "_nonfinite")
    write_json(
        out / "summary.json",
        {
            "passed": not failures,
            "failures": failures,
            "dtype": str(dtype),
            "benchmarks": results,
            "interpretation": [
                "M1 explicitly uses small dense primal and adjoint solves",
                "CUDA can be slower for tiny meshes; no speedup threshold is assumed",
                "timings synchronize device completion and separate lowering/compilation/execution",
                "API cost includes device-to-host observations, validation and diagnostics",
                "compile/load timing excludes Python process import; cache policy is recorded in the manifest",
                "CPU timing is meaningful only with no competing validation jobs",
            ],
        },
    )
    return 0 if not failures else 1
