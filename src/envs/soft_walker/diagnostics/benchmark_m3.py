"""New drivers only: synchronized forward/reverse, resident vs API, and batches.

Legacy M1/M2 sweeps are retained under --suite full, not rerun here. Persistent
cache affects compile timing, never warm execution correctness. No GPU speedup
threshold is imposed. All output buffers are synchronized before stopping timers.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from envs.soft_walker.diagnostics.benchmark import measured_compile
from envs.soft_walker.diagnostics.benchmark_m2 import audit_shapes
from envs.soft_walker.diagnostics.m3_cases import IDS
from envs.soft_walker.engine import make
from envs.soft_walker.engine.drivers import environment as drivers
from envs.soft_walker.engine.drivers.task import BodyConfig, TaskState
from envs.soft_walker.engine.physics.terrain import TerrainConfig
from envs.soft_walker.engine.types import BodyModel
from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
    evidence_write_json as write_json,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import DTypeLike

    from envs.soft_walker.engine.physics.common import StepInfo
    from envs.soft_walker.engine.types import (
        Array,
        PathLike,
    )


def benchmark_main(
    out: PathLike, dtype: DTypeLike, repeats: int = 10, dump_ir: bool = False, profile: bool = False
) -> int:
    """Run this precision-specific benchmark, retaining compile/runtime scope and failure evidence."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    results = []
    failures: list[str] = []
    structures = []
    import fnmatch
    import os

    selected = [
        k
        for k in ["fem", "mpm2", "mpm3", "beam"]
        if fnmatch.fnmatchcase(k, os.environ.get("CODESIGN_CASE_FILTER", "*"))
    ]
    if not selected:
        raise ValueError("No M3 benchmark cases match the filter")
    for kind in selected:
        dim = 2 if kind == "mpm2" else 3
        shape = (4, 2) if dim == 2 else (4, 2, 1)
        # One 1 ms control interval, four .25 ms substeps. These are diagnostic
        # settings, not the proposed 20 ms controller cadence or accuracy gate.
        cfg = BodyConfig(
            grid_shape=shape,
            size=tuple(0.05 * n for n in shape),
            control_dt=0.001,
            substeps=4,
            settle_steps=1,
            horizon=24,
            grid_margin=1,
            terrain=TerrainConfig(blocks=1),
        )
        env, base = make(IDS[kind], dtype=dtype, config=cfg, max_observation_space_size=10000)
        start = time.perf_counter()
        _, reject, info = env.reset(seed=9, embodiment=base)
        reset_s = time.perf_counter() - start
        if reject[0]:
            failures.append(kind + "/reset_rejected")
            env.close()
            continue
        m, s = cast(BodyModel, env._model), cast(TaskState, env._state)
        a = jnp.asarray([0.24, -0.17], dtype)
        fn = lambda m, s, a: drivers.advance(kind, m, s, a, cfg, env._grid)
        name = kind + "_driver_forward"
        r, data = measured_compile(
            jax.jit(fn), (m, s, a), out, name, repeats, dump_ir, profile and kind == "mpm3"
        )
        unknowns = (
            env._grid.count * dim
            if env._grid is not None
            else s.v.size * (2 if kind == "beam" else 1)
        )
        data.update(
            config=asdict(cfg),
            unknowns=unknowns,
            model_sites=len(m.mass),
            first_reset_s=reset_s,
            solver=r[1][0]._asdict(),
            accepted_dt_trace_s=r[1][1],
            simulated_seconds_per_wall_second=cfg.control_dt / data["median_s"],
        )
        if int(r[0].status) != 0:
            failures.append(name + "/invalid")
        # An ordinary API trajectory must match the same physics, with its
        # host transfers and metric/observation cost reported separately.
        api = []
        for k in range(8):
            command = np.asarray(
                [0.24 * np.cos(0.3 * k), -0.17 * np.cos(0.3 * k + 0.3), 0, 0], dtype
            )
            t = time.perf_counter()
            obs, rew, term, trunc, inf = env.step(command)
            wall = time.perf_counter() - t
            api.append(
                dict(step=k, wall_api_s=wall, reward=rew, terminated=term, truncated=trunc, **inf)
            )
            if term:
                failures.append(name + "/api_failed")
                break
        data["api_warm_median_s"] = (
            float(np.median([q["wall_api_s"] for q in api[1:]])) if len(api) > 1 else None
        )
        write_json(out / (name + "_episode.json"), api)
        write_json(out / (name + ".json"), data)
        results.append(data)
        if kind in ("mpm3", "beam"):
            aud = audit_shapes(fn, (m, s, a), unknowns, out, name)
            structures.append(aud)
            if not aud["passed"]:
                failures.append(name + "/global_dense")

            # Same eight intervals resident on device: no host observation copy.
            def resident(
                m: BodyModel, s: TaskState, a: Array
            ) -> tuple[TaskState, tuple[StepInfo, Array]]:
                def one(s: TaskState, k: int) -> tuple[TaskState, tuple[StepInfo, Array]]:
                    return fn(m, s, a * (1 - 0.02 * k))

                return jax.lax.scan(one, s, jnp.arange(8))

            rr, dd = measured_compile(
                jax.jit(resident), (m, s, a), out, kind + "_resident8", repeats, False, False
            )
            dd.update(
                intervals=8,
                physical_dt_s=cfg.control_dt,
                per_interval_median_s=dd["median_s"] / 8,
                unknowns=unknowns,
            )
            if int(rr[0].status) != 0:
                failures.append(kind + "/resident_invalid")
            write_json(out / (kind + "_resident8.json"), dd)
            results.append(dd)

            # Reuse the resident trajectory for an actual action adjoint.
            def loss(a: Array) -> Array:
                state, _ = resident(m, s, a)
                val = 20 * jnp.sum(state.v**2) + 1.0e5 * state.body.work_signed
                return jnp.where(state.status == 0, val, jnp.nan)

            gradfn = jax.value_and_grad(loss)
            rr, dd = measured_compile(
                jax.jit(gradfn), (a,), out, kind + "_resident_reverse", repeats, dump_ir, False
            )
            aud = audit_shapes(gradfn, (a,), unknowns, out, kind + "_resident_reverse")
            structures.append(aud)
            dd.update(unknowns=unknowns, intervals=8, structure=aud, gradient=rr[1])
            if not aud["passed"]:
                failures.append(kind + "/reverse_global_dense")
            if not all(np.all(np.isfinite(v)) for v in jax.tree.leaves(rr)):
                failures.append(kind + "/reverse_nonfinite")
            write_json(out / (kind + "_resident_reverse.json"), dd)
            results.append(dd)
            batch = 4
            mm = jax.tree.map(lambda x: jnp.stack([x] * batch), m)
            ss = jax.tree.map(lambda x: jnp.stack([x] * batch), s)
            aa = jnp.stack([a] * batch)
            rr, dd = measured_compile(
                jax.jit(jax.vmap(fn)), (mm, ss, aa), out, kind + "_batch4", repeats, False, False
            )
            dd.update(
                batch=batch,
                unknowns_per_environment=unknowns,
                per_env_median_s=dd["median_s"] / batch,
            )
            if np.any(np.asarray(rr[0].status) != 0):
                failures.append(kind + "/batch_invalid")
            write_json(out / (kind + "_batch4.json"), dd)
            results.append(dd)
        env.close()
        import gc

        drivers.physics_kernels.cache_clear()
        drivers.kernels.cache_clear()
        drivers.instantiation_kernel.cache_clear()
        jax.clear_caches()
        gc.collect()
    write_json(
        out / "structure_summary.json",
        {"passed": all(q["passed"] for q in structures), "audits": structures},
    )
    write_json(
        out / "summary.json",
        {
            "passed": not failures,
            "failures": failures,
            "dtype": str(dtype),
            "benchmarks": results,
            "selected_cases": selected,
            "scope": "M3 new drivers; no rerun of unchanged M2 synthetic size sweeps",
            "interpretation": [
                "all output buffers synchronized; lowering/compilation/first/warm separated",
                "compilation timing may load trusted persistent cache; see manifest/cache stats and logs",
                "shape audits check forward AND reverse; executable memory estimate is not allocator peak",
                "API versus resident includes different observation/host work; not a pure kernel speed ratio",
                "fixed development fixtures, not validated long-horizon locomotion",
            ],
        },
    )
    return int(bool(failures))
