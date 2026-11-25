"""Matrix-free forward AND adjoint size sweeps, matched kernels and host/device cost.

This is also a structural audit: inspect the nested JAXPR and lowered program,
not merely the forward solver's advertised option. Global NxN arrays fail the
gate; bounded restart-space (k x k) solves are expected in GMRES.
"""

from __future__ import annotations

import gc
import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from envs.soft_walker.diagnostics.benchmark import measured_compile, sync as sync
from envs.soft_walker.diagnostics.m2_cases import design_case
from envs.soft_walker.engine.embodiments.decoders import Fields
from envs.soft_walker.engine.numerics.integrators import kvaerno3_step
from envs.soft_walker.engine.numerics.solvers import NewtonConfig, solve_root
from envs.soft_walker.engine.physics import fem, mpm
from envs.soft_walker.engine.physics.common import DynamicsConfig
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

    from envs.soft_walker.engine.numerics.solvers import RootInfo
    from envs.soft_walker.engine.physics.common import StepInfo
    from envs.soft_walker.engine.types import (
        Array,
        PathLike,
        PhysicsModel,
        PhysicsState,
    )

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def audit_shapes(
    fun: Callable[..., Array], args: tuple[Any, ...], n: int, out: Path, name: str
) -> Report:
    """Inspect traced array shapes and record unexpected dense intermediate allocations."""
    shapes = set()
    primitives: dict[str, int] = {}
    seen = set()

    def walk(x: Array) -> None:
        if id(x) in seen:
            return
        if hasattr(x, "jaxpr"):
            seen.add(id(x))
            walk(x.jaxpr)
        elif hasattr(x, "eqns"):
            seen.add(id(x))
            for var in list(x.invars) + list(x.outvars) + list(x.constvars):
                if hasattr(var, "aval") and hasattr(var.aval, "shape"):
                    shapes.add(tuple(var.aval.shape))
            for eq in x.eqns:
                primitives[eq.primitive.name] = primitives.get(eq.primitive.name, 0) + 1
                for var in list(eq.invars) + list(eq.outvars):
                    if hasattr(var, "aval") and hasattr(var.aval, "shape"):
                        shapes.add(tuple(var.aval.shape))
                walk(eq.params)
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)

    walk(jax.make_jaxpr(fun)(*args))
    bad = [s for s in shapes if len(s) >= 2 and sum(k == n for k in s) >= 2]
    hlo = str(jax.jit(fun).lower(*args).compiler_ir(dialect="stablehlo"))
    hlo_global = bool(re.search(r"(?:tensor|memref)<" + str(n) + "x" + str(n) + r"x", hlo))
    largest = sorted([(int(np.prod(s)), s) for s in shapes], reverse=True)[:12]
    data: Report = {
        "name": name,
        "unknowns": n,
        "passed": not bad and not hlo_global,
        "global_square_shapes": bad,
        "global_square_in_stablehlo": hlo_global,
        "largest_jaxpr_arrays": largest,
        "primitive_counts": primitives,
        "scope": "nested program shape audit, NOT an allocator peak-memory measurement",
    }
    write_json(out / (name + "_structure.json"), data)
    return data


def fields(points: Array) -> Array:
    """Build the bounded material-field fixture used by the numerical benchmark."""
    n, d = points.shape
    one = jnp.ones(n, points.dtype)
    return Fields(
        one,
        one * 20000,
        one * 1000,
        one * 0.5,
        jnp.broadcast_to(jnp.eye(d, dtype=points.dtype)[0], (n, d)),
        jnp.ones(n, jnp.int32),
        (points[:, 0] > 0).astype(jnp.int32),
    )


def benchmark_main(
    out: PathLike, dtype: DTypeLike, repeats: int = 10, dump_ir: bool = False, profile: bool = False
) -> int:
    """Run this precision-specific benchmark, retaining compile/runtime scope and failure evidence."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    results = []
    struct = []
    failures: list[str] = []
    cfg = NewtonConfig()
    # Manufactured nonlinear banded systems: 64 -> 4096 unknowns. Both primal
    # and reverse have real solves; inputs are runtime parameters, not constants.
    for n in [64, 256, 1024, 4096]:
        diag = jnp.linspace(1.5, 2.5, n, dtype=dtype)

        def residual(y: Array, p: Array) -> Array:
            return diag * y + 0.13 * jnp.roll(y, 1) - 0.05 * jnp.roll(y, -1) + 0.2 * y**3 - p

        def solve(p: Array) -> tuple[Array, RootInfo]:
            return solve_root(lambda y: residual(y, p), jnp.zeros(n, dtype), cfg, diagonal=diag)

        def loss(p: Array) -> Array:
            return jnp.mean(solve(p)[0] ** 2)

        p = jnp.linspace(0.1, 0.4, n, dtype=dtype)
        for suffix, fn in [("forward", solve), ("reverse", jax.value_and_grad(loss))]:
            name = f"root_{n}_{suffix}"
            result, data = measured_compile(
                jax.jit(fn), (p,), out, name, repeats, dump_ir, profile and n == 1024
            )
            audit = audit_shapes(fn, (p,), n, out, name)
            struct.append(audit)
            if not audit["passed"]:
                failures.append(name + "/dense_global")
            data.update(unknowns=n, restart=cfg.restart, root_config=asdict(cfg), structure=audit)
            if suffix == "forward":
                data["solver_diagnostics"] = result[1]._asdict()
                if not bool(result[1].accepted):
                    failures.append(name + "/solve_failed")
            if any(not np.isfinite(x).all() for x in jax.tree.leaves(result)):
                failures.append(name + "/nonfinite")
            write_json(out / (name + ".json"), data)
            results.append(data)
        jax.clear_caches()
        gc.collect()
    # Full fixed-step implicit RK stages, including their adjoints. This closes
    # the stage-adjoint dense-fallback gate with a distinct native implementation.
    for n in [64, 256]:
        y = jnp.linspace(0.1, 0.3, n, dtype=dtype)

        def rhs(t: float | Array, y: Array, p: Array) -> Array:
            return -p * y - 0.2 * y**3 + 0.05 * jnp.roll(y, 1)

        def fun(p: Array) -> Array:
            return kvaerno3_step(rhs, jnp.asarray(0.0, dtype), y, jnp.asarray(0.01, dtype), p, cfg)[
                0
            ]

        p = jnp.asarray(12.0, dtype)
        for suffix, fn in [
            ("forward", fun),
            ("reverse", jax.value_and_grad(lambda p: jnp.mean(fun(p) ** 2))),
        ]:
            name = f"kvaerno_{n}_{suffix}"
            result, data = measured_compile(jax.jit(fn), (p,), out, name, repeats, dump_ir, False)
            audit = audit_shapes(fn, (p,), n, out, name)
            struct.append(audit)
            if not audit["passed"]:
                failures.append(name + "/dense_global")
            if any(not np.isfinite(x).all() for x in jax.tree.leaves(result)):
                failures.append(name + "/nonfinite")
            data.update(unknowns=n, structure=audit)
            write_json(out / (name + ".json"), data)
            results.append(data)
        jax.clear_caches()
        gc.collect()
    # Real mechanics size sweep: material/grid quadrature, nonlinear solves and
    # contact-free active deformation. This is not just a synthetic stencil.
    dc = DynamicsConfig(gravity=0, max_halvings=0)
    model: fem.Model | mpm.Model
    state: fem.State | mpm.State
    for kind in ["fem", "mpm"]:
        for level in [1, 2, 4, 8]:
            if kind == "fem":
                shape = (8, 4, 4) if level == 8 else (2 * level, 2, 2)
                mesh = fem.structured_mesh(shape=shape, dtype=dtype)
                model = fem.instantiate(mesh, fields(mesh.points))
                state = fem.initial_state(model)
                n = state.v.size
                active = n
                fn = lambda model, state, a: fem.trial(model, state, a, 0.001, dc)
                meta = {"element_shape": shape, "reference_volume": float(jnp.sum(mesh.weights))}
            else:
                cell_shape = (24, 12) if level == 8 else (2 * level, 2)
                cloud = mpm.uniform_cloud(cell_shape=cell_shape, dtype=dtype)
                model = mpm.instantiate(cloud, fields(cloud.points))
                state = mpm.initial_state(model)
                grid = mpm.Grid(shape=(cell_shape[0] + 6, cell_shape[1] + 6))
                n = grid.count * 2
                mass, _ = mpm.p2g(model.mass, state.v, state.C, mpm.stencil(state.x, grid), grid)
                active = int(jnp.sum(mass > 0)) * 2
                fn = lambda model, state, a: mpm.trial(model, state, a, 0.001, grid, dc)
                meta = {
                    "grid": asdict(grid),
                    "particles": len(model.mass),
                    "reference_volume": float(jnp.sum(cloud.volume)),
                }
            a = jnp.asarray([0.24, -0.17], dtype)
            name = f"{kind}_level{level}_forward"
            result, data = measured_compile(
                jax.jit(fn),
                (model, state, a),
                out,
                name,
                repeats,
                dump_ir and level == 4,
                profile and level == 4,
            )
            audit = audit_shapes(fn, (model, state, a), n, out, name)
            struct.append(audit)
            data.update(
                unknowns=n,
                active_unknowns=active,
                model=meta,
                dt=0.001,
                solver_diagnostics=result[1]._asdict(),
                structure=audit,
            )
            if result[0].status != 0:
                failures.append(name + "/solver_failure")
            if not audit["passed"]:
                failures.append(name + "/global_dense")
            write_json(out / (name + ".json"), data)
            results.append(data)
            if level == 8:
                # A real-mechanics adjoint at larger capacity, not only synthetic
                # banded systems. Loss depends on both active control channels.
                def action_loss(a: Array) -> Array:
                    final, info = fn(model, state, a)
                    return jnp.sum(final.v**2) * jnp.where(final.status == 0, 1.0, jnp.nan)

                name = kind + "_large_reverse"
                rr, rd = measured_compile(
                    jax.jit(jax.value_and_grad(action_loss)),
                    (a,),
                    out,
                    name,
                    repeats,
                    dump_ir,
                    profile,
                )
                aud = audit_shapes(jax.value_and_grad(action_loss), (a,), n, out, name)
                struct.append(aud)
                if not aud["passed"]:
                    failures.append(name + "/dense_global")
                if any(not np.isfinite(x).all() for x in jax.tree.leaves(rr)):
                    failures.append(name + "/nonfinite")
                rd.update(
                    unknowns=n, active_unknowns=active, structure=aud, gradient=np.asarray(rr[1])
                )
                write_json(out / (name + ".json"), rd)
                results.append(rd)
            if level == 2:
                # Batching and resident trajectory versus repeated host dispatch.
                for batch in [4, 16]:
                    mm = jax.tree.map(lambda x: jnp.stack([x] * batch), model)
                    ss = jax.tree.map(lambda x: jnp.stack([x] * batch), state)
                    aa = jnp.stack([a] * batch)
                    r, bd = measured_compile(
                        jax.jit(jax.vmap(fn)),
                        (mm, ss, aa),
                        out,
                        f"{kind}_batch{batch}",
                        repeats,
                        False,
                        False,
                    )
                    bd.update(batch=batch, per_env_median_s=bd["median_s"] / batch)
                    if np.any(np.asarray(r[0].status) != 0):
                        failures.append(f"{kind}_batch{batch}/failure")
                    write_json(out / (f"{kind}_batch{batch}.json"), bd)
                    results.append(bd)

                def resident(
                    m: PhysicsModel, s: PhysicsState, a: Array
                ) -> tuple[PhysicsState, StepInfo]:
                    def body(s: PhysicsState, i: int) -> tuple[PhysicsState, StepInfo]:
                        return fn(m, s, a * (1 - 0.04 * i))

                    return jax.lax.scan(body, s, jnp.arange(8))

                r, bd = measured_compile(
                    jax.jit(resident),
                    (model, state, a),
                    out,
                    f"{kind}_resident8",
                    repeats,
                    False,
                    False,
                )
                bd.update(intervals=8, per_interval_median_s=bd["median_s"] / 8)
                if r[0].status != 0:
                    failures.append(f"{kind}_resident8/failure")
                write_json(out / (f"{kind}_resident8.json"), bd)
                results.append(bd)
            jax.clear_caches()
            gc.collect()
        # Full geometry->reset/settle->trajectory->objective reverse audit.
        loss, roll, theta, meta, _, _ = design_case(kind, dtype, steps=3)
        for suffix, fn in [("forward", loss), ("reverse", jax.value_and_grad(loss))]:
            name = kind + "_design_" + suffix
            r, bd = measured_compile(jax.jit(fn), (theta,), out, name, repeats, dump_ir, False)
            n = 112 if kind == "mpm" else 36
            aud = audit_shapes(fn, (theta,), n, out, name)
            struct.append(aud)
            bd.update(config=meta, unknowns=n, structure=aud)
            if not aud["passed"]:
                failures.append(name + "/global_dense")
            if any(not np.isfinite(x).all() for x in jax.tree.leaves(r)):
                failures.append(name + "/nonfinite")
            write_json(out / (name + ".json"), bd)
            results.append(bd)
        jax.clear_caches()
        gc.collect()
    write_json(
        out / "structure_summary.json",
        {"passed": all(x["passed"] for x in struct), "audits": struct},
    )
    write_json(
        out / "summary.json",
        {
            "passed": not failures,
            "failures": failures,
            "benchmarks": results,
            "dtype": str(dtype),
            "interpretation": [
                "global dense materialization checked in forward and reverse, including native RK stages",
                "reported iterations count Newton iterations; GMRES internal iterations unavailable, not fabricated",
                "no GPU speedup assumed; compare capacity, active unknowns, accepted solves, accuracy and resident dispatch",
                "memory_analysis reports executable allocation estimates, not device peak memory",
                "synchronized timing samples separate lowering, compilation, first call and warmed execution",
            ],
        },
    )
    return int(bool(failures))
