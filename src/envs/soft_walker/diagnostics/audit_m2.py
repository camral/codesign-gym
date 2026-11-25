"""M2 derivative ladders, integration refinement, and mechanics evidence.

Every +h/-h evaluation is saved. Smooth float64 cases require two consecutive
second-order intervals; float32 requires one or a declared three-point
roundoff plateau with mandatory independent float64 certification. Accuracy
and order are separate checks. No thresholds are relaxed automatically in response to a failure.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from envs.soft_walker.diagnostics.m2_cases import design_case
from envs.soft_walker.engine.embodiments.decoders import Fields as Fields
from envs.soft_walker.engine.numerics.integrators import kvaerno3_step
from envs.soft_walker.engine.numerics.solvers import (
    NewtonConfig,
    linear_solve as linear_solve,
    solve_root,
)
from envs.soft_walker.engine.physics import material
from envs.soft_walker.engine.physics.common import DynamicsConfig as DynamicsConfig
from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
    DifferenceReport,
    evidence_record as record,
    evidence_write_json as write_json,
    finite_difference_ladder,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike, DTypeLike

    from envs.soft_walker.engine.types import (
        Array,
        PathLike,
    )

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def audit_main(out: PathLike, dtype: DTypeLike) -> int:
    """Run this precision-specific numerical audit, write its evidence, and return a failure exit code."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    is64 = dtype == np.dtype("float64")
    reports: list[DifferenceReport] = []
    failures: list[str] = []
    arrays = {}
    configs = {}
    conditioning = {}

    def run(
        name: str,
        fn: Callable[..., Any],
        theta: Array,
        directions: Sequence[tuple[str, ArrayLike]],
        steps: Iterable[float] | None = None,
        rtol: float | None = None,
        require_order: bool = True,
    ) -> None:
        value = jax.jit(fn)
        gradfun = jax.jit(jax.grad(fn))
        theta = jnp.asarray(theta, dtype)
        start = time.perf_counter()
        grad = jax.block_until_ready(gradfun(theta))
        record(
            "m2_gradient",
            name=name,
            gradient=grad,
            first_compile_and_execute_s=time.perf_counter() - start,
        )
        arrays[name + "_gradient"] = np.asarray(grad)
        arrays[name + "_value"] = np.asarray(value(theta))
        arrays[name + "_theta"] = np.asarray(theta)
        if not np.isfinite(grad).all():
            failures.append(name + "/nonfinite_gradient")
        h = 0.64 / 2.0 ** np.arange(15 if is64 else 11) if steps is None else steps
        for label, dr in directions:
            dr = jnp.asarray(dr, dtype)
            ad = float(jnp.vdot(grad, dr))
            ladder = finite_difference_ladder(
                value, theta, dr, steps=h, autodiff=ad, name=name + "/" + label
            )
            good = [
                r["observed_order"] is not None and 1.55 < r["observed_order"] < 2.45
                for r in ladder["rows"]
            ]
            window = any(a and b for a, b in zip(good, good[1:])) if is64 else any(good)
            threshold = rtol or (3e-5 if is64 else 1e-3)
            # In float32, a nearly quadratic response can reach the rounding
            # plateau before an observable h^2 interval. Require THREE adjacent
            # accurate points, never a lucky minimum, and a mandatory independent
            # float64/cross-precision audit before certifying such a case.
            accurate = [
                np.isfinite(r["relative_error"]) and r["relative_error"] < threshold
                for r in ladder["rows"]
            ]
            plateau = not is64 and any(all(accurate[k : k + 3]) for k in range(len(accurate) - 2))
            finite_ladder = all(np.isfinite(r["central"]) for r in ladder["rows"])
            passed = bool(
                finite_ladder
                and np.isfinite(ad)
                and abs(ad) > 1e-10
                and ladder["best_relative_error"] < threshold
                and (window or plateau or not require_order)
            )
            ladder.update(
                {
                    "passed": passed,
                    "relative_threshold": threshold,
                    "absolute_threshold": 0.0,
                    "second_order_window": window,
                    "order_required": require_order,
                    "roundoff_plateau": bool(plateau and not window),
                    "cross_precision_required": bool(plateau and not window),
                }
            )
            reports.append(ladder)
            if not passed:
                failures.append(ladder["name"])
            print(
                f"{ladder['name']}: relative={ladder['best_relative_error']:.3e} second_order={window} passed={passed}",
                flush=True,
            )
            write_json(
                out / "gradient_ladders.json",
                {"reports": reports, "failures": failures, "dtype": str(dtype)},
            )

    # Local material gradients include geometry, stiffness, nu and actuator.
    for dim in [2, 3]:
        F0 = jnp.eye(dim, dtype=dtype).at[0, 0].set(1.16).at[0, 1].set(0.13)
        D = jnp.asarray(np.random.default_rng(4).normal(size=(dim, dim)) * 0.04, dtype)
        n = jnp.eye(dim, dtype=dtype)[0]
        kind = jnp.asarray(1)

        def fun(t: Array) -> Array:
            return (
                material.energy_density(
                    F0 + t[0] * D,
                    20000 * jnp.exp(t[1]),
                    0.35 + 0.07 * jnp.tanh(t[2]),
                    0.12 * t[3],
                    n,
                    kind,
                )
                / 20000.0
            )

        run(
            f"material_{dim}d",
            fun,
            [0.2, 0.1, 0.13, 0.3],
            [
                (x, np.eye(4)[i])
                for i, x in enumerate(["geometry", "stiffness", "poisson", "actuation"])
            ],
        )
    # Implicit MPM and FEM: ALL parameter blocks, settling, contact, and several
    # time steps. Projection directions are not rescaled after seeing results.
    for name in ["fem", "mpm"]:
        loss, roll, theta, meta, _, _ = design_case(name, dtype)
        configs[name] = meta
        dirs = [(label, np.eye(7)[i]) for i, label in enumerate(meta["parameters"])]
        dirs.append(("mixed", np.asarray([0.3, -0.2, 0.4, -0.25, 0.15, 0.65, -0.45])))
        run(name + "_rollout", loss, theta, dirs)
        # Report parameter conditioning without modifying user gradients or
        # objectives. These are dimensionless decoder/control coordinates, not
        # raw SI derivatives. One scalar per block in this fixture.
        block_grad = np.abs(arrays[name + "_rollout_gradient"])
        positive = block_grad[block_grad > 1e-12]
        target = float(np.median(positive)) if positive.size else 0.0
        suggestions = np.clip(target / np.maximum(block_grad, 1e-12), 0.1, 10.0)
        conditioning[name] = {
            "parameters": meta["parameters"],
            "absolute_gradient": block_grad,
            "max_to_min_nonzero": float(positive.max() / positive.min()) if positive.size else None,
            "suggested_coordinate_multiplier": suggestions,
            "multiplier_bounds": [0.1, 10.0],
            "automatic_rescaling": False,
            "warning": "Loss- and design-specific; not a physical sensitivity ranking. Never replaces the raw gradients.",
        }
        s, info, settle = jax.jit(roll)(theta)
        jax.block_until_ready(s)
        arrays[name + "_velocity"] = np.asarray(s.v)
        arrays[name + "_configuration"] = np.asarray(s.u if name == "fem" else s.x)
        arrays[name + "_actuator"] = np.asarray(s.actuator)
        record(
            "m2_rollout",
            name=name,
            config=meta,
            state=s._asdict(),
            diagnostics=info._asdict(),
            settling=settle._asdict(),
        )
        write_json(
            out / (name + "_rollout.json"),
            {
                "config": meta,
                "diagnostics": info._asdict(),
                "settling": settle._asdict(),
                "status": s.status,
                "time": s.time,
                "work_signed": s.work_signed,
                "work_positive": s.work_positive,
            },
        )
        if int(s.status) != 0:
            failures.append(name + "/rollout_invalid")
    # Geometry-family paths into actual continuum solves. The direct decoder
    # already has a full audit above; remaining presets get mixed directions.
    for dec in ["cppn", "bspline", "primitives"]:
        loss, roll, theta, meta, _, _ = design_case(
            "mpm", dtype, decoder_name=dec, steps=2, contact_enabled=False
        )
        run("mpm_" + dec, loss, theta, [("mixed", [0.4, -0.2, 0.3, -0.1, 0.2, 0.7, -0.4])])
        configs["mpm_" + dec] = meta
    # Native Kvaerno3 against exact nonlinear-independent linear oscillator.
    nc = NewtonConfig(atol64=3e-13, atol32=2e-7, linear_rtol64=1e-11, linear_rtol32=1e-6)

    def rhs(t: float | Array, y: Array, p: Array) -> Array:
        return jnp.stack((y[1], -jnp.exp(p[0]) * 100 * y[0] - 0.8 * y[1] + p[1]))

    y0 = jnp.asarray([0.2, 0.1], dtype)
    p = jnp.asarray([0.12, 0.23], dtype)

    def integrate(p: Array, h: float, n: int) -> Array:
        def body(y: Array, i: int) -> tuple[Array, None]:
            return kvaerno3_step(rhs, i * h, y, h, p, nc)[0], None

        return jax.lax.scan(jax.checkpoint(body), y0, jnp.arange(n))[0]

    run(
        "native_kvaerno",
        lambda p: jnp.vdot(integrate(p, 0.01, 5), jnp.asarray([1.0, 0.1], dtype)),
        p,
        [("stiffness", [1.0, 0.0]), ("mixed", [0.7, -0.3])],
    )
    import scipy.linalg
    import scipy.optimize

    aug = np.array(
        [[0.0, 1.0, 0.0], [-100 * np.exp(float(p[0])), -0.8, float(p[1])], [0.0, 0.0, 0.0]]
    )
    exact = (scipy.linalg.expm(0.08 * aug) @ np.array([0.2, 0.1, 1.0]))[:2]
    tref = []
    for n in [4, 8, 16, 32]:
        h = 0.08 / n
        y = np.asarray(jax.jit(lambda p: integrate(p, h, n))(p))
        be = np.linalg.matrix_power(np.linalg.inv(np.eye(3) - h * aug), n) @ np.array(
            [0.2, 0.1, 1.0]
        )
        # displacement/velocity scaling explicitly declared.
        err = np.linalg.norm((y - exact) * [10.0, 1.0])
        berr = np.linalg.norm((be[:2] - exact) * [10.0, 1.0])
        tref.append(
            {
                "steps": n,
                "dt": h,
                "native_kvaerno_error": err,
                "backward_euler_error": berr,
                "native_state": y,
                "exact": exact,
            }
        )
    for i in range(1, len(tref)):
        tref[i]["kvaerno_observed_order"] = float(
            np.log2(tref[i - 1]["native_kvaerno_error"] / tref[i]["native_kvaerno_error"])
        )
        tref[i]["be_observed_order"] = float(
            np.log2(tref[i - 1]["backward_euler_error"] / tref[i]["backward_euler_error"])
        )
    if not any(2.7 < r.get("kvaerno_observed_order", 0) < 3.4 for r in tref):
        failures.append("kvaerno_third_order")
    if not all(r["native_kvaerno_error"] < r["backward_euler_error"] for r in tref):
        failures.append("kvaerno_vs_be")
    write_json(out / "integrator_refinement.json", tref)
    # Newton tolerance sweeps: accuracy gates must not hide unconverged roots.
    sweeps = []
    for atol in [1e-5, 1e-8, 1e-11] if is64 else [1e-3, 1e-5, 3e-7]:
        c = replace(nc, atol64=atol, atol32=atol)

        def scalar(p: Array) -> Array:
            return solve_root(lambda x: x + x**3 - p, jnp.asarray([0.1], dtype), c)[0][0]

        p1 = jnp.asarray([0.3], dtype)
        v = float(jax.jit(scalar)(p1))
        g = float(jax.jit(jax.grad(scalar))(p1)[0])
        exactroot = float(scipy.optimize.brentq(lambda x: x + x**3 - 0.3, 0, 1))
        sweeps.append(
            {
                "atol": atol,
                "root": v,
                "root_error": abs(v - exactroot),
                "gradient": g,
                "analytic_at_true_root": 1 / (1 + 3 * exactroot**2),
            }
        )
    write_json(out / "root_tolerance_sweep.json", sweeps)
    np.savez_compressed(out / "reference_arrays.npz", **arrays)
    write_json(out / "case_configs.json", configs)
    write_json(out / "gradient_conditioning.json", conditioning)
    summary: Report = {
        "dtype": str(dtype),
        "gradient_ladders": len(reports),
        "passed": not failures,
        "failures": failures,
        "absolute_floor_cases": [],
        "roundoff_limited_cases": [r["name"] for r in reports if r.get("roundoff_plateau")],
        "scope": "M2 kernels; MPM/FEM full environment drivers remain M3",
        "derivative_scope": "first-order reverse and JVP; fixed discrete topology, regularized contact; not all nonsmooth events",
    }
    write_json(out / "summary.json", summary)
    return 0 if not failures else 1
