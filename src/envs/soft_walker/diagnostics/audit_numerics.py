"""Executed numerical audits: local kernels, decoders, complete design/control AD.

Each ladder retains every +/- evaluation, the AD projection, relative/absolute
errors and observed convergence orders. No single hand-picked epsilon is used.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, cast

import jax
import jax.numpy as jnp
import numpy as np

if __package__ in (None, ""):
    # Direct-file execution must select this checkout before package imports.
    _source_root = str(Path(__file__).resolve().parents[3])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)

from envs.soft_walker.engine import make
from envs.soft_walker.engine.differentiation import bind_rollout
from envs.soft_walker.engine.embodiments.decoders import (
    CPPNDecoder,
    FieldSpec,
    cast_embedding,
    make_decoder,
)
from envs.soft_walker.engine.physics import spring
from envs.soft_walker.engine.physics.spring_config import SpringConfig
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

    from envs.soft_walker.engine.embodiments.decoders import Decoder
    from envs.soft_walker.engine.types import (
        Array,
        Embedding,
        HostArray,
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
    steps = 0.64 / (2.0 ** np.arange(15 if is64 else 11))

    def run(
        name: str,
        fn: Callable[..., Any],
        theta: Array,
        directions: Sequence[tuple[str, ArrayLike]],
        rtol: float | None = None,
        atol: float = 0.0,
        require_order: bool = True,
        h: Iterable[float] | None = None,
    ) -> HostArray:
        theta = jnp.asarray(theta, dtype=dtype)
        value = jax.jit(fn)
        ad = jax.jit(jax.grad(fn))
        t = time.perf_counter()
        grad = jax.block_until_ready(ad(theta))
        ad_first = time.perf_counter() - t
        assert np.isfinite(grad).all(), f"Nonfinite gradient in {name}"
        record("audit_gradient", name=name, gradient=grad, first_ad_compile_and_run_s=ad_first)
        arrays[name + "_gradient"] = np.asarray(grad)
        for label, dr in directions:
            dr = jnp.asarray(dr, dtype=dtype)
            projection = float(jnp.vdot(grad, dr))
            ladder = finite_difference_ladder(
                value,
                theta,
                dr,
                steps=h if h is not None else steps,
                autodiff=projection,
                name=name + "/" + label,
            )
            orders = [r["observed_order"] for r in ladder["rows"]]
            # At least two adjacent second-order ratios in a truncation-dominated
            # region for float64. Float32 must exhibit at least one such ratio;
            # its full curve (including roundoff) remains evidence to inspect.
            good = [o is not None and 1.6 < o < 2.4 for o in orders]
            has_window = (
                any(good[i] and good[i + 1] for i in range(len(good) - 1)) if is64 else any(good)
            )
            threshold = rtol if rtol is not None else (2e-6 if is64 else 2e-3)
            accuracy = ladder["best_absolute_error"] < atol + threshold * abs(projection)
            nontrivial = abs(projection) > 1e-10
            passed = accuracy and nontrivial and (has_window or not require_order)
            ladder.update(
                {
                    "passed": passed,
                    "relative_threshold": threshold,
                    "absolute_threshold": atol,
                    "second_order_window": has_window,
                    "absolute_floor_used": bool(
                        accuracy and ladder["best_relative_error"] >= threshold
                    ),
                    "order_required": require_order,
                    "nontrivial_direction": nontrivial,
                }
            )
            reports.append(ladder)
            print(
                f"{name}/{label}: best rel={ladder['best_relative_error']:.3e}, second_order={has_window}, pass={passed}",
                flush=True,
            )
            if not passed:
                failures.append(ladder["name"])
            write_json(
                out / "gradient_ladders.json",
                {"dtype": str(dtype), "reports": reports, "failures": failures},
            )
        return np.asarray(grad)

    # Local spring energy: deformation, material scale and active strain.
    cfg = SpringConfig(grid_shape=(2, 1), gravity=0, ground=False, settle_steps=0, substeps=4)
    env, b = make(dtype=dtype, config=cfg)
    env.reset(embodiment=b)
    m, s = cast(spring.Model, env._model), cast(spring.State, env._state)
    shape = s.x.shape
    direction = jnp.asarray(np.random.default_rng(91).normal(size=shape) * 0.012, dtype=dtype)

    def energy(theta: Array) -> Array:
        x = s.x + theta[0] * direction
        model = m._replace(stiffness=m.stiffness * jnp.exp(theta[1]))
        return spring.elastic_energy(model, x, jnp.asarray([theta[2], -0.1], dtype=dtype)) * 100

    run(
        "local_energy",
        energy,
        [0.21, 0.08, 0.17],
        [("geometry", [1, 0, 0]), ("material", [0, 1, 0]), ("actuation", [0, 0, 1])],
    )

    # All decoders in 2D and 3D; changes affect geometry and physical properties.
    for dim in (2, 3):
        points = np.array([[0.65, 0.12], [-0.54, 0.38], [0.15, -0.63]], dtype=dtype)
        if dim == 3:
            points = np.column_stack((points, np.array([0.31, -0.27, 0.19], dtype=dtype)))
        points = jnp.asarray(points, dtype=dtype)
        for name in ("direct", "cppn", "primitives", "bspline"):
            d = make_decoder(name, spec=FieldSpec(dim=dim, design_mode="relaxed"), dtype=dtype)
            base = cast_embedding(d, d.base())
            if name == "primitives":
                base["kind"] = base["kind"].at[0].set(0)

            def decoded(
                theta: Array,
                d: Decoder = d,
                base: Embedding = base,
                name: str = name,
                points: Array = points,
            ) -> Array:
                e = dict(base)
                if name in ("direct", "bspline"):
                    e["values"] = e["values"].at[..., 0].add(theta[0]).at[..., 1].add(theta[1])
                elif name == "cppn":
                    e["bias"] = (
                        e["bias"]
                        .at[cast(CPPNDecoder, d).hidden]
                        .add(theta[0])
                        .at[cast(CPPNDecoder, d).hidden + 1]
                        .add(theta[1])
                    )
                else:
                    e["half_size"] = e["half_size"].at[0, 0].add(0.2 * theta[0])
                    e["center"] = e["center"].at[0, 0].add(0.12 * theta[0])
                    e["features"] = e["features"].at[:, 0].add(theta[1])
                f = d.decode(e, points)
                weights = jnp.asarray([0.7, 1.1, 1.5], dtype=dtype)
                return jnp.sum(weights * f.occupancy * (0.3 + f.young / 20000.0))

            run(
                f"decoder_{name}_{dim}d",
                decoded,
                [0.12, 0.05],
                [("geometry", [1, 0]), ("stiffness", [0, 1])],
            )

    # Independent scalar nonlinear residual: analytic implicit derivative too.
    root_cfg = replace(cfg, root_tol64=2e-13, root_tol32=4e-7)

    def root(p: Array) -> Array:
        y, aux = spring.solve_root(lambda y: y + y**3 - p, jnp.zeros(1, dtype=dtype), root_cfg)
        return y[0]

    rg = run("implicit_root", root, [0.2], [("parameter", [1])])
    y = float(jax.jit(root)(jnp.asarray([0.2], dtype=dtype)))
    analytic = 1 / (1 + 3 * y * y)
    root_error = abs(rg[0] - analytic)
    if root_error > (1e-10 if is64 else 3e-6):
        failures.append("implicit_root_analytic")
    record("implicit_root_analytic", value=y, derivative=analytic, error=root_error)

    # Full nonlinear contact rollout, including material-dependent settling.
    rc = SpringConfig(
        grid_shape=(2, 1),
        settle_steps=1,
        substeps=4,
        horizon=3,
        root_tol64=3e-12,
        root_tol32=1.2e-6,
    )
    re, rb = make(dtype=dtype, config=rc, design_mode="relaxed", max_observation_space_size=128)
    p = cast_embedding(re.decoder, rb)
    roll = bind_rollout(re)

    def candidate(theta: Array) -> tuple[Embedding, Array]:
        v = (
            p["values"]
            .at[..., 0]
            .add(theta[0])
            .at[..., 1]
            .add(theta[1])
            .at[..., 2]
            .add(theta[2])
            .at[..., 3]
            .add(theta[3])
        )
        a = jnp.stack(
            (
                jnp.stack((theta[4], theta[5])),
                jnp.stack((theta[4] * 0.8, theta[5] * 0.7)),
                jnp.stack((theta[4] * 0.65, theta[5] * 0.5)),
            )
        )
        return {"values": v}, a

    def result(theta: Array) -> Report:
        emb, acts = candidate(theta)
        return cast(Report, roll(emb, acts))

    def diagnostic_loss(theta: Array) -> Array:
        r = result(theta)
        loss = 1e4 * r["metrics"]["work_signed_j"] + 0.1 * jnp.sum(r["state"].v ** 2)
        return jnp.where(r["metrics"]["valid"], loss, jnp.nan)

    theta = np.array([0.05, 0.03, -0.02, 0.04, 0.18, -0.12], dtype=dtype)
    directions = [
        (label, np.eye(6)[i])
        for i, label in enumerate(
            ("geometry", "stiffness", "density", "strength", "control0", "control1")
        )
    ]
    directions.append(("mixed", np.array([0.3, -0.4, 0.2, -0.1, 0.7, -0.45])))
    grad = run("full_rollout", diagnostic_loss, theta, directions, rtol=3e-5 if is64 else 3e-3)

    def task_loss(theta: Array) -> Array:
        r = result(theta)
        o = r["metrics"]["objective"]
        return o[0] + rc.energy_weight * o[1]

    run(
        "terminal_objective",
        task_loss,
        theta,
        [("mixed", np.array([0.3, -0.4, 0.2, -0.1, 0.7, -0.45]))],
        rtol=3e-5 if is64 else 3e-3,
        atol=0.0 if is64 else 1e-6,
    )
    r = jax.jit(result)(jnp.asarray(theta))
    jax.block_until_ready(r)
    arrays.update(
        {
            "rollout_x": np.asarray(r["state"].x),
            "rollout_v": np.asarray(r["state"].v),
            "rollout_objective": np.asarray(r["metrics"]["objective"]),
            "rollout_gradient": grad,
            "rollout_theta": theta,
            "rollout_max_residual": np.asarray(r["diagnostics"].max_residual),
        }
    )
    record("full_rollout_solver", diagnostics=r["diagnostics"]._asdict(), metrics=r["metrics"])
    if not bool(r["metrics"]["valid"]):
        failures.append("invalid_rollout")

    # Actual Kvaerno3 spring control derivative, not just an unrelated scalar ODE.
    kc = SpringConfig(
        grid_shape=(1, 1),
        settle_steps=0,
        substeps=4,
        integrator="kvaerno3",
        root_tol64=3e-12,
        root_tol32=1.2e-6,
    )
    ke, kb = make(dtype=dtype, config=kc)
    kr = bind_rollout(ke)
    kp = cast_embedding(ke.decoder, kb)

    def k_loss(a: Array) -> Array:
        r = kr(kp, jnp.stack((a, jnp.asarray([-0.1], dtype=dtype)), axis=-1))
        return r["metrics"]["work_positive_j"] * 1e4 + 0.1 * jnp.sum(r["state"].v ** 2)

    run(
        "kvaerno3_control",
        k_loss,
        [0.2],
        [("control", [1])],
        rtol=2e-5 if is64 else 3e-3,
        h=0.12 / (2.0 ** np.arange(12 if is64 else 9)),
    )

    np.savez_compressed(out / "reference_arrays.npz", **arrays)
    summary: Report = {
        "dtype": str(dtype),
        "gradient_ladders": len(reports),
        "passed": not failures,
        "failures": failures,
        "root_analytic_error": root_error,
        "absolute_floor_cases": [x["name"] for x in reports if x.get("absolute_floor_used")],
        "limitations": [
            "fixed contact regularization",
            "no derivative through discrete hardening",
            "float32 second-order windows can be short; all epsilon evaluations are retained",
            "tiny float32 terminal-objective differences use a declared 1e-6 absolute gradient floor; cross-precision AD comparison is a separate required gate when both precisions run",
        ],
    }
    write_json(out / "summary.json", summary)
    return 0 if not failures else 1


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="float64")
    a = p.parse_args()
    raise SystemExit(audit_main(a.out, a.dtype))
