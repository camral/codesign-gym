"""Spring integration accuracy and cost against a tight Kvaerno5 reference.

All rows use one physical model and include stiffness/control sensitivities.
Kvaerno5 and unmodified Diffrax Kvaerno3 are SMALL dense reference paths only.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as ox

from envs.soft_walker.diagnostics.benchmark import measured_compile
from envs.soft_walker.engine.embodiments.decoders import FieldSpec, cast_embedding, make_decoder
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

    from envs.soft_walker.engine.types import (
        Array,
        PathLike,
    )

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def integration_report(out: PathLike, dtype: DTypeLike, repeats: int = 3) -> list[str]:
    """Compare implicit time-integration paths against the reference trajectory and report failures."""
    out = Path(out)
    dtype = np.dtype(dtype)
    is64 = dtype == np.dtype("float64")
    cfg = SpringConfig(
        grid_shape=(2, 1), ground=False, gravity=0.0, settle_steps=0, substeps=10, control_dt=0.02
    )
    dec = make_decoder("direct", spec=FieldSpec(design_mode="relaxed"), dtype=dtype)
    emb = cast_embedding(dec, dec.base())
    topo = spring.topology(cfg, dtype)
    fields = dec.decode(emb, topo.points)
    m = spring.instantiate(fields, topo, cfg, 2)
    state = spring.initial_state(m)
    state = state._replace(x=state.x * jnp.array([1.015, 0.99], dtype))
    p = jnp.array([0.08, 0.24, -0.17], dtype)

    def model(p: Array) -> spring.Model:
        scale = jnp.exp(p[0])
        return m._replace(stiffness=m.stiffness * scale, damping=m.damping * jnp.sqrt(scale))

    n = state.x.size

    def pack(s: spring.State) -> Array:
        return jnp.concatenate(
            ((s.x - state.x).reshape(-1) / cfg.length_ref, s.v.reshape(-1) / cfg.velocity_ref)
        )

    direction = jnp.linspace(0.5, 1.5, 2 * n, dtype=dtype)

    def reference(p: Array) -> Array:
        mm = model(p)
        mass = jnp.where(mm.active, mm.mass, 1.0)

        def rhs(t: float | Array, y: Array, args: tuple[Any, ...]) -> Array:
            x = state.x + y[:n].reshape(state.x.shape) * cfg.length_ref
            v = y[n:].reshape(state.v.shape) * cfg.velocity_ref
            a = p[1:] + (state.actuator - p[1:]) * jnp.exp(-t / cfg.actuator_tau)
            force, _ = spring.forces(mm, x, v, a, cfg)
            return jnp.concatenate(
                (
                    v.reshape(-1) / cfg.length_ref,
                    (force / mass[:, None]).reshape(-1) / cfg.velocity_ref,
                )
            )

        tol = 2e-10 if is64 else 2e-6
        solver = dfx.Kvaerno5(root_finder=ox.Newton(rtol=tol, atol=tol), root_find_max_steps=30)
        # Unit-time scaling prevents acceleration tolerances from falling below
        # float32's useful range; high-order reference is not declared exact.
        rr = lambda t, y, args: cfg.control_dt * rhs(t * cfg.control_dt, y, args)
        sol = dfx.diffeqsolve(
            dfx.ODETerm(rr),
            solver,
            t0=jnp.array(0.0, dtype),
            t1=jnp.array(1.0, dtype),
            dt0=jnp.array(0.02, dtype),
            y0=pack(state),
            max_steps=1024,
            stepsize_controller=dfx.PIDController(rtol=tol, atol=tol),
            throw=False,
        )
        ok = sol.result == dfx.RESULTS.successful
        return sol.ys[0] * jnp.where(ok, 1.0, jnp.nan)

    ref_fun = lambda p: (jnp.vdot(reference(p), direction), reference(p))
    (refloss, refstate), refgrad = jax.block_until_ready(
        jax.jit(jax.value_and_grad(ref_fun, has_aux=True))(p)
    )
    rows = []
    failures: list[str] = []
    for integrator, linear in [
        ("backward_euler", "dense"),
        ("kvaerno3", "dense"),
        ("kvaerno3_native", "gmres"),
    ]:
        for steps in [10, 20, 40]:
            c = replace(cfg, substeps=steps, integrator=integrator, linear_solver=linear)

            def forward(p: Array) -> tuple[Array, Array]:
                ss, info = spring.advance(model(p), state, p[1:], c)
                y = pack(ss) * jnp.where(ss.status == 0, 1.0, jnp.nan)
                return jnp.vdot(y, direction), y

            fn = jax.jit(jax.value_and_grad(forward, has_aux=True))
            name = f"integration_{integrator}_{steps}"
            result, cost = measured_compile(fn, (p,), out, name, repeats, False, False)
            (loss, y), grad = result
            error = float(jnp.linalg.norm(y - refstate))
            ge = float(jnp.linalg.norm(grad - refgrad) / jnp.linalg.norm(refgrad))
            ok = bool(np.isfinite(y).all() and np.isfinite(grad).all())
            row: Report = dict(
                integrator=integrator,
                linear_solver=linear,
                dt=c.dt,
                substeps=steps,
                scaled_state_error=error,
                gradient_relative_l2_error=ge,
                gradient=np.asarray(grad),
                passed=ok,
                cost=cost,
            )
            rows.append(row)
            record("integrator_comparison", **row)
            if not ok:
                failures.append(name + "/nonfinite")
            write_json(
                out / "spring_integration.json",
                dict(
                    reference="adaptive tight Kvaerno5, same physical model",
                    reference_state=refstate,
                    reference_gradient=refgrad,
                    parameters=p,
                    config=asdict(cfg),
                    rows=rows,
                ),
            )
        jax.clear_caches()
    if not np.isfinite(refstate).all() or not np.isfinite(refgrad).all():
        failures.append("Kvaerno5_reference")
    # Accuracy target bands define an equal-accuracy cost comparison rather than
    # assuming that matching nominal dt matches damping or gradient quality.
    selections = []
    for target in [1e-3, 1e-4]:
        for integrator in ["backward_euler", "kvaerno3", "kvaerno3_native"]:
            eligible = [
                r
                for r in rows
                if r["integrator"] == integrator
                and r["scaled_state_error"] <= target
                and r["gradient_relative_l2_error"] <= target
            ]
            best = min(eligible, key=lambda r: r["cost"]["median_s"]) if eligible else None
            selections.append(
                dict(
                    target=target,
                    integrator=integrator,
                    matched=best is not None,
                    dt=best["dt"] if best else None,
                    median_value_and_grad_s=best["cost"]["median_s"] if best else None,
                )
            )
    write_json(out / "spring_equal_accuracy.json", selections)
    return failures
