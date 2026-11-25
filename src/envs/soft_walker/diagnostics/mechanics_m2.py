"""Independent Hex8 cantilever reference and dynamic energy/refinement evidence.

The cantilever is a SMALL-STRAIN static diagnostic of the rest tangent, not a
claim of nonlinear beam accuracy. Its independent SciPy reference is assembled
from the isotropic tensor formula, not from JAX differentiation. Production
continuum stepping remains matrix-free backward Euler.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from jax.scipy.sparse.linalg import cg

from envs.soft_walker.diagnostics.benchmark_m2 import fields
from envs.soft_walker.engine.numerics.solvers import NewtonConfig
from envs.soft_walker.engine.physics import fem, mpm
from envs.soft_walker.engine.physics.common import DynamicsConfig
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
    from scipy.sparse import csr_matrix

    from envs.soft_walker.engine.physics.common import StepInfo
    from envs.soft_walker.engine.physics.fem import Mesh as HexMesh
    from envs.soft_walker.engine.types import (
        Array,
        PathLike,
        PhysicsState,
    )

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def numpy_hex_stiffness(mesh: HexMesh, young: float | Array, poisson: float | Array) -> csr_matrix:
    """CSR reference assembly; only local 24x24 blocks are dense."""
    grad = np.asarray(mesh.grad, np.float64)
    weights = np.asarray(mesh.weights, np.float64)
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    ke = np.zeros((8, 3, 8, 3))
    for q in range(8):
        g = grad[q]
        dot = g @ g.T
        for i in range(3):
            for j in range(3):
                ke[:, i, :, j] += weights[0, q] * (
                    lam * np.outer(g[:, i], g[:, j])
                    + mu * np.outer(g[:, j], g[:, i])
                    + (mu * dot if i == j else 0.0)
                )
    dof = 3 * np.asarray(mesh.cells)[:, :, None] + np.arange(3)
    dof = dof.reshape(-1, 24)
    n = len(mesh.reference) * 3
    return sp.coo_matrix(
        (
            np.broadcast_to(ke.reshape(24, 24), (len(dof), 24, 24)).reshape(-1),
            (np.repeat(dof, 24, axis=1).reshape(-1), np.tile(dof, (1, 24)).reshape(-1)),
        ),
        shape=(n, n),
    ).tocsr()


def mechanics_main(out: PathLike, dtype: DTypeLike) -> int:
    """Check constitutive, conservation, and solver invariants for the requested precision."""
    m: fem.Model | mpm.Model
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    is64 = dtype == np.dtype("float64")
    failures: list[str] = []
    bending = []
    cfg = DynamicsConfig(gravity=0, max_halvings=0)
    for shape in [(4, 1, 1), (8, 2, 2), (12, 3, 3)]:
        mesh = fem.structured_mesh(
            shape=shape, size=(0.2, 0.04, 0.04), origin=(0, 0, 0), dtype=dtype
        )
        x = np.asarray(mesh.reference)
        fixed = np.broadcast_to((x[:, 0] < 1e-7)[:, None], x.shape)
        tip = x[:, 0] > x[:, 0].max() - 1e-7
        free = ~fixed.reshape(-1)
        rhs = np.zeros_like(x, dtype=np.float64)
        rhs[tip, 1] = -1e-4 / tip.sum()
        for nu in [0.25, 0.35, 0.45]:
            m = fem.instantiate(mesh, fields(mesh.points), poisson=nu, fixed=jnp.asarray(fixed))
            K = numpy_hex_stiffness(mesh, 20000.0, nu)
            ref = np.zeros(x.size)
            ref[free] = sla.spsolve(K[free][:, free], rhs.reshape(-1)[free])
            diagonal = jnp.asarray(np.where(free, K.diagonal(), 1.0), dtype)
            b = jnp.asarray(rhs.reshape(-1), dtype)
            freej = jnp.asarray(free)

            def op(u: Array) -> Array:
                uu = jnp.where(freej, u, 0.0).reshape(x.shape)
                tangent = jax.jvp(
                    lambda v: fem.internal_force(cast(fem.Model, m), v, jnp.zeros(2, dtype), cfg),
                    (jnp.zeros_like(cast(fem.Model, m).mesh.reference),),
                    (uu,),
                )[1]
                return jnp.where(freej, -tangent.reshape(-1), u)

            tol = 2e-10 if is64 else 2e-5

            def raw(A: Array, b: Array) -> Array:
                return cg(A, b, tol=tol, atol=0.0, M=lambda z: z / diagonal, maxiter=2500)[0]

            def solve(b: Array) -> Array:
                return jax.lax.custom_linear_solve(op, b, solve=raw, symmetric=True)

            t = time.perf_counter()
            u = jax.block_until_ready(jax.jit(solve)(b))
            wall = time.perf_counter() - t
            residual_vector = np.asarray(op(u) - b, dtype=np.float64)
            residual = float(np.linalg.norm(residual_vector) / np.linalg.norm(np.asarray(b)))
            # Slender bending is ill-conditioned. In float32, even casting the
            # independent float64 solution back to float32 can leave a sizeable
            # RHS-relative force residual. Report that fact; gate normwise
            # backward error plus independent displacement agreement, rather
            # than pretending an arbitrarily small RHS-relative residual exists.
            kfree = K[free][:, free]
            op_norm_inf = float(np.asarray(abs(kfree).sum(axis=1)).max())
            denom = op_norm_inf * float(np.max(np.abs(np.asarray(u)))) + float(
                np.max(np.abs(np.asarray(b)))
            )
            backward_error = float(np.max(np.abs(residual_vector)) / max(denom, 1e-30))
            cast_residual = float(
                jnp.linalg.norm(op(jnp.asarray(ref, dtype)) - b) / jnp.linalg.norm(b)
            )
            residual_gate = (
                (residual < 2e-7) if is64 else (backward_error < 32 * np.finfo(dtype).eps)
            )

            error = float(np.linalg.norm(np.asarray(u) - ref) / np.linalg.norm(ref))
            deflection = -float(np.asarray(u).reshape(x.shape)[tip, 1].mean())
            eb = 1e-4 * 0.2**3 / (3 * 20000 * (0.04 * 0.04**3 / 12))
            # The constrained rest tangent is SPD for positive Lamé moduli.
            # Independently probe symmetry and a positive quadratic form.
            rng = np.random.default_rng(89)
            p = jnp.asarray(rng.normal(size=x.size), dtype)
            q = jnp.asarray(rng.normal(size=x.size), dtype)
            symmetry = float(
                jnp.abs(jnp.vdot(p, op(q)) - jnp.vdot(q, op(p)))
                / (jnp.linalg.norm(p) * jnp.linalg.norm(op(q)))
            )
            passed = bool(
                np.isfinite(u).all()
                and error < (2e-7 if is64 else 1e-4)
                and residual_gate
                and symmetry < (1e-10 if is64 else 1e-5)
                and jnp.vdot(p, op(p)) > 0
            )
            row = dict(
                shape=shape,
                unknowns=x.size,
                poisson=nu,
                relative_reference_error=error,
                true_relative_residual=residual,
                normwise_backward_error=backward_error,
                float64_reference_cast_rhs_residual=cast_residual,
                float32_gate="backward error < 32eps AND relative displacement error < 1e-4; RHS-relative residual is always reported, not hidden",
                tip_deflection_m=deflection,
                euler_bernoulli_m=eb,
                deflection_ratio_to_eb=deflection / eb,
                symmetry_probe=symmetry,
                first_compile_and_execution_s=wall,
                passed=passed,
            )
            bending.append(row)
            record("cantilever", **row)
            print(
                "cantilever",
                shape,
                nu,
                "error",
                error,
                "residual",
                residual,
                "ratio",
                deflection / eb,
                flush=True,
            )
            if not passed:
                failures.append(f"cantilever/{shape}/{nu}")
            write_json(out / "cantilever.json", bending)
        # Keep bounded host memory while compiling independent mesh capacities.
        jax.clear_caches()
    # Same continuum law/reference volume: 3D affine homogeneous MPM vs FEM.
    mesh = fem.structured_mesh(
        shape=(4, 2, 2), size=(0.1, 0.05, 0.05), origin=(0, 0, 0), dtype=dtype
    )
    cloud = mpm.uniform_cloud(cell_shape=(4, 2, 2), origin=(0, 0, 0), dtype=dtype)
    fm = fem.instantiate(mesh, fields(mesh.points))
    pm = mpm.instantiate(cloud, fields(cloud.points))
    H = jnp.asarray([[0.07, 0.1, 0.0], [-0.02, -0.03, 0.01], [0.0, 0.02, 0.04]], dtype)
    ea = fem.elastic_energy(fm, mesh.reference @ H.T, jnp.zeros(2, dtype), cfg)
    eb = mpm.elastic_energy(
        pm,
        jnp.broadcast_to(jnp.eye(3, dtype=dtype) + H, (len(cloud.reference), 3, 3)),
        jnp.zeros(2, dtype),
        cfg,
    )
    matched = dict(
        fem_energy_J=float(ea),
        mpm_energy_J=float(eb),
        relative_error=float(jnp.abs(ea - eb) / jnp.abs(ea)),
    )
    if matched["relative_error"] > (2e-6 if is64 else 3e-5):
        failures.append("homogeneous_FEM_MPM_energy")
    write_json(out / "homogeneous_comparison.json", matched)
    # Passive release: inspect kinetic+elastic loss independently from supplied
    # actuator work. MPM includes its APIC affine kinetic term h_grid^2/4 ||C||².
    dom: fem.Mesh | mpm.Cloud
    s: fem.State | mpm.State
    refinements = {}
    for kind in ["fem", "mpm"]:
        nc = NewtonConfig(atol64=3e-12, atol32=1e-6, linear_rtol32=2e-6)
        dc = DynamicsConfig(gravity=0, max_halvings=0, newton=nc)
        if kind == "fem":
            dom = fem.structured_mesh(shape=(2, 1, 1), dtype=dtype)
            m = fem.instantiate(dom, fields(dom.points))
            s = fem.initial_state(m)
            s = s._replace(u=dom.reference * jnp.asarray([0.025, -0.005, 0.01], dtype))

            def trial(s: PhysicsState, h: float) -> tuple[PhysicsState, StepInfo]:
                return fem.trial(cast(fem.Model, m), cast(fem.State, s), jnp.zeros(2, dtype), h, dc)

            def energy(s: PhysicsState) -> Array:
                return jnp.sum(fem.energy_budget(cast(fem.Model, m), cast(fem.State, s), dc))

            def vec(s: PhysicsState) -> Array:
                return jnp.concatenate((s.v.reshape(-1), 100 * cast(fem.State, s).u.reshape(-1)))
        else:
            dom = mpm.uniform_cloud(cell_shape=(2, 1), dtype=dtype)
            m = mpm.instantiate(dom, fields(dom.points))
            grid = mpm.Grid(shape=(8, 7))
            s = mpm.initial_state(m)
            F = jnp.diag(jnp.asarray([1.025, 0.995], dtype))
            s = s._replace(F=jnp.broadcast_to(F, s.F.shape), x=s.x @ F.T)

            def trial(s: PhysicsState, h: float) -> tuple[PhysicsState, StepInfo]:
                return mpm.trial(
                    cast(mpm.Model, m), cast(mpm.State, s), jnp.zeros(2, dtype), h, grid, dc
                )

            def energy(s: PhysicsState) -> Array:
                kinetic = 0.5 * jnp.sum(
                    m.mass
                    * (
                        jnp.sum(s.v**2, axis=-1)
                        + grid.spacing**2 / 4 * jnp.sum(cast(mpm.State, s).C ** 2, axis=(-2, -1))
                    )
                )
                return kinetic + mpm.elastic_energy(
                    cast(mpm.Model, m), cast(mpm.State, s).F, s.actuator, dc
                )

            def vec(s: PhysicsState) -> Array:
                return jnp.concatenate(
                    (s.v.reshape(-1), (cast(mpm.State, s).F - jnp.eye(2, dtype=dtype)).reshape(-1))
                )

        initial = float(energy(s))
        rows = []
        states = []
        for steps in [4, 8, 16, 32, 64]:
            h = 0.004 / steps

            def run(s: PhysicsState) -> tuple[PhysicsState, tuple[Array, StepInfo]]:
                def body(s: PhysicsState, i: int) -> tuple[PhysicsState, tuple[Array, StepInfo]]:
                    s, info = trial(s, h)
                    return s, (energy(s), info)

                return jax.lax.scan(body, s, jnp.arange(steps))

            last, (energies, info) = jax.block_until_ready(jax.jit(run)(s))
            e = np.asarray(energies)
            states.append(np.asarray(vec(last)))
            rise = float(np.max(np.diff(np.r_[initial, e])))
            allowance = initial * (2e-8 if is64 else 4e-4)
            ok = bool(
                last.status == 0
                and np.isfinite(e).all()
                and rise <= allowance
                and last.work_positive == 0
            )
            row = dict(
                steps=steps,
                dt=h,
                initial_energy_J=initial,
                final_energy_J=float(e[-1]),
                relative_energy_loss=float((initial - e[-1]) / initial),
                max_energy_increase_J=rise,
                energy_increase_allowance_J=allowance,
                energies_J=e.tolist(),
                passed=ok,
                max_nonlinear_residual=float(jnp.max(info.max_residual)),
                newton_iterations=float(jnp.sum(info.nonlinear_iterations)),
            )
            if not ok:
                failures.append(f"{kind}/passive_energy/{steps}")
            rows.append(row)
            print(kind, "steps", steps, "pass", ok, "loss", row["relative_energy_loss"], flush=True)
        for row, v in zip(rows, states):
            row["state_error_to_finest"] = float(np.linalg.norm(v - states[-1]))
        for i in range(1, len(rows) - 1):
            rows[i]["observed_refinement_order"] = float(
                np.log2(
                    rows[i - 1]["state_error_to_finest"]
                    / max(rows[i]["state_error_to_finest"], 1e-30)
                )
            )
        # Compare to a refined numerical trajectory, not an exact PDE solution.
        if not all(
            rows[i + 1]["state_error_to_finest"] < rows[i]["state_error_to_finest"]
            for i in range(3)
        ):
            failures.append(kind + "/temporal_refinement_nonmonotone")
        refinements[kind] = dict(
            rows=rows,
            reference="same-discretization 64-step trajectory; not continuum ground truth",
            energy_definition="lumped kinetic + elastic; MPM kinetic additionally includes APIC affine term",
        )
        write_json(out / "dynamic_refinement.json", refinements)
        jax.clear_caches()
    from envs.soft_walker.diagnostics.integration_m2 import integration_report

    failures.extend(integration_report(out, dtype))
    report: Report = dict(
        passed=not failures,
        failures=failures,
        cantilever_cases=len(bending),
        dtype=str(dtype),
        scope="linear cantilever and small passive release; no self-contact or nearly-incompressible correction implemented",
    )
    write_json(out / "summary.json", report)
    return int(bool(failures))
