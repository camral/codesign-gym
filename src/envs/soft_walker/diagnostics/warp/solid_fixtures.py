"""Small matched JAX/native solid fixtures and independent residual references.

These fixtures support offline parity checks. Production bundle conversion lives
in engine.warp_native.solid_conversion and does not depend on diagnostic code.
"""

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from envs.soft_walker.engine.physics import beam, fem, spring
    from envs.soft_walker.engine.physics.common import DynamicsConfig
    from envs.soft_walker.engine.physics.spring import SpringConfig

import jax
import jax.numpy as jnp
import numpy as np

from envs.soft_walker.engine.warp_native.solid_conversion import from_jax
from envs.soft_walker.engine.warp_native.solid_model import SolidBundle


def fixture(
    physics: "str", precision: "str" = "float64", contact: "bool" = False
) -> "tuple[SolidBundle, spring.Model | beam.Model | fem.Model, spring.State | beam.State | fem.State, SpringConfig | DynamicsConfig, Callable[..., object], Callable[..., object]]":
    """Return a small matched solid bundle, JAX state/config and physics callables."""
    from envs.soft_walker.engine.embodiments.decoders import Fields
    from envs.soft_walker.engine.numerics.solvers import NewtonConfig
    from envs.soft_walker.engine.physics import beam, fem, spring
    from envs.soft_walker.engine.physics.common import DynamicsConfig
    from envs.soft_walker.engine.physics.contact import PlaneContact
    from envs.soft_walker.engine.physics.spring_config import SpringConfig

    t: spring.Topology | beam.Topology
    model: spring.Model | beam.Model | fem.Model
    state: spring.State | beam.State | fem.State
    config: SpringConfig | DynamicsConfig
    step: Callable[..., object]
    force: Callable[..., object]
    dtype = jnp.float64 if precision == "float64" else jnp.float32
    d = 2 if physics == "spring" else 3
    direction = np.zeros((2, d))
    direction[:, 0 if physics == "beam" else 1] = 1
    fields = Fields(
        jnp.array([1.0, 0.8], dtype),
        jnp.array([15000.0, 23000.0], dtype),
        jnp.array([950.0, 1020.0], dtype),
        jnp.array([0.8, 0.7], dtype),
        jnp.array(direction, dtype),
        jnp.array([1, 1], jnp.int32),
        jnp.array([0, 1], jnp.int32),
    )
    dt = 0.00025
    bottom = -0.0001 if contact else 0.025
    if physics == "spring":
        config = SpringConfig(
            grid_shape=(2, 1),
            size=(0.1, 0.05),
            control_dt=dt,
            substeps=1,
            settle_steps=0,
            gravity=3.0,
            ground=contact,
            initial_clearance=max(bottom, 0),
            root_tol64=2e-11,
        )
        t = spring.topology(cast("SpringConfig", config), dtype)
        t = t._replace(
            reference=t.reference
            + jnp.array([0.0, bottom - float(jnp.min(t.reference[:, 1]))], dtype)
        )
        model = spring.instantiate(fields, t, cast("SpringConfig", config), 2)
        state = spring.initial_state(cast("spring.Model", model))
        step = lambda s, c: spring.substep(
            cast("spring.Model", model), s, c, cast("SpringConfig", config)
        )[0]
        force = lambda s, a: spring.forces(
            cast("spring.Model", model), s.x, s.v, a, cast("SpringConfig", config)
        )[0]
    else:
        config = DynamicsConfig(
            gravity=3.0,
            max_halvings=0,
            contact=PlaneContact(enabled=contact),
            newton=NewtonConfig(atol64=2e-11),
        )
        if physics == "fem":
            mesh = fem.structured_mesh((2, 1, 1), (0.1, 0.05, 0.05), (0.0, bottom, 0.0), dtype)
            model = fem.instantiate(mesh, fields, channels=2)
            state = fem.initial_state(cast("fem.Model", model))
            step = lambda s, c: fem.trial(
                cast("fem.Model", model), s, c, dt, cast("DynamicsConfig", config)
            )[0]
            force = lambda s, a: fem.force(
                cast("fem.Model", model), s.u, s.v, a, cast("DynamicsConfig", config)
            )[0]
        else:
            t = beam.topology((2, 1, 1), (0.1, 0.05, 0.05), (0.0, bottom, 0.0), dtype)
            model = beam.instantiate(cast("beam.Topology", t), fields, channels=2)
            state = beam.initial_state(cast("beam.Model", model))
            step = lambda s, c: beam.trial(
                cast("beam.Model", model), s, c, dt, cast("DynamicsConfig", config)
            )[0]
            force = lambda s, a: beam.forces(
                cast("beam.Model", model),
                s.u,
                s.rotation,
                s.v,
                s.omega,
                a,
                cast("DynamicsConfig", config),
            )[:2]
    return from_jax(physics, model, state, config, dt=dt), model, state, config, step, force


def reference_functions(
    physics: "str",
    model: "spring.Model | beam.Model | fem.Model",
    state: "spring.State | beam.State | fem.State",
    config: "SpringConfig | DynamicsConfig",
    dt: "float",
) -> "tuple[Callable[[jax.Array, jax.Array], jax.Array], Callable[[jax.Array, jax.Array], dict[str, jax.Array]]]":
    """Independent JAX residual/update, without compiling its iterative solver.

    Used only in small parity tests. It evaluates the existing force/work laws
    and differentiates the converged discrete equation, not Warp's implementation.
    """
    import jax.numpy as jnp

    from envs.soft_walker.engine.physics import beam, fem, material, spring

    n = len(model.mass)
    d = 2 if physics == "spring" else 3
    qwidth = 6 if physics == "beam" else d
    mass = jnp.where(model.mass > 0, model.mass, 1.0)
    free = model.active[:, None]
    if physics == "fem":
        free = free & (~cast("fem.Model", model).fixed)

    def update(
        q: "jax.Array", command: "jax.Array"
    ) -> "tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]":
        """Map candidate generalized velocities into one step's solid state."""
        q = q.reshape(n, qwidth)
        v = jnp.where(free, q[:, :d] * config.velocity_ref, 0.0)
        a = command + (state.actuator - command) * jnp.exp(-dt / config.actuator_tau)
        if physics == "spring":
            u = cast("spring.State", state).x - cast("spring.Model", model).reference + dt * v
            R = jnp.broadcast_to(jnp.eye(d, dtype=q.dtype), (n, d, d))
            w = jnp.zeros_like(v)
        else:
            u = cast("beam.State | fem.State", state).u + dt * v
            if physics == "beam":
                w = q[:, 3:] * config.velocity_ref / cast("beam.Model", model).topology.h
                R = beam.exp_so3(dt * w) @ cast("beam.State", state).rotation
            else:
                w = jnp.zeros_like(v)
                R = jnp.broadcast_to(jnp.eye(d, dtype=q.dtype), (n, d, d))
        return u, v, R, w, a

    def residual(q: "jax.Array", command: "jax.Array") -> "jax.Array":
        """Evaluate the matched discrete residual in dimensionless velocity units."""
        u, v, R, w, a = update(q, command)
        if physics == "spring":
            f, _ = spring.forces(
                cast("spring.Model", model),
                cast("spring.Model", model).reference + u,
                v,
                a,
                cast("SpringConfig", config),
            )
        elif physics == "fem":
            f, _ = fem.force(cast("fem.Model", model), u, v, a, cast("DynamicsConfig", config))
        else:
            f, torque, _ = beam.forces(
                cast("beam.Model", model), u, R, v, w, a, cast("DynamicsConfig", config)
            )
        rv = (v - state.v - dt * f / mass[:, None]) / config.velocity_ref
        if physics == "beam":
            inertia = jnp.where(model.mass > 0, cast("beam.Model", model).inertia, 1.0)
            rw = (
                cast("beam.Model", model).topology.h
                * (w - cast("beam.State", state).omega - dt * torque / inertia[:, None])
                / config.velocity_ref
            )
            out = jnp.concatenate((rv, rw), axis=-1)
            mask = jnp.broadcast_to(model.active[:, None], out.shape)
        else:
            out = rv
            mask = free
        return jnp.where(mask, out, q.reshape(n, qwidth)).reshape(-1)

    def outputs(q: "jax.Array", command: "jax.Array") -> "dict[str, jax.Array]":
        """Return the matched solid state and signed/positive-work ledger."""
        u, v, R, w, a = update(q, command)
        if physics == "spring":
            work = spring.actuation_work(
                cast("spring.Model", model),
                cast("spring.Model", model).reference + u,
                state.actuator,
                a,
            )
            _, diss = spring.forces(
                cast("spring.Model", model),
                cast("spring.Model", model).reference + u,
                v,
                a,
                cast("SpringConfig", config),
            )
        elif physics == "fem":
            delta = fem.deformation_delta(cast("fem.Model", model), u)
            work = (
                cast("beam.Model | fem.Model", model).alpha[:, None]
                * cast("fem.Model", model).mesh.weights
                * material.work_density(
                    jnp.eye(3, dtype=q.dtype) + delta,
                    model.young[:, None],
                    cast("beam.Model | fem.Model", model).poisson[:, None],
                    fem.amounts(
                        cast("fem.Model", model), state.actuator, cast("DynamicsConfig", config)
                    )[:, None],
                    fem.amounts(cast("fem.Model", model), a, cast("DynamicsConfig", config))[
                        :, None
                    ],
                    cast("beam.Model | fem.Model", model).direction[:, None, :],
                    cast("beam.Model | fem.Model", model).kind[:, None],
                    delta_F=delta,
                )
            )
            _, diss = fem.force(cast("fem.Model", model), u, v, a, cast("DynamicsConfig", config))
        else:
            gamma, _, _, oldpre = beam.strains(
                cast("beam.Model", model), u, R, state.actuator, cast("DynamicsConfig", config)
            )
            dp = oldpre * jnp.expm1(
                config.max_log_stretch
                * (cast("beam.Model", model).strain_map @ (a - state.actuator))
            )
            dg = dp[:, None] * cast("beam.Model", model).topology.axis
            work = cast("beam.Model", model).topology.h * (
                0.5 * jnp.einsum("ei,eij,ej->e", dg, cast("beam.Model", model).extension, dg)
                - jnp.einsum("ei,eij,ej->e", gamma, cast("beam.Model", model).extension, dg)
            )
            _, _, _, diss, _ = beam.surface(
                cast("beam.Model", model), u, R, v, w, cast("DynamicsConfig", config)
            )
        ledger = jnp.stack(
            (
                state.work_signed + jnp.sum(work),
                state.work_positive + jnp.sum(jnp.maximum(work, 0.0)),
                state.dissipated + dt * diss,
                state.time + dt,
            )
        )
        return dict(u=u, v=v, rotation=R, omega=w, actuator=a, ledger=ledger)

    return residual, outputs
