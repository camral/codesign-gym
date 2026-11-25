"""Optional local constitutive blocks for implicit MPM Krylov products.

Blocks contain dP/dF for one particle (at most 3x3x3x3), never a global grid
Jacobian. A custom JVP exposes their multiplication to JAX linearization, so
the blocks are computed once at each linearization point and reused by its
forward and transpose products. Material and actuator derivatives remain exact.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from . import material

if TYPE_CHECKING:
    from .common import DynamicsConfig
    from .mpm import Model, State, Stencil

StressArguments = tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


def particle_stress(
    delta: jax.Array,
    young: jax.Array,
    poisson: jax.Array,
    amount: jax.Array,
    direction: jax.Array,
    kind: jax.Array,
    active_stress: jax.Array,
    law: str,
) -> jax.Array:
    """Evaluate the unchanged first Piola stress, with dimensions ending in (dim, dim).

    ``delta`` is F-I, ``young`` and ``active_stress`` are in Pa, and ``amount``
    is dimensionless active log strain. The selected law decides which actuator
    field is physically used. Leading dimensions broadcast as in material.py.
    """
    if law == "active_stress":
        passive = material.stress_from_delta(
            delta, young, poisson, jnp.zeros_like(amount), direction, kind
        )
        return passive + material.active_stress_from_delta(delta, active_stress, direction, kind)
    if law != "active_strain":
        raise ValueError("Unknown MPM actuation law")
    return material.stress_from_delta(delta, young, poisson, amount, direction, kind)


def particle_tangent(
    delta: jax.Array,
    young: jax.Array,
    poisson: jax.Array,
    amount: jax.Array,
    direction: jax.Array,
    kind: jax.Array,
    active_stress: jax.Array,
    law: str,
) -> jax.Array:
    """Return exact per-particle dP_ij/dF_ab blocks with shape (N, dim, dim, dim, dim).

    Inputs have one leading particle axis; scalar material arrays and directions
    must already match it. Integer actuator labels remain fixed discrete inputs.
    All differentiation and allocation stay on the selected JAX device.
    """
    derivative = jax.jacfwd(partial(particle_stress, law=law), argnums=0)
    return jax.vmap(derivative)(delta, young, poisson, amount, direction, kind, active_stress)


@partial(jax.custom_jvp, nondiff_argnums=(7,))
def cached_particle_stress(
    delta: jax.Array,
    young: jax.Array,
    poisson: jax.Array,
    amount: jax.Array,
    direction: jax.Array,
    kind: jax.Array,
    active_stress: jax.Array,
    law: str,
) -> jax.Array:
    """Evaluate ordinary stress while exposing a reusable local deformation Jacobian.

    The custom derivative changes only linearization work. Stress values and
    derivatives with respect to all continuous material/actuator inputs follow
    ``particle_stress``. Transposition uses the actual local tensor transpose.
    """
    return particle_stress(delta, young, poisson, amount, direction, kind, active_stress, law)


@cached_particle_stress.defjvp
def _cached_stress_jvp(
    law: str, primals: StressArguments, tangents: StressArguments
) -> tuple[jax.Array, jax.Array]:
    """Separate deformation blocks from the remaining exact parameter derivatives."""
    value = particle_stress(*primals, law)
    blocks = particle_tangent(*primals, law)
    deformation_derivative = jnp.einsum("pijab,pab->pij", blocks, tangents[0])

    def parameters_only(
        young: jax.Array,
        poisson: jax.Array,
        amount: jax.Array,
        direction: jax.Array,
        kind: jax.Array,
        active_stress: jax.Array,
    ) -> jax.Array:
        """Hold deformation fixed while preserving every other continuous derivative."""
        return particle_stress(
            primals[0], young, poisson, amount, direction, kind, active_stress, law
        )

    parameter_derivative = jax.jvp(parameters_only, primals[1:], tangents[1:])[1]
    return value, deformation_derivative + parameter_derivative


def grid_tangent_blocks(
    model: Model,
    state: State,
    actuator: jax.Array,
    velocity: jax.Array,
    grid_mass: jax.Array,
    st: Stencil,
    dt: jax.Array,
    cfg: DynamicsConfig,
) -> jax.Array:
    """Assemble exact node-diagonal momentum Jacobians at one fixed velocity.

    Returned blocks have shape (grid_buffer_nodes, dim, dim), matching either
    full or compact stencil indices. Elastic and smooth-contact derivatives use
    the actual discrete update and retain nonsymmetric friction terms. This is
    only a preconditioner: couplings between different nodes are still evaluated
    by the complete matrix-free residual. Inactive grid rows become identity.
    """
    from . import contact
    from .mpm import amounts, stress_amounts, surface_mapping

    dim = velocity.shape[-1]
    local_velocity = velocity[st.ids]
    gradient = jnp.einsum("pki,pkj->pij", local_velocity, st.gradient)
    delta = state.F - jnp.eye(dim, dtype=state.F.dtype) + dt * (gradient @ state.F)
    delta = jnp.where(model.active[:, None, None], delta, 0.0)
    tangent = particle_tangent(
        delta,
        model.young,
        model.poisson,
        amounts(model, actuator, cfg),
        model.direction,
        model.kind,
        stress_amounts(model, actuator, cfg),
        cfg.actuation_model,
    )
    reference_gradient = jnp.einsum("pji,pkj->pki", state.F, st.gradient)
    elastic = jnp.einsum("pijab,pkj,pkb->pkia", tangent, reference_gradient, reference_gradient)
    elastic = (dt**2 * model.alpha * model.cloud.volume)[:, None, None, None] * elastic
    blocks = jnp.zeros((velocity.shape[0], dim, dim), velocity.dtype)
    blocks = blocks.at[st.ids.reshape(-1)].add(elastic.reshape(-1, dim, dim))
    if cfg.contact.enabled:
        q0, weights = surface_mapping(model, state, st)
        face_velocity = jnp.einsum("pfk,pki->pfi", weights, local_velocity)

        def face_force(v: jax.Array, q: jax.Array, area: jax.Array) -> jax.Array:
            """Differentiate the actual predicted marker position and slip velocity together."""
            return contact.response(q + dt * v, v, area, cfg.contact, model.terrain)[0]

        derivative = jax.vmap(jax.jacfwd(face_force, argnums=0))(
            face_velocity.reshape(-1, dim), q0.reshape(-1, dim), model.contact_area.reshape(-1)
        ).reshape(q0.shape[:-1] + (dim, dim))
        local_contact = -dt * jnp.einsum("pfk,pfij->pkij", weights**2, derivative)
        blocks = blocks.at[st.ids.reshape(-1)].add(local_contact.reshape(-1, dim, dim))
    active = grid_mass > 0
    mass = jnp.where(active, grid_mass, 1.0)
    eye = jnp.eye(dim, dtype=velocity.dtype)
    blocks = eye + blocks / mass[:, None, None]
    return jnp.where(active[:, None, None], blocks, eye)


def inverse_node_blocks(blocks: jax.Array) -> jax.Array:
    """Invert independent 2x2/3x3 blocks without a global matrix or library LU call.

    Singular/nonfinite blocks remain nonfinite and are rejected by the linear
    solver; no diagonal substitution or tolerance adjustment is made here.
    Preconditioner construction is deliberately excluded from physical gradients.
    """
    return jax.lax.stop_gradient(jnp.swapaxes(material.inverse_transpose(blocks), -1, -2))
