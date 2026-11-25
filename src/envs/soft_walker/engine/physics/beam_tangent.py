"""Optional local beam-load Jacobians for repeated matrix-free products.

Each beam keeps a bounded 12-by-24 derivative of endpoint forces/torques with
respect to endpoint translations and rotation matrices. The outer integrator
still differentiates its exact SO(3) update, and assembly still couples the
full lattice. Material and controller derivatives are not frozen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from .local_tangent import LocalParameters, cached_local_map

if TYPE_CHECKING:
    from .beam import Model
    from .common import DynamicsConfig


def local_load(geometry: jax.Array, parameters: LocalParameters) -> jax.Array:
    """Return flat endpoint force/torque pairs from the unchanged link energy.

    Geometry contains two translations followed by two 3-by-3 rotations. Parameters
    are the extension and bending stiffness tensors, reference axis, link length
    in metres, and dimensionless preferred log stretch. Rotational variations
    are spatial increments, so torques use the same world frame as ``beam``.
    """
    from .beam import exp_so3, log_so3

    extension, bend, axis, length, log_preferred = parameters
    displacement = geometry[:6].reshape(2, 3)
    rotations = geometry[6:].reshape(2, 3, 3)

    def energy(increment: jax.Array) -> jax.Array:
        """Apply independent endpoint translation and spatial rotation variations."""
        increments = increment.reshape(2, 6)
        u = displacement + increments[:, :3]
        R = exp_so3(increments[:, 3:]) @ rotations
        relative = jnp.swapaxes(R[0], -1, -2) @ R[1]
        rv = log_so3(relative)
        mid = R[0] @ exp_so3(0.5 * rv)
        local_delta = (mid - jnp.eye(3, dtype=u.dtype)).T @ axis
        local_delta = local_delta + mid.T @ (u[1] - u[0]) / length
        gamma = local_delta - jnp.expm1(log_preferred) * axis
        kappa = rv / length
        return (
            0.5
            * length
            * (
                jnp.einsum("i,ij,j->", gamma, extension, gamma)
                + jnp.einsum("i,ij,j->", kappa, bend, kappa)
            )
        )

    return -jax.grad(energy)(jnp.zeros(12, geometry.dtype))


def elastic_force_torque(
    model: Model, u: jax.Array, R: jax.Array, actuator: jax.Array, cfg: DynamicsConfig
) -> tuple[jax.Array, jax.Array]:
    """Assemble cached local beam loads without changing the physical residual.

    Inputs and results are device arrays with node-major shape (nodes, 3), except
    rotations (nodes, 3, 3) and channel actuator values. Inactive-node masking,
    constitutive coefficients, active geometry, and force signs match the
    uncached beam implementation. The cache belongs to each JAX linearization.
    """
    edges = model.topology.edges
    u = jnp.where(model.active[:, None], u, 0.0)
    R = jnp.where(model.active[:, None, None], R, jnp.eye(3, dtype=u.dtype))
    geometry = jnp.concatenate(
        (u[edges].reshape(len(edges), 6), R[edges].reshape(len(edges), 18)), axis=1
    )
    parameters = (
        model.extension,
        model.bend,
        model.topology.axis,
        jnp.broadcast_to(model.topology.h, (len(edges),)),
        cfg.max_log_stretch * (model.strain_map @ actuator),
    )
    local = cached_local_map(local_load, geometry, parameters).reshape(-1, 6)
    loads = jnp.zeros((u.shape[0], 6), u.dtype).at[edges.reshape(-1)].add(local)
    loads = jnp.where(model.active[:, None], loads, 0.0)
    return loads[:, :3], loads[:, 3:]
