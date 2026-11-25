"""Optional edge force Jacobians for JAX spring linearizations."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .local_tangent import LocalParameters, cached_local_map
from .physics_constants import SPRING_SQUARED_LENGTH_FLOOR


def local_force(geometry: jax.Array, parameters: LocalParameters) -> jax.Array:
    """Return one endpoint force in newtons for an elastic, damped edge.

    Geometry contains displacement then relative velocity, each of length two.
    Parameters are stiffness (N/m), damping (N s/m), and preferred length (m).
    The collapse guard and force law match the ordinary spring implementation.
    """
    delta, relative_velocity = geometry[:2], geometry[2:]
    stiffness, damping, preferred = parameters
    length = jnp.sqrt(
        jnp.sum(delta * delta) + jnp.asarray(SPRING_SQUARED_LENGTH_FLOOR, geometry.dtype)
    )
    normal = delta / length
    speed = jnp.sum(relative_velocity * normal)
    return (stiffness * (length - preferred) + damping * speed) * normal


def edge_forces(
    displacement: jax.Array,
    velocity: jax.Array,
    stiffness: jax.Array,
    damping: jax.Array,
    preferred: jax.Array,
) -> jax.Array:
    """Evaluate edge loads with cached 2-by-4 position/velocity derivative blocks.

    Vector arrays have shape (edges, 2), scalar arrays (edges,). Every continuous
    input remains differentiable. The caller handles inactive nodes, assembly,
    contact, and dissipation; this helper only changes repeated linear work.
    """
    geometry = jnp.concatenate((displacement, velocity), axis=1)
    return cached_local_map(local_force, geometry, (stiffness, damping, preferred))
