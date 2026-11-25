"""Cache bounded element Jacobians within an ordinary JAX linearization.

This helper never builds a global state Jacobian. A caller supplies independent
elements with flattened local geometry and a local response map. The response
derivative with respect to geometry is materialized once per linearization;
material, actuator, and reference-geometry derivatives remain exact.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

LocalParameters = tuple[jax.Array, ...]
LocalMap = Callable[[jax.Array, LocalParameters], jax.Array]


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def cached_local_map(
    evaluate: LocalMap, geometry: jax.Array, parameters: LocalParameters
) -> jax.Array:
    """Evaluate independent local responses with an explicitly reusable Jacobian.

    ``geometry`` has shape (elements, local_dofs). Every parameter leaf has the
    same leading element axis; the callable returns a flat local response. All
    arrays and cached blocks stay on the input device. The map must be pure and
    must not mix elements; global coupling belongs in the caller's assembly.
    """
    return jax.vmap(evaluate)(geometry, parameters)


@cached_local_map.defjvp
def _cached_local_jvp(
    evaluate: LocalMap,
    primals: tuple[jax.Array, LocalParameters],
    tangents: tuple[jax.Array, LocalParameters],
) -> tuple[jax.Array, jax.Array]:
    """Cache geometry blocks while retaining every continuous parameter derivative."""
    geometry, parameters = primals
    geometry_tangent, parameter_tangents = tangents
    value = jax.vmap(evaluate)(geometry, parameters)
    blocks = jax.vmap(jax.jacfwd(evaluate, argnums=0))(geometry, parameters)
    geometry_response = jnp.einsum("eoi,ei->eo", blocks, geometry_tangent)

    def parameters_only(values: LocalParameters) -> jax.Array:
        """Evaluate the same local map with geometry held fixed."""
        return jax.vmap(evaluate)(geometry, values)

    parameter_response = jax.jvp(parameters_only, (parameters,), (parameter_tangents,))[1]
    return value, geometry_response + parameter_response
