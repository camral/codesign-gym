"""Dimension-independent face connectivity and local nonmanifold-junction checks."""

from __future__ import annotations

from types import ModuleType

import jax
import jax.numpy as jnp
import numpy as np

from .. import host_topology as _host_topology
from ..host_topology import (
    errors as _host_errors,
    local_pattern as _host_local_pattern,
    local_table as local_table,
    pattern_slices,
)
from .types import Array, Shape

_connected = _host_topology.connected  # Historical host helper import.


def local_pattern(occupied: Array, xp: ModuleType) -> Array:
    """Encode local occupancy bit patterns with the selected NumPy/JAX operations."""
    if xp is np:
        return _host_local_pattern(np.asarray(occupied))
    pad = xp.pad(occupied, [(1, 1)] * occupied.ndim)
    ix = xp.zeros(tuple(n + 1 for n in occupied.shape), dtype=xp.int32)
    for bit, selection in enumerate(pattern_slices(occupied.shape)):
        ix = ix + pad[selection].astype(xp.int32) * (1 << bit)
    return ix


def errors(alpha: Array, shape: Shape) -> list[str]:
    """Report the shared host topology rules without changing the engine API."""
    return _host_errors(alpha, shape)


def valid(alpha: jax.Array, shape: Shape) -> jax.Array:
    """Evaluate the same topology constraints inside compiled JAX validation."""
    a = alpha.reshape(shape) > 0
    initial = jax.nn.one_hot(jnp.argmax(a.reshape(-1)), a.size, dtype=jnp.int32).reshape(shape) > 0

    def fill(_: jax.Array, v: jax.Array) -> jax.Array:
        """Propagate face-connected occupancy by one lattice cell."""
        neighbors = v
        for axis in range(a.ndim):
            pads = [(0, 0)] * a.ndim
            pads[axis] = (1, 1)
            p = jnp.pad(v, pads)
            lo = [slice(None)] * a.ndim
            hi = list(lo)
            lo[axis] = slice(0, -2)
            hi[axis] = slice(2, None)
            neighbors = neighbors | p[tuple(lo)] | p[tuple(hi)]
        return neighbors & a

    def step(
        carry: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Stop connectivity propagation once its reached set no longer changes."""
        count, reached, _ = carry
        new = fill(count, reached)
        return count + 1, new, jnp.any(new != reached)

    # Boolean validation is nondifferentiable. Stop once propagation converges
    # instead of paying N iterations for every dense 3D candidate (O(N^2)).
    _, reached, _ = jax.lax.while_loop(
        lambda c: (c[0] < a.size) & c[2],
        step,
        (jnp.asarray(0, jnp.int32), initial & a, jnp.asarray(True)),
    )
    return (
        jnp.any(a)
        & jnp.all(reached == a)
        & jnp.all(jnp.asarray(local_table(a.ndim))[local_pattern(a, jnp)])
    )
