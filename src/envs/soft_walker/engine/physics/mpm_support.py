"""Fixed-capacity device indexing for occupied MPM interpolation support.

Only integer grid indices are sorted. Physical interpolation weights, gradients,
and offsets retain their original differentiable expressions in the caller.
No array is sized by a device scalar and no background-grid-sized map is built.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class CompactSupport(NamedTuple):
    """Map particle incidences to a bounded ordered list of background grid nodes.

    ``ids`` and ``covered`` have shape ``(particles, stencil_width)``. Every id is
    safe to gather even on overflow; callers must mask uncovered contributions.
    ``node_ids`` has static length ``capacity`` with unused entries equal to the
    background node count. ``valid`` is false if any occupied node was omitted.
    All arrays remain on the selected JAX device; indexing is piecewise discrete.
    """

    ids: jax.Array
    covered: jax.Array
    node_ids: jax.Array
    live_count: jax.Array
    valid: jax.Array


def compact_support(
    ids: jax.Array, active: jax.Array, *, grid_nodes: int, capacity: int
) -> CompactSupport:
    """Build a bounded support map without allocating a background-grid array.

    ``ids`` contains in-bounds integer interpolation indices, including safe
    indices of particles whose physical support was separately rejected.
    ``active`` selects particles contributing positive mass. Every stencil node
    of an active particle is retained, including nodes with zero interpolation
    weight. Inactive particles may still gather a represented node shared with
    active particles, preserving their diagnostic state in the full-grid path.

    Overflow is reported on device, never silently accepted. Shape or static
    capacity errors raise before tracing the numerical calculation.
    """
    if ids.ndim != 2 or active.shape != ids.shape[:1] or ids.size == 0:
        raise ValueError("Expected nonempty (particles, stencil_width) ids and particle mask")
    if type(capacity) is not int or capacity < 1:
        raise ValueError("capacity must be a positive integer")
    if type(grid_nodes) is not int or grid_nodes < 1:
        raise ValueError("grid_nodes must be a positive integer")
    sentinel = jnp.asarray(grid_nodes, ids.dtype)
    keys = jnp.sort(jnp.where(active[:, None], ids, sentinel).reshape(-1))
    first = jnp.concatenate((jnp.ones(1, dtype=jnp.bool_), keys[1:] != keys[:-1]))
    first = first & (keys < sentinel)
    rank = jnp.cumsum(first.astype(jnp.int32)) - 1
    live_count = jnp.sum(first, dtype=jnp.int32)
    destinations = jnp.where(first, rank, capacity)
    node_ids = jnp.full((capacity,), sentinel, ids.dtype).at[destinations].set(keys, mode="drop")

    # Search all incidences, including inactive particles. Missing nodes gather
    # from a safe slot but their weights/gradients are masked by the caller.
    slots = jnp.searchsorted(node_ids, ids, side="left")
    safe_slots = jnp.minimum(slots, capacity - 1)
    covered = (slots < capacity) & (node_ids[safe_slots] == ids)
    return CompactSupport(safe_slots, covered, node_ids, live_count, live_count <= capacity)
