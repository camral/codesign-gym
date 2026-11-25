"""NumPy-only occupancy checks shared by task validation and JAX adapters.

The lookup table describes face connectivity around one grid vertex. Numerical
backends can use the same table without importing an accelerator here.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np
from numpy.typing import ArrayLike

from .contracts.fields import HostArray


def connected(occupied: HostArray) -> bool:
    """Return whether occupied cells form one face-connected component."""
    nodes = set(map(tuple, np.argwhere(occupied)))
    if not nodes:
        return True
    todo, seen = [next(iter(nodes))], set()
    while todo:
        point = todo.pop()
        if point in seen:
            continue
        seen.add(point)
        for axis in range(occupied.ndim):
            for sign in (-1, 1):
                neighbor = list(point)
                neighbor[axis] += sign
                key = tuple(neighbor)
                if key in nodes and key not in seen:
                    todo.append(key)
    return seen == nodes


@lru_cache(None)
def local_table(dim: int) -> HostArray:
    """Cache face-connectivity of every occupancy pattern in a 2^dim block."""
    count = 2**dim
    return np.asarray(
        [
            connected(np.asarray([(mask >> bit) & 1 for bit in range(count)]).reshape((2,) * dim))
            for mask in range(2**count)
        ]
    )


def pattern_slices(shape: tuple[int, ...]) -> tuple[tuple[slice, ...], ...]:
    """Return padded array slices in the bit order used by the shared lookup table."""
    return tuple(
        tuple(slice(bit, bit + extent + 1) for bit, extent in zip(bits, shape))
        for bits in product((0, 1), repeat=len(shape))
    )


def local_pattern(occupied: HostArray) -> HostArray:
    """Encode the occupied cells around each vertex, including boundary vertices."""
    padded = np.pad(occupied, [(1, 1)] * occupied.ndim)
    patterns = np.zeros(tuple(extent + 1 for extent in occupied.shape), dtype=np.int32)
    for bit, selection in enumerate(pattern_slices(occupied.shape)):
        patterns = patterns + padded[selection].astype(np.int32) * (1 << bit)
    return patterns


def errors(
    alpha: ArrayLike,
    shape: tuple[int, ...],
    *,
    junction_error: str = "NONMANIFOLD_JUNCTION",
) -> list[str]:
    """Report empty, disconnected or locally nonmanifold positive occupancy.

    ``junction_error`` retains historical caller-specific error vocabulary; it
    does not change the connectivity or local geometry rule.
    """
    occupied = np.asarray(alpha).reshape(shape) > 0
    if not np.any(occupied):
        return ["EMPTY_GEOMETRY"]
    result = []
    if not connected(occupied):
        result.append("DISCONNECTED_GEOMETRY")
    if not np.all(local_table(occupied.ndim)[local_pattern(occupied)]):
        result.append(junction_error)
    return result
