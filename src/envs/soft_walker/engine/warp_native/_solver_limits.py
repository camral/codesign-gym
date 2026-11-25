"""Host-only capacity checks for native solver implementations."""

from __future__ import annotations

import math

ADAPTIVE_GMRES_MAX_RESTART = 128


def validate_adaptive_restart(restart: int, scalar_dofs: int) -> None:
    """Reject only an effective restart unsupported by dynamic adaptive GMRES.

    Callers retain the existing DOF clamp, including requested restarts greater than
    the implementation limit on small systems. Other GMRES implementations do not
    share this constraint. Configuration already validates positive integer inputs.
    """
    if min(restart, scalar_dofs) > ADAPTIVE_GMRES_MAX_RESTART:
        raise ValueError(
            f"adaptive_dynamic GMRES effective restart exceeds {ADAPTIVE_GMRES_MAX_RESTART}"
        )


def mpm_support_capacity(
    particle_count: int, grid_shape: tuple[int, ...], compact_nodes: int
) -> tuple[int, int]:
    """Return incidence and support capacities before allocating grid buffers.

    The quadratic tensor-product stencil has three nodes per spatial axis. This
    matches the native kernel's stencil and preserves its original capacity rules.
    """
    incidence_count = particle_count * 3 ** len(grid_shape)
    if incidence_count >= 2**31 - 1:
        raise ValueError("Particle-support incidence count exceeds int32 addressing")
    automatic_capacity = min(math.prod(grid_shape), incidence_count)
    capacity = min(automatic_capacity, compact_nodes or automatic_capacity)
    return incidence_count, capacity
