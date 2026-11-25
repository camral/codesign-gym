"""Host terrain surface queries for presentation, separate from physical contact."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from envs.soft_walker.engine.physics.terrain import Terrain
from envs.soft_walker.engine.types import HostArray
from envs.soft_walker.host_terrain import height_field


def height_at(x: ArrayLike, z: ArrayLike, terrain: Terrain) -> HostArray:
    """Upper vertical intersection, for pictures only (not contact forces).

    Uses the same rounded-box surface as the SDF. At side boundaries the visible
    union is piecewise smooth; no interpolation modifies the physical terrain.
    """
    x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
    normal = np.asarray(terrain.normal)
    boxes = (
        (np.asarray(center), np.asarray(half), float(radius), True)
        for center, half, radius, enabled in zip(
            terrain.centers, terrain.half_sizes, terrain.rounding, terrain.active
        )
        if bool(enabled)
    )
    return height_field(x, z, normal, float(terrain.offset), boxes, spatial=len(normal) == 3)
