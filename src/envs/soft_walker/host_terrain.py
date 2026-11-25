"""NumPy display heights for a plane and rounded-box union.

Callers retain their input/default policies. This module only evaluates geometry;
it neither broadcasts inputs explicitly nor validates a plane as a height field.
It can also be imported by the standalone recording viewer.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

HostArray: TypeAlias = NDArray[Any]
Coordinates: TypeAlias = Sequence[float] | HostArray
RoundedBox: TypeAlias = tuple[Coordinates, Coordinates, float, bool]


def height_field(
    x: HostArray,
    z: HostArray,
    normal: HostArray,
    offset: float,
    boxes: Iterable[RoundedBox],
    *,
    spatial: bool | None = None,
) -> HostArray:
    """Evaluate the upper vertical surface in metres at horizontal x/z samples.

    A missing ``spatial`` uses each primitive's dimension; scenario adapters may
    instead use the terrain dimension. Scalar coercion stays with each adapter
    so saved Python geometry and device geometry keep their previous precision.
    """
    plane_spatial = len(normal) > 2 if spatial is None else spatial
    lateral = normal[2] * z if plane_spatial else 0.0
    height = (offset - normal[0] * x - lateral) / normal[1]
    for center, half, radius, enabled in boxes:
        if not enabled:
            continue
        dx = np.maximum(np.abs(x - center[0]) - (half[0] - radius), 0.0)
        box_spatial = len(center) == 3 if spatial is None else spatial
        dz = np.maximum(np.abs(z - center[2]) - (half[2] - radius), 0.0) if box_spatial else 0.0
        radial2 = dx * dx + dz * dz
        top = center[1] + half[1] - radius + np.sqrt(np.maximum(radius * radius - radial2, 0.0))
        height = np.maximum(height, np.where(radial2 <= radius * radius + 1e-20, top, -np.inf))
    return height
