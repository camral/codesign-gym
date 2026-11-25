"""Host validation and plane/box upload shared by native physical schemas."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import warp as wp

from .config import MPMConfig
from .host_types import FieldArrays


def validate_terrain(terrain: FieldArrays, dim: int, *, normal_first: bool) -> None:
    """Validate fixed plane/box storage, preserving the caller's error order.

    MPM historically checks its plane normal first; solids first check box
    centers and all array shapes. Keep those error boundaries for malformed
    input while sharing the accepted schema and physical bounds.
    """
    required = {"normal", "offset", "centers", "half_sizes", "rounding", "active"}
    if set(terrain) != required:
        raise ValueError("Invalid resolved terrain schema")
    values = {key: np.asarray(value) for key, value in terrain.items()}
    if normal_first:
        if values["normal"].shape != (dim,) or not np.isclose(
            np.linalg.norm(values["normal"]), 1.0, rtol=1e-6
        ):
            raise ValueError("Terrain plane normal must have unit length")
    elif values["centers"].ndim != 2 or values["centers"].shape[1] != dim:
        raise ValueError("Invalid terrain box centers")
    count = len(values["centers"])
    shapes = {
        "normal": (dim,),
        "offset": (),
        "centers": (count, dim),
        "half_sizes": (count, dim),
        "rounding": (count,),
        "active": (count,),
    }
    if any(values[key].shape != shape for key, shape in shapes.items()):
        raise ValueError("Terrain array shapes disagree")
    if any(not np.all(np.isfinite(value)) for value in values.values()):
        raise ValueError("Terrain values must be finite")
    if not normal_first and not np.isclose(np.linalg.norm(values["normal"]), 1.0, rtol=1e-6):
        raise ValueError("Terrain plane normal must have unit length")
    if np.any(values["half_sizes"] <= 0) or np.any(values["rounding"] < 0):
        raise ValueError("Invalid rounded-box dimensions")
    if np.any((values["active"] != 0) & (values["active"] != 1)):
        raise ValueError("Terrain active flags must be boolean")


def populate_terrain(
    target: SimpleNamespace,
    terrain: FieldArrays | None,
    config: MPMConfig,
    kernels: SimpleNamespace,
    device: str | wp.Device,
) -> None:
    """Upload independent plane/box storage into an existing concrete struct."""
    dim = config.dim
    target.plane_normal = kernels.vec(
        *(np.eye(dim)[config.vertical_axis] if terrain is None else terrain["normal"])
    )
    target.plane_offset = config.contact_height if terrain is None else float(terrain["offset"])
    for name, key, dtype, empty in (
        ("box_centers", "centers", kernels.vec, np.empty((0, dim))),
        ("box_half", "half_sizes", kernels.vec, np.empty((0, dim))),
        ("box_rounding", "rounding", kernels.real, np.empty(0)),
        ("box_active", "active", int, np.empty(0, np.int32)),
    ):
        value = empty if terrain is None else terrain[key]
        setattr(target, name, wp.array(np.asarray(value), dtype=dtype, device=device))
