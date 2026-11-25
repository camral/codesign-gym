"""SoftWalker per-cell embodiment: one flat vector of design properties per simulation cell.

Each cell of the body grid carries ``occupancy, young, density, strength, direction (dim), kind, channel, poisson``
(width ``7 + dim``); the vector is the C-order concatenation over cells. Occupancy is a hard switch (a cell is solid iff
``occupancy >= 0.5``) and is independent of the other fields: mass, stiffness and actuation come only from their own
parameters. ``kind`` (0 none / 1 directional / 2 isotropic) and ``channel`` must be exact integers.

Validation is structural (see the repo rule): bounds, shape, finiteness, integer fields, a fibre for every directional
actuator, and a body that is non-empty, face-connected and free of corner/edge-only junctions.
The shared host topology checks never import JAX.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, DTypeLike

    from .contracts.api_types import Fields, HostArray

import numpy as np
from gymnasium import spaces

from . import host_topology as _host_topology
from .contracts.fields import DENSITY_BOUNDS, POISSON_BOUNDS, YOUNG_BOUNDS

# Preserve historical helper imports while keeping one NumPy implementation.
_connected = _host_topology.connected
_local_pattern = _host_topology.local_pattern
_local_table = _host_topology.local_table

# name -> (width, low, high); direction's width is the dimension. Bounds are the JAX engine's design envelope.
FIELDS = ("occupancy", "young", "density", "strength", "direction", "kind", "channel", "poisson")
_BOUNDS = {
    "occupancy": (0.0, 1.0),
    "young": YOUNG_BOUNDS,
    "density": DENSITY_BOUNDS,
    "strength": (0.0, 1.0),
    "direction": (-1.0, 1.0),
    "kind": (0.0, 2.0),
    "poisson": POISSON_BOUNDS,
}
OCCUPANCY_THRESHOLD = 0.5  # Midpoint of the serialized [0,1] switch; equality is solid.


def width(dim: int) -> int:
    """Number of scalar design entries per cell in the selected dimension."""
    # Seven scalar properties plus one dim-vector; this is a serialized schema.
    return 7 + dim


def _slices(dim: int) -> dict[str, slice]:
    """Map each named cell property to its fixed flat-vector column slice."""
    out, start = {}, 0
    for name in FIELDS:
        n = dim if name == "direction" else 1
        out[name] = slice(start, start + n)
        start += n
    return out


def bounds(dim: int, cells: int, channels: int) -> tuple[HostArray, HostArray]:
    """Per-entry (low, high) of the flat vector."""
    lo, hi = np.empty(width(dim)), np.empty(width(dim))
    for name, s in _slices(dim).items():
        lo[s], hi[s] = (0.0, channels - 1.0) if name == "channel" else _BOUNDS[name]
    return np.tile(lo, cells), np.tile(hi, cells)


class CellEmbodimentSpace(spaces.Box):
    """Flat Box with a compact description of the repeated per-cell layout."""

    def __init__(self, grid: tuple[int, ...], channels: int, dtype: DTypeLike) -> None:
        self.grid = tuple(grid)
        low, high = bounds(len(grid), int(np.prod(grid)), channels)
        super().__init__(low.astype(dtype), high.astype(dtype), dtype=dtype)

    def __repr__(self) -> str:
        fields = []
        for name, indices in _slices(len(self.grid)).items():
            label = f"{name}[{len(self.grid)}]" if name == "direction" else name
            fields.append(f"{label}=[{self.low[indices.start]:g}, {self.high[indices.start]:g}]")
        return (f"CellEmbodimentSpace(grid={self.grid}, shape={self.shape}, dtype={self.dtype}, "
                f"per_cell={{" + ", ".join(fields) + "})")


def to_cells(vector: ArrayLike, dim: int) -> Fields:
    """Flat vector -> per-cell property arrays (no validation)."""
    table = np.asarray(vector).reshape(-1, width(dim))
    return {
        name: (table[:, s] if name == "direction" else table[:, s.start])
        for name, s in _slices(dim).items()
    }


def from_cells(props: Fields, dim: int, dtype: DTypeLike) -> HostArray:
    """Per-cell property arrays -> flat vector."""
    count = len(props["occupancy"])
    table = np.empty((count, width(dim)), dtype)
    for name, s in _slices(dim).items():
        table[:, s] = np.asarray(props[name]).reshape(count, -1)
    return table.ravel()


def topology_errors(solid: HostArray) -> list[str]:
    """Report empty, disconnected or corner/edge-only junctions in a Boolean grid."""
    return _host_topology.errors(solid, solid.shape)


# --- validation -------------------------------------------------------------------------------------------------------


def structural_errors(vector: ArrayLike, grid: tuple[int, ...], channels: int) -> list[str]:
    """All structural errors of a flat per-cell vector (empty list = valid design)."""
    dim, cells = len(grid), int(np.prod(grid))
    try:
        v = np.asarray(vector, dtype=float)
    except (TypeError, ValueError):
        return ["TYPE"]
    if v.shape != (cells * width(dim),):
        return [f"SHAPE: expected ({cells * width(dim)},), got {v.shape}"]
    if not np.all(np.isfinite(v)):
        return ["NONFINITE"]
    lo, hi = bounds(dim, cells, channels)
    if np.any(v < lo) or np.any(v > hi):
        return ["BOUNDS"]
    props = to_cells(v, dim)
    errors = []
    if np.any(props["kind"] != np.round(props["kind"])) or np.any(
        props["channel"] != np.round(props["channel"])
    ):
        errors.append("INTEGER_REQUIRED: kind/channel")
    # Solid cells only. Whether an empty cell may have a fibreless directional actuator is the engine's call, which
    # validate_embodiment gets by building the body (the solid builders refuse it, the MPM ones do not).
    solid = props["occupancy"] >= OCCUPANCY_THRESHOLD
    if np.any(solid & (np.round(props["kind"]) == 1) & (np.linalg.norm(props["direction"], axis=-1) == 0)):
        errors.append("DIRECTIONAL_ACTUATOR_WITHOUT_FIBER")
    errors.extend(topology_errors((props["occupancy"] >= OCCUPANCY_THRESHOLD).reshape(grid)))
    return errors
