"""Embodiment design layer for the RobotWarehouse co-design environment.

The embodiment is **shelf occupancy**: a ``MultiBinary(N_eligible)`` vector with one
bit per eligible (non-goal) grid cell -- bit ``i`` set means "place a shelf on
eligible cell ``i``". This is the collision-free, decode-free representation of a
combinatorial layout: a cell is either occupied or not, so two shelves can never
share a cell (the failure mode of a coordinate encoding), and the optimizer's
vector *is* the design (no top-K / factoradic decode in between).

The number of shelves is variable. The preset's ``num_shelves`` is the **floor**
(minimum budget = design pressure): a design is rejected only if it places fewer.
The only upper bound is structural -- at least ``num_agents`` cells must stay free
so the robots have somewhere to stand -- i.e. ``popcount <= H*W - num_agents`` (a
very generous cap; it only excludes near-complete fills). The grid, fleet, and goals
are fixed per preset; only shelf occupancy is designed. Aisles are *not* fixed:
Jumanji's highway corridors are neutralised (see ``generator.py``), so corridors
emerge from where shelves are not, and the goal cells are the only reserved floor.

Validation is **structural only** -- a design is rejected only if the warehouse
cannot be constructed from it (wrong shape/dtype, non-binary values) or if it falls
below the shelf-count floor. A "bad" but constructible layout (blocked corridors,
shelves far from goals) is a valid embodiment that simply earns low reward;
discovering that is the optimizer's job, not the environment's.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np
from gymnasium import spaces

from .config import RobotWarehouseConfig

EmbType = np.ndarray  # (n_eligible,) binary occupancy vector (one bit per eligible cell)


def grid_size(cfg: RobotWarehouseConfig) -> Tuple[int, int]:
    """(H, W) warehouse grid, matching Jumanji's derivation from the shelf params."""
    h = (cfg.column_height + 1) * cfg.shelf_rows + 2
    w = 3 * cfg.shelf_columns + 1
    return int(h), int(w)


def goal_cells(cfg: RobotWarehouseConfig) -> np.ndarray:
    """``(2, 2)`` array of the reserved goal cells as ``(x, y)`` grid coordinates.

    Mirrors Jumanji's goal placement (two cells at the middle of the bottom row);
    Jumanji stores them as ``(y, x)`` so we return ``(x, y) = (H-1, W//2-1)`` and
    ``(H-1, W//2)`` to match the grid's coordinate convention.
    """
    h, w = grid_size(cfg)
    return np.array([[h - 1, w // 2 - 1], [h - 1, w // 2]], dtype=np.int32)


def n_eligible(cfg: RobotWarehouseConfig) -> int:
    """Number of eligible (non-goal) cells = the length of the occupancy vector."""
    h, w = grid_size(cfg)
    return int(h * w - len(goal_cells(cfg)))


def make_embodiment_space(cfg: RobotWarehouseConfig) -> spaces.MultiBinary:
    """The ``embodiment_space``: one occupancy bit per eligible cell."""
    return spaces.MultiBinary(n_eligible(cfg))


def base_embodiment(cfg: RobotWarehouseConfig, generator) -> EmbType:
    """RWARE-inspired paired racks with connected aisles and the preset shelf count.

    Keep a vertical aisle every third column, the delivery row, and two queue
    cells above the goals clear. Prefer RWARE's horizontal aisles and larger
    queue area where the preset budget allows them. Dense presets may fill
    these extra spaces, but every shelf still borders a connected vertical aisle.
    Custom designs are not constrained to this reference pattern.
    """
    h, w = grid_size(cfg)
    row, col = np.asarray(generator.eligible_cells).T
    queue_columns = (col == w // 2 - 1) | (col == w // 2)
    dense = (col % 3 != 0) & (row < h - 1) & ~((row == h - 2) & queue_columns)
    reference = dense & (row % (cfg.column_height + 1) != 0)
    reference &= ~((row > h - (cfg.column_height + 3)) & queue_columns)
    target = cfg.num_shelves
    if target > int(dense.sum()):
        raise ValueError("shelf minimum exceeds the capacity of the reference aisle pattern")
    occupied = np.flatnonzero(reference)
    if len(occupied) > target:
        # Spread omissions across the rack blocks instead of emptying one end.
        keep = occupied[np.linspace(0, len(occupied) - 1, target, dtype=int)]
        reference[:] = False
        reference[keep] = True
    elif len(occupied) < target:
        extra = np.flatnonzero(dense & ~reference)
        reference[extra[:target - len(occupied)]] = True
    return reference.astype(np.int8)


def validate_embodiment(embodiment: Mapping[str, Any], cfg: RobotWarehouseConfig) -> bool:
    """Return True iff ``embodiment`` yields a constructible warehouse.

    Structural checks only: correct shape, integer, binary ``{0, 1}`` values, at
    least ``num_shelves`` occupied cells (the design-pressure floor), and at most
    ``H*W - num_agents`` (leave room for the robots). Placement *quality* -- and any
    count within those bounds -- is deliberately not checked.
    """
    emb = np.asarray(embodiment)
    if emb.shape != (n_eligible(cfg),) or not np.issubdtype(emb.dtype, np.integer):
        return False
    if np.any((emb != 0) & (emb != 1)):
        return False
    count = int(emb.sum())
    h, w = grid_size(cfg)
    if count < cfg.num_shelves:  # shelf-count floor (minimum budget)
        return False
    if count > h * w - cfg.num_agents:  # leave room for the fleet (constructibility cap)
        return False
    return True
