"""Preset definitions for the RobotWarehouse co-design environment.

Each preset fixes a warehouse footprint (grid), a fleet of robots, and a shelf
*budget* -- the number of shelves to place. The embodiment is *where* those
shelves go on the fixed grid (see ``embodiment.py``); the budget is the design
pressure, so QoE (deliveries) is left untouched.

The grid is not set directly: Jumanji derives it from structural params as
``((column_height + 1) * shelf_rows + 2, 3 * shelf_columns + 1)`` (``shelf_columns``
must be odd). Each preset picks those params to hit a target footprint; we then
ignore Jumanji's structured shelf layout and place ``num_shelves`` shelves at the
embodiment's positions instead.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict


@dataclass(frozen=True)
class RobotWarehouseConfig:
    """Static configuration for one RobotWarehouse co-design preset."""

    name: str
    # structural params (choose to hit the target grid; shelf_columns must be odd)
    shelf_rows: int
    shelf_columns: int
    column_height: int
    # fleet + perception (fixed, not designed)
    num_agents: int
    sensor_range: int
    # design budget: shelf occupancy is a MultiBinary vector over eligible cells;
    # ``num_shelves`` is the *floor* (minimum placed shelves = design pressure), not a
    # fixed count -- the optimizer may place more (up to H*W - num_agents).
    num_shelves: int          # minimum shelves the designer must place (floor budget)
    request_queue_size: int   # simultaneously requested shelves (<= num_shelves floor)
    # episode
    time_limit: int


# Fill = num_shelves / (H*W) against the FULL grid, since aisles are emergent
# (only the goal cells are reserved -- see generator.py). Difficulty ladders on
# both footprint and fill: denser floors make navigability the binding constraint.
PRESETS: Dict[str, RobotWarehouseConfig] = {
    # 10x10 floor (100 cells), small fleet -- ~25% fill: obstacles start to matter.
    "small": RobotWarehouseConfig(
        name="small",
        shelf_rows=1, shelf_columns=3, column_height=7,
        num_agents=2, sensor_range=1,
        num_shelves=25, request_queue_size=12, time_limit=500,
    ),
    # 16x16 floor (256 cells), default fleet -- ~45% fill: congestion + navigability.
    "medium": RobotWarehouseConfig(
        name="medium",
        shelf_rows=2, shelf_columns=5, column_height=6,
        num_agents=4, sensor_range=1,
        num_shelves=115, request_queue_size=24, time_limit=500,
    ),
    # 20x22 floor (440 cells), large fleet -- ~60% fill: dense, navigability dominates.
    "congested": RobotWarehouseConfig(
        name="congested",
        shelf_rows=3, shelf_columns=7, column_height=5,
        num_agents=8, sensor_range=1,
        num_shelves=264, request_queue_size=32, time_limit=500,
    ),
}


def make_config(setup: str = "small", **overrides: Any) -> RobotWarehouseConfig:
    """Return the :class:`RobotWarehouseConfig` for ``setup``, with optional field overrides."""
    if setup not in PRESETS:
        raise ValueError(f"unknown robot_warehouse setup {setup!r}; choose from {sorted(PRESETS)}")
    cfg = PRESETS[setup]
    if overrides:
        cfg = replace(cfg, **overrides)
    if cfg.shelf_columns % 2 != 1:
        raise ValueError("shelf_columns must be odd (Jumanji constraint)")
    if cfg.request_queue_size > cfg.num_shelves:
        raise ValueError("request_queue_size must be <= num_shelves")
    return cfg
