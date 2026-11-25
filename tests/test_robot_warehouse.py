"""Tests for the RobotWarehouse layout co-design environment (embodiment module + build)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jumanji")

from envs.robot_warehouse import embodiment as E
from envs.robot_warehouse.config import PRESETS, make_config
from envs.robot_warehouse.generator import CoDesignWarehouseGenerator

SETUPS = list(PRESETS)


def _gen(setup):
    return CoDesignWarehouseGenerator(make_config(setup))


# --- config -----------------------------------------------------------------------

@pytest.mark.parametrize("setup", SETUPS)
def test_grid_and_budget_fit(setup):
    cfg = make_config(setup)
    gen = _gen(setup)
    # our numpy grid formula matches Jumanji's generator
    assert E.grid_size(cfg) == tuple(gen.grid_size)
    # enough eligible cells for the shelf floor, and queue <= floor
    assert gen.eligible_cells.shape[0] >= cfg.num_shelves
    assert cfg.request_queue_size <= cfg.num_shelves


def test_bad_config_rejected():
    with pytest.raises(ValueError):
        make_config("small", shelf_columns=4)      # even -> invalid
    with pytest.raises(ValueError):
        make_config("small", request_queue_size=99)  # > num_shelves floor


# --- embodiment module ------------------------------------------------------------

@pytest.mark.parametrize("setup", SETUPS)
def test_space_is_multibinary_over_eligible(setup):
    cfg = make_config(setup)
    gen = _gen(setup)
    space = E.make_embodiment_space(cfg)
    # one occupancy bit per eligible (non-goal) cell
    assert space.shape == (gen.eligible_cells.shape[0],)
    assert E.n_eligible(cfg) == gen.eligible_cells.shape[0]


@pytest.mark.parametrize("setup", SETUPS)
def test_base_is_valid_floor_occupancy(setup):
    cfg = make_config(setup)
    base = E.base_embodiment(cfg, _gen(setup))
    assert base.shape == (E.n_eligible(cfg),)
    assert int(base.sum()) == cfg.num_shelves          # base occupies exactly the floor
    assert E.validate_embodiment(base, cfg)
    assert E.make_embodiment_space(cfg).contains(base)


@pytest.mark.parametrize("setup", SETUPS)
def test_highways_reserve_goals_only(setup):
    """Aisles are emergent: the highway skeleton is neutralised to the goal cells."""
    cfg = make_config(setup)
    gen = _gen(setup)
    hw = np.asarray(gen.highways)
    reserved = set(map(tuple, np.argwhere(hw != 0)))          # (x, y) cells
    goals = set(map(tuple, E.goal_cells(cfg)))
    assert reserved == goals
    # a cell that WAS a stock Jumanji corridor (row 0) is now free floor
    assert hw[0, 0] == 0


@pytest.mark.parametrize("setup", SETUPS)
def test_eligible_is_full_grid_minus_goals(setup):
    """Placement domain = every cell except the reserved goals."""
    cfg = make_config(setup)
    gen = _gen(setup)
    h, w = E.grid_size(cfg)
    assert gen.eligible_cells.shape[0] == h * w - len(E.goal_cells(cfg))


@pytest.mark.parametrize("setup", SETUPS)
def test_validate_count_bounds(setup):
    """Structural-only: reject below the floor or above the fleet-room cap; a design
    with more shelves than the floor is valid (variable count)."""
    cfg = make_config(setup)
    n = E.n_eligible(cfg)
    K = cfg.num_shelves
    h, w = E.grid_size(cfg)
    cap = h * w - cfg.num_agents
    base = E.base_embodiment(cfg, _gen(setup))
    assert E.validate_embodiment(base, cfg)

    # more than the floor -> still valid (the budget is a minimum, not a fixed count)
    if K < n:
        more = base.copy(); more[np.flatnonzero(base == 0)[0]] = 1
        assert int(more.sum()) == K + 1
        assert E.validate_embodiment(more, cfg)

    # below the floor -> rejected
    fewer = base.copy(); fewer[np.flatnonzero(fewer)[0]] = 0
    assert int(fewer.sum()) == K - 1
    assert not E.validate_embodiment(fewer, cfg)

    # above the fleet-room cap -> rejected (only reachable when cap < vector length)
    if cap < n:
        over = np.ones(n, dtype=np.int8)
        assert not E.validate_embodiment(over, cfg)


def test_validate_structural_only():
    cfg = make_config("small")
    n = E.n_eligible(cfg)
    base = E.base_embodiment(cfg, _gen("small"))
    assert E.validate_embodiment(base, cfg)

    # wrong shape -> rejected
    assert not E.validate_embodiment(base[:-1], cfg)
    # non-binary values -> rejected (structurally not an occupancy vector)
    two = base.copy(); two[base.argmin()] = 2
    assert not E.validate_embodiment(two, cfg)
    # float dtype -> rejected
    assert not E.validate_embodiment(base.astype(np.float32), cfg)

    # a "bad" but constructible layout (all shelves crammed into the first cells)
    # is STILL valid -- quality is the optimizer's job, not the env's.
    block = np.zeros(n, dtype=np.int8); block[: cfg.num_shelves] = 1
    assert E.validate_embodiment(block, cfg)


@pytest.mark.parametrize("setup,count", [("small", 25), ("medium", 115), ("congested", 264)])
def test_reference_aisles_connect_every_shelf_to_goals(setup, count):
    from collections import deque

    cfg, gen = make_config(setup), _gen(setup)
    base = E.base_embodiment(cfg, gen)
    assert int(base.sum()) == count
    np.testing.assert_array_equal(gen._base_mask(), base)
    h, w = E.grid_size(cfg)
    occupied = np.zeros((h, w), bool)
    cells = gen.eligible_cells[base.astype(bool)]
    occupied[cells[:, 0], cells[:, 1]] = True
    goals = set(map(tuple, E.goal_cells(cfg)))
    reached, queue = set(goals), deque(goals)
    while queue:
        row, col = queue.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cell = row + dr, col + dc
            if (0 <= cell[0] < h and 0 <= cell[1] < w
                    and not occupied[cell] and cell not in reached):
                reached.add(cell)
                queue.append(cell)
    assert len(reached) == int((~occupied).sum())
    for row, col in cells:
        # A loaded shelf can leave directly into an aisle connected to a goal.
        assert any((row + dr, col + dc) in reached
                   for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))
