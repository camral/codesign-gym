"""Custom Jumanji generator + env subclass for occupancy-based shelf co-design.

Jumanji's stock ``RandomGenerator`` fills *every* non-highway cell with a shelf and
couples the grid size to the shelf params. For co-design we instead keep the grid
fixed (from the preset's structural params) and let the embodiment be a **binary
occupancy vector** over the eligible (non-goal) cells: bit ``i`` places a shelf on
``eligible_cells[i]``. Shelf slot ``i`` is bound one-to-one to eligible cell ``i``,
so the Jumanji shelf array is sized to ``max_shelves = len(eligible_cells)`` and a
slot is *active* iff its bit is set. Inactive slots are simply never written onto
the grid, so they are invisible to agents, observations, collisions, and delivery.

We also **neutralise Jumanji's fixed highway skeleton**. Stock RobotWarehouse
reserves a corridor network; for layout co-design the aisles should *emerge* from
where shelves are not, so the only reserved cells are the **goals** (kept clear as
delivery points). ``highways`` is rewritten to mark the goal cells only.

Because shelf slots are padded to ``max_shelves`` but only some are active, Jumanji's
request-queue replenishment (which samples from the whole shelf pool) would hand out
inactive, off-grid shelves that can never be delivered -- stalling reward. So
``CoDesignRobotWarehouse`` overrides ``_update_reward_and_request_queue`` to sample
replacement requests only among *active* (on-grid), not-already-queued shelves.

``build_state(key, active_mask)`` is a pure function of a PRNG key and a
``(max_shelves,)`` binary vector, so it ``jax.vmap``s over a batch of different
occupancies (the grid shape is constant, so shapes stay homogeneous).
"""

from __future__ import annotations

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # allocate VRAM on demand, not JAX's default ~75% pool
import jax
import jax.numpy as jnp
import numpy as np

from jumanji.environments.routing.robot_warehouse.env import RobotWarehouse
from jumanji.environments.routing.robot_warehouse.generator import GeneratorBase
from jumanji.environments.routing.robot_warehouse.types import State
from jumanji.environments.routing.robot_warehouse.utils import compute_action_mask
from jumanji.environments.routing.robot_warehouse.utils_shelf import update_shelf
from jumanji.environments.routing.robot_warehouse.utils_spawn import (
    _POSSIBLE_DIRECTIONS,
    spawn_agent,
    spawn_shelf,
)
from jumanji.environments.routing.robot_warehouse.constants import _AGENTS, _SHELVES

from .config import RobotWarehouseConfig
from .embodiment import base_embodiment


class CoDesignWarehouseGenerator(GeneratorBase):
    """Places shelves at embodiment-chosen (occupancy) positions on a fixed grid."""

    def __init__(self, cfg: RobotWarehouseConfig) -> None:
        super().__init__(
            shelf_rows=cfg.shelf_rows,
            shelf_columns=cfg.shelf_columns,
            column_height=cfg.column_height,
            num_agents=cfg.num_agents,
            sensor_range=cfg.sensor_range,
            request_queue_size=cfg.request_queue_size,
        )
        # Emergent aisles: replace Jumanji's corridor skeleton with a highway mask
        # that reserves the goal cells only. ``goals`` rows are ``(y, x)`` while the
        # grid/highways are indexed ``[x, y]`` (matching ``env.step``'s lookup
        # ``highways[agent.position.x, agent.position.y]``), so swap the columns.
        goals = jnp.asarray(self.goals)
        self._highways = (
            jnp.zeros(self._grid_size, dtype=self._highways.dtype)
            .at[goals[:, 1], goals[:, 0]]
            .set(True)
        )
        # Placement domain = every cell except the reserved goals. Shelf slot i is
        # bound to eligible cell i, so max_shelves == number of eligible cells.
        self._eligible = jnp.argwhere(jnp.abs(self.highways - 1))
        self._max_shelves = int(self._eligible.shape[0])
        self._min_shelves = int(cfg.num_shelves)  # design-pressure floor
        self._reference_mask = jnp.asarray(base_embodiment(cfg, self), dtype=jnp.int32)

        # Override the stock (full-block) shelf set: one slot per eligible cell.
        self._num_shelves = self._max_shelves
        self._shelf_ids = jnp.arange(self._max_shelves)
        self._not_in_queue_size = self._max_shelves - cfg.request_queue_size

    @property
    def num_shelves(self) -> int:
        return self._max_shelves

    @property
    def max_shelves(self) -> int:
        return self._max_shelves

    @property
    def min_shelves(self) -> int:
        return self._min_shelves

    @property
    def eligible_cells(self) -> np.ndarray:
        """``(n_eligible, 2)`` integer array of valid shelf cells (numpy)."""
        return np.asarray(self._eligible)

    def _base_mask(self) -> jnp.ndarray:
        """Occupancy vector for the reference layout with connected aisles."""
        return self._reference_mask

    def build_state(self, key, active_mask) -> State:
        """Build a warehouse ``State`` from a ``(max_shelves,)`` occupancy vector.

        Only active (bit-set) slots are placed on the grid; the initial request queue
        is drawn from active shelves. Agents spawn only on non-shelf cells (goals stay
        eligible -- agents must stand on them to deliver).
        """
        gx, gy = self._grid_size
        coords = self._eligible  # (max_shelves, 2); slot i lives at eligible cell i
        active = active_mask.astype(bool)
        num_agents = len(self._agent_ids)
        key, pos_key, dir_key, queue_key = jax.random.split(key, 4)

        # --- agents: sample distinct cells that are not occupied by an active shelf --
        shelf_flat = coords[:, 0] * gy + coords[:, 1]
        occupied = jnp.zeros((gx * gy,), dtype=bool).at[shelf_flat].set(active)
        free = ~occupied
        agent_flat = jax.random.choice(
            pos_key, gx * gy, shape=(num_agents,), replace=False, p=free / jnp.sum(free)
        )
        agent_coords = jnp.transpose(jnp.asarray(jnp.unravel_index(agent_flat, self._grid_size)))
        agent_dirs = jax.random.choice(dir_key, _POSSIBLE_DIRECTIONS, shape=(num_agents,))
        agents = jax.vmap(spawn_agent)(agent_coords, agent_dirs)

        # --- request queue: sample distinct requested shelves from ACTIVE slots ------
        active_p = active.astype(jnp.float32)
        request_queue = jax.random.choice(
            queue_key, self._max_shelves, shape=(self._request_queue_size,),
            replace=False, p=active_p / jnp.sum(active_p),
        )
        requested_ids = jnp.zeros((self._max_shelves,), dtype=jnp.int32).at[request_queue].set(1)
        shelves = jax.vmap(spawn_shelf)(coords, requested_ids)

        # --- grid: agents on the agent layer, only active shelves on the shelf layer -
        grid = jnp.zeros((2, gx, gy), dtype=jnp.int32)
        grid = grid.at[_AGENTS, agent_coords[:, 0], agent_coords[:, 1]].set(self._agent_ids + 1)
        shelf_vals = jnp.where(active, self._shelf_ids + 1, 0)
        grid = grid.at[_SHELVES, coords[:, 0], coords[:, 1]].set(shelf_vals)

        return State(
            grid=grid,
            agents=agents,
            shelves=shelves,
            request_queue=request_queue,
            step_count=jnp.array(0, int),
            action_mask=compute_action_mask(grid, agents),
            key=key,
        )

    def __call__(self, key) -> State:
        """Reference aisle layout used by the underlying environment reset."""
        return self.build_state(key, self._base_mask())


class CoDesignRobotWarehouse(RobotWarehouse):
    """RobotWarehouse whose request replenishment respects the active shelf set.

    Shelf slots are padded to ``max_shelves`` with only some active (on the grid).
    Stock replenishment samples ``setdiff1d(shelf_ids, request_queue)`` over the whole
    pool, which would request inactive, off-grid shelves (never deliverable) and stall
    reward. Here the replacement is drawn only from active (on-grid), not-in-queue
    shelves -- active is derived from the shelf layer of the grid.
    """

    def _update_reward_and_request_queue(self, key, reward, request_queue, grid, shelves, goal):
        y, x = goal
        shelf_id = grid[_SHELVES, x, y]

        def deliver(key, reward, request_queue, shelves, shelf_id):
            key, request_key = jax.random.split(key)
            # Active shelves are exactly those present on the shelf layer of the grid.
            seen = jnp.zeros((self.shelf_ids.shape[0] + 1,), dtype=bool)
            seen = seen.at[grid[_SHELVES].reshape(-1)].set(True)
            active = seen[1:]  # index i -> shelf id i present on the grid
            in_queue = jnp.zeros((self.shelf_ids.shape[0],), dtype=bool).at[request_queue].set(True)
            candidates = (active & ~in_queue).astype(jnp.float32)
            new_request_id = jax.random.choice(
                request_key, self.shelf_ids, p=candidates / jnp.maximum(jnp.sum(candidates), 1.0)
            )
            replace_index = jnp.argwhere(jnp.equal(request_queue, shelf_id - 1), size=1)
            request_queue = request_queue.at[replace_index].set(new_request_id)
            reward += 1.0
            shelves = update_shelf(shelves, shelf_id - 1, "is_requested", 0)
            shelves = update_shelf(shelves, new_request_id, "is_requested", 1)
            return key, reward, request_queue, shelves

        shelf_at_goal = (~jnp.equal(shelf_id, 0)) & jnp.isin(shelf_id, request_queue + 1)
        return jax.lax.cond(
            shelf_at_goal,
            deliver,
            lambda k, r, rq, g, _: (k, r, rq, g),
            key,
            reward,
            request_queue,
            shelves,
            shelf_id,
        )
