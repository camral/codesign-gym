"""Conventional builder consumed by ``codesign_gym``'s registry.

``codesign_gym.make_vec`` looks up ``envs.robot_warehouse.registration:build_vec`` and
calls it with a batch size plus the common codesign-gym kwargs. A single builder serves
every preset via the ``setup`` argument (mirroring lq_structure/microgrid/mobile_env); it
returns the env's native (JAX-vmapped) vectorized form
(:class:`~envs.robot_warehouse.vector_env.RobotWarehouseVectorEnv`), which exposes
``base_embodiment``.

Presets (see :data:`envs.robot_warehouse.config.PRESETS`):
  - ``small``     : 10x10 floor, 2 robots, at least 25 shelves.
  - ``medium``    : 16x16 floor, 4 robots, at least 115 shelves.
  - ``congested`` : 20x22 floor, 8 robots, at least 264 shelves.

Remaining keyword args override :class:`~envs.robot_warehouse.config.RobotWarehouseConfig`
fields (e.g. ``num_agents``, ``num_shelves``, ``time_limit``).
"""

from __future__ import annotations

from typing import Any, Optional

from .config import make_config
from .vector_env import RobotWarehouseVectorEnv


def build_vec(
    num_envs: int,
    *,
    setup: str = "small",
    is_eval_env: bool = False,
    render_mode: Optional[str] = None,
    **cfg_overrides: Any,
) -> RobotWarehouseVectorEnv:
    """Build the vectorized RobotWarehouse co-design env for a given preset."""
    cfg = make_config(setup, **cfg_overrides)
    return RobotWarehouseVectorEnv(
        num_envs,
        cfg=cfg,
        is_eval_env=is_eval_env,
        render_mode=render_mode,
    )
