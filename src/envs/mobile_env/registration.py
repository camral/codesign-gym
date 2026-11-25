"""Conventional builder consumed by ``codesign_gym``'s registry.

``codesign_gym.make_vec`` looks up ``envs.mobile_env.registration:build_vec`` and
calls it with a batch size plus the common codesign-gym kwargs. A single builder
serves every preset via the ``setup`` argument (mirroring lq_structure/microgrid);
it returns the env's native vectorized form
(:class:`~envs.mobile_env.vector_env.MobileEnvVectorEnv`), which owns its batch of
mobile-env instances and exposes ``base_embodiment``.

Presets (see :data:`envs.mobile_env.config.PRESETS`):
  - ``dense_urban``   : city-centre, pedestrians, low-power small cells, budget 3-8 BS.
  - ``suburban``      : residential + roads, slow-vehicular, micro->macro, budget 3-7 BS.
  - ``rural_highway`` : sparse coverage, high mobility, few high-power macros, budget 1-4 BS.

Remaining keyword args override :class:`~envs.mobile_env.config.MobileEnvConfig`
fields (e.g. ``num_ues``, ``max_bs``, ``tx_max``, ``velocity``).
"""

from __future__ import annotations

from typing import Any, Optional

from .config import make_config
from .vector_env import MobileEnvVectorEnv


def build_vec(
    num_envs: int,
    *,
    setup: str = "dense_urban",
    is_eval_env: bool = False,
    render_mode: Optional[str] = None,
    **cfg_overrides: Any,
) -> MobileEnvVectorEnv:
    """Build the vectorized mobile-env co-design env for a given preset."""
    cfg = make_config(setup, **cfg_overrides)
    return MobileEnvVectorEnv(
        num_envs,
        cfg=cfg,
        is_eval_env=is_eval_env,
        render_mode=render_mode,
    )
