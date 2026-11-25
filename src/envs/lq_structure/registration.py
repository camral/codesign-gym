"""Conventional builder consumed by ``codesign_gym``'s registry.

``codesign_gym.make_vec`` looks up ``envs.lq_structure.registration:build_vec`` and calls
it with a batch size plus the common codesign-gym kwargs. It returns the env's native
vectorized form (:class:`LQStructureVectorEnv`), which exposes ``base_embodiment``.
"""

from __future__ import annotations

from .config import make_config, ControllerType
from .vector_env import LQStructureVectorEnv


def build_vec(
    num_envs: int,
    *,
    setup: ControllerType = "lqr",
    is_eval_env: bool = False,
    render_mode: str | None = None,
    n: int = 6,
    **cfg_overrides,
) -> LQStructureVectorEnv:
    """Build the vectorized LQ-structure env for a given setup.

    ``setup`` selects the controller regime (lqr / lqg / qp); ``n`` is the chain size;
    remaining keyword args override :class:`LQConfig` fields.
    """
    cfg = make_config(setup, n=n, **cfg_overrides)
    return LQStructureVectorEnv(
        cfg,
        num_envs=num_envs,
        is_eval_env=is_eval_env,
        render_mode=render_mode,
    )
