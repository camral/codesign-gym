"""Conventional builder consumed by ``codesign_gym``'s registry.

``codesign_gym.make_vec`` looks up ``envs.f1tenth.registration:build_vec`` and calls it with a
batch size plus the common codesign-gym kwargs. It returns the env's native (JAX-vmapped)
vectorized form (:class:`~envs.f1tenth.vector_env.F1TenthVectorEnv`), which exposes
``base_embodiment`` (the stock F1TENTH car).

Presets select the track; the design ranges and physical couplings live in
:class:`~envs.f1tenth.config.F1TenthConfig`.
"""

from __future__ import annotations

from typing import Any, Optional

from .config import make_config
from .vector_env import F1TenthVectorEnv


def build_vec(
    num_envs: int,
    *,
    setup: str = "spielberg",
    is_eval_env: bool = False,
    render_mode: Optional[str] = None,
    **cfg_overrides: Any,
) -> F1TenthVectorEnv:
    """Build the vectorized F1TENTH co-design env for a given preset (track)."""
    cfg = make_config(setup, **cfg_overrides)
    return F1TenthVectorEnv(
        num_envs,
        cfg=cfg,
        is_eval_env=is_eval_env,
        render_mode=render_mode,
    )
