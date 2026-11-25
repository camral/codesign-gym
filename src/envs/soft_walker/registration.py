"""Conventional builder consumed by ``codesign_gym``'s registry.

``codesign_gym.make_vec`` looks up ``envs.soft_walker.registration:build_vec``. One builder serves every
``SoftWalker-<Driver>[Stairs|Gaps]`` id via the ``driver`` argument (``spring``, ``mpm2``, ``mpm3``, ``beam``, ``fem``) and
``terrain`` (``flat``, ``stairs``, ``gaps`` for every driver), and returns
:class:`~envs.soft_walker.vector_env.SoftWalkerVectorEnv` on the native Warp backend.

``dtype`` defaults to float64 and ``device`` selects CUDA when available. The
``execution`` option defaults to ``auto`` (concurrent CUDA streams, serial CPU);
``serial`` retains sequential execution for comparison, and ``parallel`` requires
CUDA. ``render_mode`` accepts ``None`` or ``'rgb_array'``. The physical task is
fixed per id (see :mod:`envs.soft_walker.config`). ``physics_steps_per_action``
optionally changes action frequency at the same physics dt; the default keeps
40 substeps per action, and episode duration rounds up by less than one control
interval when the requested hold does not divide the preset episode length.
"""

from __future__ import annotations

from core.construction import warp_construction

from typing import Optional

from . import config as C
from .vector_env import SoftWalkerVectorEnv


@warp_construction
def build_vec(num_envs: int, *, driver: str, is_eval_env: bool = False, dtype: str = C.DEFAULT_DTYPE, device: Optional[str] = None,
              terrain: str = "flat", render_mode: Optional[str] = None, execution: str = "auto",
              physics_steps_per_action: Optional[int] = None) -> SoftWalkerVectorEnv:
    """Build the vectorized SoftWalker co-design env for one physics driver on one terrain."""
    return SoftWalkerVectorEnv(num_envs, driver=driver, is_eval_env=is_eval_env, dtype=dtype, device=device, terrain=terrain,
                               render_mode=render_mode, execution=execution, physics_steps_per_action=physics_steps_per_action)
