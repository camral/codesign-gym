"""Native-vectorized HandPickUp builder.

Importing this module runs the package ``__init__`` first, which picks MuJoCo's
GL backend before the env module below imports mujoco, so a driver reaching
``build_vec`` through the registry gets it.
"""

from __future__ import annotations

from core.construction import warp_construction
from core.device import resolve_mujoco_device

from .codesign_env import HandPickUpCoDesignEnv
from .config import PRESETS


@warp_construction
def build_vec(num_envs, *, is_eval_env=False, setup="cube", device=None, **cfg):
    """Build the HandPickUp co-design env for ``setup`` and return it.

    ``setup`` selects a preset in :mod:`config`; ``cfg`` overrides individual
    preset keys (and carries the runner's ``device``, ``seed``, ``render_mode``).
    The action and observation sizes are fixed by the hand.
    """
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")

    cfg["device"] = resolve_mujoco_device(device)
    if setup not in PRESETS:
        raise ValueError(f"unknown setup {setup!r}; have {sorted(PRESETS)}")
    return HandPickUpCoDesignEnv(num_envs=num_envs, is_eval_env=is_eval_env,
                                 **{**PRESETS[setup], **cfg})
