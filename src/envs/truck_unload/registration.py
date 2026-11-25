"""Native-vectorized TruckUnload builder.

Importing this module runs the package ``__init__`` first, which picks MuJoCo's
GL backend before the env module below imports mujoco, so a driver reaching
``build_vec`` through the registry gets it.
"""

from __future__ import annotations

from core.construction import warp_construction
from core.device import resolve_mujoco_device

from .codesign_env import TruckUnloadCoDesignEnv
from .config import PRESETS


@warp_construction
def build_vec(num_envs, *, is_eval_env=False, setup="single", device=None, **cfg):
    """Build the TruckUnload co-design env for ``setup`` and return it.

    ``setup`` selects a preset in :mod:`config`; ``cfg`` overrides individual
    preset keys.
    """
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")

    cfg["device"] = resolve_mujoco_device(device)
    if setup not in PRESETS:
        raise ValueError(f"unknown setup {setup!r}; have {sorted(PRESETS)}")
    kwargs = {**PRESETS[setup], **cfg}

    return TruckUnloadCoDesignEnv(num_envs, is_eval_env=is_eval_env, **kwargs)
