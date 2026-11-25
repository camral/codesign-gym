"""Native-vectorized PanelCleaner builder.

Importing this module runs the package ``__init__`` first, which picks MuJoCo's
GL backend before the env module below imports mujoco, so a driver reaching
``build_vec`` through the registry gets it.
"""

from __future__ import annotations

from core.construction import warp_construction
from core.device import resolve_mujoco_device

from .codesign_env import PanelCleanerCoDesignEnv
from .config import PRESETS


@warp_construction
def build_vec(num_envs, *, is_eval_env=False, setup="reach", device=None, **cfg):
    """Build the PanelCleaner co-design env for ``setup`` and return it.

    ``setup`` selects a preset in :mod:`config`; ``cfg`` overrides individual
    preset keys.
    """
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")

    cfg["device"] = resolve_mujoco_device(device)
    if setup not in PRESETS:
        raise ValueError(f"unknown setup {setup!r}; have {sorted(PRESETS)}")
    kwargs = {**PRESETS[setup], **cfg}

    # The evaluation env is the one that gets rendered to video: put its camera
    # behind the robot's torso, tracking it from the back as it walks up the
    # array (camera_mode="track"), rather than the fixed whole-scene view used
    # elsewhere. An explicit camera_mode in cfg still wins.
    if is_eval_env and "camera_mode" not in cfg:
        kwargs["camera_mode"] = "track"

    return PanelCleanerCoDesignEnv(num_envs, is_eval_env=is_eval_env, **kwargs)
