"""Native-vectorized Hopper co-design builder."""

from __future__ import annotations

from core.construction import warp_construction

from core.device import resolve_mujoco_device

from .codesign_env import HopperCoDesignEnv
from .config import make_config

_RUNNER_KEYS = ("render_mode", "use_graph", "seed", "viz_env_id", "debug")


@warp_construction
def build_vec(num_envs, *, is_eval_env=False, setup=None, device=None, **cfg):
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    # Action (3) and observation (11) sizes are fixed by the shared topology.
    runner = {k: cfg.pop(k) for k in _RUNNER_KEYS if k in cfg}
    runner["device"] = resolve_mujoco_device(device)
    config = make_config(setup, **cfg)
    return HopperCoDesignEnv(num_envs=num_envs, config=config, is_eval_env=is_eval_env, **runner)
