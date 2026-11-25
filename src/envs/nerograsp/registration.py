"""Native-vectorized NeroGrasp and NeroTools builders."""

from dataclasses import replace
import importlib.util

from core.construction import warp_construction
from core.device import resolve_mujoco_device
from core.mujoco_rendering import validate_render_mode

from .config import NeroGraspConfig, NeroToolsConfig


def _prepare_config(num_envs, config_cls, *, use_visacd, config, kwargs):
    render_mode = kwargs.pop("render_mode", None)
    validate_render_mode(render_mode)
    if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    explicit_backend = "convex_decomposition_backend" in kwargs or bool(use_visacd)
    cfg = replace(config_cls() if config is None else config, **kwargs)
    if use_visacd:
        cfg = replace(cfg, convex_decomposition_backend="visacd")
    cfg.validate()
    if cfg.use_cpu and cfg.device is not None and str(cfg.device) != "cpu":
        raise ValueError("use_cpu=True conflicts with a non-CPU device")
    device = resolve_mujoco_device("cpu" if cfg.use_cpu else cfg.device)
    cfg = replace(cfg, device=device, use_cpu=device == "cpu")
    if not explicit_backend and cfg.decomposition_backend == "cuacd":
        if cfg.use_cpu or importlib.util.find_spec("cuacd") is None:
            cfg = replace(cfg, convex_decomposition_backend="vhacd")
    return cfg, render_mode


@warp_construction(forward_verbose=True)
def build_vec(num_envs, *, is_eval_env=False, use_visacd=False, config=None, **kwargs):
    """Build a NeroGrasp co-design env."""
    cfg, render_mode = _prepare_config(num_envs, NeroGraspConfig, use_visacd=use_visacd, config=config, kwargs=kwargs)
    from .codesign_env import NeroGraspCoDesignEnv

    return NeroGraspCoDesignEnv(config=cfg, num_envs=num_envs, is_eval_env=is_eval_env, render_mode=render_mode)


@warp_construction(forward_verbose=True)
def build_tools_vec(num_envs, *, is_eval_env=False, use_visacd=False, config=None, **kwargs):
    """Build a NeroGrasp-Tools (parallel gripper + articulated tool) co-design env."""
    cfg, render_mode = _prepare_config(num_envs, NeroToolsConfig, use_visacd=use_visacd, config=config, kwargs=kwargs)
    from .codesign_env import NeroToolsCoDesignEnv

    return NeroToolsCoDesignEnv(config=cfg, num_envs=num_envs, is_eval_env=is_eval_env, render_mode=render_mode)
