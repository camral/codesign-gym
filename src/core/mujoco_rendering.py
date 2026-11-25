"""Public render-mode handling for native MuJoCo Warp batches.

The simulators own their offscreen renderer and model mirroring. This module only
selects the requested lane(s) and manages a live passive MuJoCo viewer.
"""

from __future__ import annotations

import os
import sys

import mujoco
import mujoco.viewer
import mujoco_warp as mjw
import warp as wp


RENDER_MODES = ("human", "rgb_array")


def validate_render_mode(mode):
    if mode is not None and mode not in RENDER_MODES:
        raise ValueError(f"unsupported render_mode {mode!r}; choose from {RENDER_MODES}")


def render_batch(env, draw, env_idx=None):
    """Gymnasium vector result, with optional single-lane selection."""
    if env_idx is not None:
        env_idx = int(env_idx)
        if not 0 <= env_idx < env.num_envs:
            raise IndexError("env_idx out of range")
    if env.render_mode is None:
        return None
    if env.render_mode == "human":
        show_human(env, env_idx)
        return None
    if env_idx is not None:
        return draw(env, env_idx=env_idx)
    return tuple(draw(env, env_idx=i) for i in range(env.num_envs))


def show_human(env, env_idx=None):
    """Show one device world in MuJoCo's interactive passive viewer."""
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError('render_mode="human" requires a display; use "rgb_array" on a headless host')
    index = env.viz_env_id if env_idx is None else int(env_idx)
    if not 0 <= index < env.num_envs:
        raise IndexError("env_idx out of range")
    viewer = getattr(env, "_human_viewer", None)
    # Variant-model envs have one compiled CPU model per lane. The other envs
    # mirror each lane into a private CPU model shared by the viewer.
    variants = getattr(env, "variants", None)
    legacy = hasattr(env, "cp_data") and not hasattr(env, "gpu_data")
    model = variants[index] if variants else None
    if legacy and model is None:
        if index != 0 and variants:
            raise ValueError(f"No retained render model for environment {index}")
        model = env.m
    if viewer is not None and not viewer.is_running():
        close_human(env)
        viewer = None
    if viewer is not None and model is not None and env._human_model is not model:
        close_human(env)
        viewer = None
    if viewer is None:
        if model is None:
            model = env._make_render_model()
        data = mujoco.MjData(model)
        viewer = mujoco.viewer.launch_passive(model, data)
        env._human_model, env._human_data, env._human_viewer = model, data, viewer
        env._setup_camera(viewer.cam)
    env.viz_env_id = index
    model, data = env._human_model, env._human_data
    with viewer.lock():
        if legacy:
            with wp.ScopedDevice(env.device):
                mjw.get_data_into(data, model, env.cp_data, index)
        elif variants:
            env._mirror_render_model(index)
            with env._wp_scope():
                mjw.get_data_into(data, model, env.gpu_data, index)
        else:
            env._mirror_render_model(model, index)
            with env._wp_scope():
                mjw.get_data_into(data, model, env.gpu_data, index)
        mujoco.mj_forward(model, data)
        if env.debug:
            viewer.user_scn.ngeom = 0
            env._add_debug_visualizations(viewer.user_scn, index)
    # MuJoCo uploads heightfields and textures through the viewer thread. Its
    # update methods acquire the lock themselves, so they must run outside it.
    if getattr(env, "_viewer_hfields_dirty", False):
        for hfield_id in env.hfield_ids:
            viewer.update_hfield(int(hfield_id))
        for texture_id in getattr(env, "pad_color_tex_ids", ()):
            viewer.update_texture(int(texture_id))
        env._viewer_hfields_dirty = False
    viewer.sync()


def close_human(env):
    viewer = getattr(env, "_human_viewer", None)
    if viewer is not None:
        viewer.close()
        env._human_viewer = None
        env._human_model = None
        env._human_data = None
