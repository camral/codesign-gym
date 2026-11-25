"""Interactive MuJoCo viewer helpers for a batched env.

The passive viewer shows one world (``env.viz_env_id``) while every world keeps
stepping on the GPU. Keys: Space/P pause or resume, ``]``/``[`` switch world.
"""

import time

import mujoco
import mujoco.viewer
import mujoco_warp as mjw


def make_viewer_key_callback(pause_state: dict):
    def _on_key(keycode):
        if keycode in (ord(" "), ord("P"), ord("p")):
            pause_state["paused"] = not pause_state["paused"]
            print("[viewer] " + ("paused" if pause_state["paused"] else "resumed"),
                  flush=True)
        elif keycode == ord("]"):
            pause_state["viz_env_delta"] = pause_state.get("viz_env_delta", 0) + 1
        elif keycode == ord("["):
            pause_state["viz_env_delta"] = pause_state.get("viz_env_delta", 0) - 1
    return _on_key


def sync_viewer(env, viewer) -> None:
    """Copy the selected world off the GPU and repaint the viewer."""
    env._mirror_render_model(env._viewer_m, env.viz_env_id)
    with env._wp_scope():
        mjw.get_data_into(env._viewer_d, env._viewer_m, env.gpu_data, env.viz_env_id)
    mujoco.mj_kinematics(env._viewer_m, env._viewer_d)
    if env.debug:
        viewer.user_scn.ngeom = 0
        env._add_debug_visualizations(viewer.user_scn, env.viz_env_id)
    viewer.sync()


def wait_while_paused(viewer, pause_state: dict, env) -> None:
    while viewer and pause_state["paused"] and viewer.is_running():
        if pause_state.get("viz_env_delta"):
            return
        sync_viewer(env, viewer)
        time.sleep(0.05)


def refresh_viewer(env, viewer, current_render_model, pause_state):
    """Launch the passive viewer on a single CPU render copy of ``env.m``."""
    if viewer is None:
        env._viewer_m = env._make_render_model()
        env._viewer_d = mujoco.MjData(env._viewer_m)
        env._mirror_render_model(env._viewer_m, env.viz_env_id)
        viewer = mujoco.viewer.launch_passive(
            env._viewer_m, env._viewer_d,
            key_callback=make_viewer_key_callback(pause_state))
        current_render_model = env._viewer_m
        env._setup_camera(viewer.cam)
        if hasattr(mujoco.mjtVisFlag, "mjVIS_REFLECTION"):
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_REFLECTION] = 0
        print("[viewer] Space/P pause | ] next world | [ prev world", flush=True)
    return viewer, current_render_model


def apply_pending_env_switch(env, viewer, current_render_model, pause_state):
    """Apply a queued ``[``/``]`` world switch. Returns (viewer, model, switched)."""
    delta = pause_state.pop("viz_env_delta", 0)
    if not delta or viewer is None or not viewer.is_running():
        return viewer, current_render_model, False
    env.viz_env_id = (env.viz_env_id + delta) % env.num_envs
    print(f"[viewer] showing world {env.viz_env_id}", flush=True)
    sync_viewer(env, viewer)
    return viewer, current_render_model, True
