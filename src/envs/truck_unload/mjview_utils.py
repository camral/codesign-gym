"""Interactive MuJoCo viewer helpers for a batched env.

The passive viewer shows one world (``env.viz_env_id``) while every world keeps
stepping on the GPU. It draws from its own model copy, into which each sync
writes the shown world, so switching worlds keeps the window. Keys: Space/P
pause or resume, ``]``/``[`` switch world.

- launch_viewer: open the viewer on a new model copy;
- sync_viewer / wait_while_paused: copy the shown world to the CPU and repaint;
- apply_pending_env_switch: apply a queued world switch.
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
    """Copy the shown world off the GPU and repaint the viewer."""
    m, d, w = env._viewer_m, env._viewer_d, env.viz_env_id
    with viewer.lock():                          # keeps the render thread out during the write
        env._mirror_render_model(m, w)
        with env._wp_scope():
            mjw.get_data_into(d, m, env.gpu_data, w)
        mujoco.mj_kinematics(m, d)               # get_data_into copies state, not world-frame poses
        if env.debug:
            viewer.user_scn.ngeom = 0
            env._add_debug_visualizations(viewer.user_scn, w)
    viewer.sync()


def wait_while_paused(viewer, pause_state: dict, env) -> None:
    while viewer and pause_state["paused"] and viewer.is_running():
        if pause_state.get("viz_env_delta"):
            return  # let the caller apply the world switch, then re-enter
        sync_viewer(env, viewer)
        time.sleep(0.05)


def launch_viewer(env, pause_state: dict):
    """Open the passive viewer on a new model copy of the env, and return it."""
    env._viewer_m = env._make_render_model()
    env._viewer_d = mujoco.MjData(env._viewer_m)
    viewer = mujoco.viewer.launch_passive(
        env._viewer_m, env._viewer_d,
        key_callback=make_viewer_key_callback(pause_state))
    env._setup_camera(viewer.cam)
    if hasattr(mujoco.mjtVisFlag, "mjVIS_REFLECTION"):
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_REFLECTION] = 0
    print("[viewer] Space/P pause | ] next world | [ prev world", flush=True)
    return viewer


def apply_pending_env_switch(env, viewer, pause_state) -> bool:
    """Apply a queued ``[``/``]`` world switch. Return True if the world changed."""
    delta = pause_state.pop("viz_env_delta", 0)
    if not delta or viewer is None or not viewer.is_running():
        return False
    env.viz_env_id = (env.viz_env_id + delta) % env.num_envs
    print(f"[viewer] showing world {env.viz_env_id}", flush=True)
    sync_viewer(env, viewer)
    return True
