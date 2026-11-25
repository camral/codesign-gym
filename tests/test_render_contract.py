"""Public rendering behavior for a native MuJoCo Warp environment."""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

import codesign_gym as cg


def _viewer_unavailable() -> bool:
    """MuJoCo's passive viewer needs mjpython on macOS (it runs on the Cocoa UI thread) and a display elsewhere."""
    if sys.platform == "darwin":
        import mujoco.viewer
        return mujoco.viewer._MJPYTHON is None
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


needs_viewer = pytest.mark.skipif(_viewer_unavailable(), reason="interactive viewer needs a display (mjpython on macOS)")

DEVICE = os.environ.get("CODESIGN_RENDER_TEST_DEVICE", "cpu")


def test_mujoco_render_modes_and_vector_frames():
    with pytest.raises(ValueError, match="render_mode"):
        cg.make_vec("HalfCheetah-v4", num_envs=1, device=DEVICE, render_mode="bogus")

    env, _ = cg.make_vec("HalfCheetah-v4", num_envs=2, device=DEVICE, use_graph=False)
    try:
        env.reset(seed=0)
        assert env.render() is None
        assert env.metadata["render_modes"] == ["human", "rgb_array"]
    finally:
        env.close()

    env, _ = cg.make_vec("HalfCheetah-v4", num_envs=2, device=DEVICE, use_graph=False,
                         render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frames = env.render()
        assert isinstance(frames, tuple) and len(frames) == 2
        assert all(isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.shape[-1] == 3
                   and frame.dtype == np.uint8 for frame in frames)
        assert env.render(env_idx=1).shape == frames[1].shape
        with pytest.raises(IndexError):
            env.render(env_idx=2)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ["BallCatcher-Pitch", "SolarCleaner-Reach", "TruckUnload-Single"])
def test_other_mujoco_batch_families_return_one_frame_per_lane(env_id):
    env, _ = cg.make_vec(env_id, num_envs=2, device=DEVICE, use_graph=False,
                         render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frames = env.render()
        assert isinstance(frames, tuple) and len(frames) == 2
        assert all(isinstance(frame, np.ndarray) and frame.dtype == np.uint8
                   and frame.shape[-1] == 3 for frame in frames)
        assert env.render(env_idx=1).shape == frames[1].shape
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ["RotateInHand-Cube", "HandPickUp-Cube"])
def test_hand_batches_render_each_lane(env_id):
    env, _ = cg.make_vec(env_id, num_envs=2, device=DEVICE, use_graph=False,
                         render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frames = env.render()
        assert isinstance(frames, tuple) and len(frames) == 2
        assert all(frame.shape[-1] == 3 and frame.dtype == np.uint8 for frame in frames)
        assert env.render(env_idx=1).shape == frames[1].shape
    finally:
        env.close()


def test_nerograsp_render_mode_reaches_its_native_renderer():
    env, _ = cg.make_vec("NeroGrasp-Thin", num_envs=2, device=DEVICE,
                         render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frames = env.render()
        assert isinstance(frames, tuple) and len(frames) == 2
        assert all(frame.shape[-1] == 3 and frame.dtype == np.uint8 for frame in frames)
    finally:
        env.close()


@needs_viewer
def test_mujoco_human_viewer_refreshes_on_reset_and_step():
    env, _ = cg.make("HalfCheetah-v4", device=DEVICE, use_graph=False, render_mode="human")
    try:
        env.reset(seed=0)
        viewer = env.vector_env._human_viewer
        assert viewer is not None and viewer.is_running()
        assert env.render() is None
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert env.vector_env._human_viewer is viewer and viewer.is_running()
    finally:
        env.close()
    assert env.vector_env._human_viewer is None


@needs_viewer
def test_nerograsp_human_viewer_uses_retained_model():
    env, _ = cg.make("NeroGrasp-Thin", device=DEVICE, render_mode="human")
    try:
        env.reset(seed=0)
        viewer = env.vector_env._human_viewer
        assert viewer is not None and viewer.is_running()
        assert env.render() is None
    finally:
        env.close()


@needs_viewer
def test_hand_human_viewer_refreshes_heightfields():
    env, _ = cg.make("RotateInHand-Cube", device=DEVICE, use_graph=False, render_mode="human")
    try:
        env.reset(seed=0)
        viewer = env.vector_env._human_viewer
        assert viewer is not None and viewer.is_running()
        env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        assert viewer.is_running() and env.render() is None
    finally:
        env.close()


# Each env's knob for an episode of one control step
_ONE_STEP_EPISODE = {
    "BallCatcher-Pitch": {"max_episode_steps": 5},
    "SolarCleaner-Reach": {"next_cell_step_limit": 1, "next_panel_step_limit": 1},
    "TruckUnload-Single": {"time_limit_s": 0.04},
    "RotateInHand-Cube": {"total_limit_s": 0.04},
    "HandPickUp-Cube": {"total_limit_s": 0.04},
}


@pytest.mark.parametrize("env_id", sorted(_ONE_STEP_EPISODE))
def test_mujoco_batches_render_through_an_autoreset(env_id):
    env, base = cg.make_vec(env_id, num_envs=2, device=DEVICE, use_graph=False,
                            render_mode="rgb_array", **_ONE_STEP_EPISODE[env_id])
    try:
        env.reset(seed=0, embodiment=base)
        actions = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, terminated, truncated, _ = env.step(actions)
        assert bool((terminated | truncated).all())
        env.step(actions)                                  # respawns both copies
        frames = env.render()
        assert isinstance(frames, tuple) and len(frames) == 2
    finally:
        env.close()
