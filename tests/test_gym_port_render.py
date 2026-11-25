"""Render tests for the MuJoCo gym ports: the camera follows the body and the drawn floor reaches it far downrange."""

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

import warp as wp  # noqa: E402
import codesign_gym as cg  # noqa: E402

PORTS = ["HalfCheetah-v4", "Ant-v4", "Walker2d-v4", "Hopper-v4", "Swimmer-v4", "Humanoid-v4"]
# Downrange distance to teleport to: far past the stock 20/40 m floors, well inside a 1000 m one.
FAR_X = 300.0


def _body_bbox(frame):
    # Bodies are orange/tan (R well above B); the floor checker and sky are grey/green (R ~ B).
    ys, xs = np.nonzero(frame[..., 0].astype(int) - frame[..., 2].astype(int) > 60)
    assert len(xs) > 100, "body not in frame"
    # Bottom edge left out: the reflective floor mirrors the body, and that reflection changes with the checker under it.
    return np.array([xs.min(), xs.max(), ys.min()])


def _floor_drawn(frame):
    # Bottom rows are mostly floor (MuJoCo blue checker, B > G > R) when the floor is drawn
    # under the camera, and the darker sky/background past its edge otherwise.
    r, g, b = frame[-60:].reshape(-1, 3).astype(int).T
    return ((b > r + 10) & (b >= g) & (r + g + b > 140)).mean() > 0.5


def _render_at(inner, x):
    q = inner.gpu_data.qpos.numpy()
    q[:, 0] = x
    wp.copy(inner.gpu_data.qpos, wp.array(q, dtype=inner.gpu_data.qpos.dtype, device=inner.gpu_data.qpos.device))
    return inner.render(0)


@pytest.mark.parametrize("env_id", PORTS)
def test_camera_tracks_body_over_drawn_floor(env_id):
    env, base = cg.make_vec(env_id, num_envs=1, device="cpu", use_graph=False, render_mode="rgb_array")
    try:
        env.reset(seed=0, embodiment=np.asarray(base)[None])
        try:
            near = _render_at(env, 0.0)
        except Exception as exc:  # no OpenGL context on this machine
            pytest.skip(f"offscreen rendering unavailable: {exc}")
        far = _render_at(env, FAR_X)
        assert np.abs(_body_bbox(near) - _body_bbox(far)).max() <= 5, "camera did not follow the body"
        assert _floor_drawn(near) and _floor_drawn(far), "floor not drawn under the body"
    finally:
        env.close()
