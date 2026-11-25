"""Stepping a vector env before its first reset() raises a clear error instead of running on unset state."""

import importlib.util

import numpy as np
import pytest

import codesign_gym as cg

MUJOCO = dict(device="cpu", use_graph=False)
CASES = [
    ("SolarCleaner-Reach", dict(MUJOCO, panel_rows=1, panel_cols=1, n_sensor_rows=1, n_sensor_cols=1), "mujoco_warp"),
    ("TruckUnload-Single", MUJOCO, "mujoco_warp"),
    ("Warehouse-Small", {}, "jumanji"),
    ("Racing-Spielberg", {}, "f1tenth_gym_jax"),
    ("SoftWalker-Spring2D", dict(device="cpu"), "warp"),
]


@pytest.mark.parametrize("env_id, cfg, module", CASES, ids=[c[0] for c in CASES])
def test_step_before_reset_raises(env_id, cfg, module):
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} not installed")
    env, _ = cg.make_vec(env_id, 2, **cfg)
    try:
        with pytest.raises(RuntimeError, match=r"call reset\(\) before step\(\)"):
            env.step(np.asarray(env.action_space.sample()))
    finally:
        env.close()
