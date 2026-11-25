"""Batched design arguments: per-lane validate_embodiment, (num_envs,) rejection flags,
and None lanes (base design at reset, keep at step)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import codesign_gym as cg
from core.embodiment_batch import design_rows, lane_valid

CPU = dict(device="cpu", use_graph=False)


def test_design_rows_forms():
    base = np.zeros(3, np.float32)
    rows, given = design_rows(None, base, 2)
    assert rows.shape == (2, 3) and not given.any()
    rows, given = design_rows(np.ones(3), base, 2)
    assert (rows == 1).all() and given.all()
    rows, given = design_rows([None, np.full(3, 2.0)], base, 2)
    assert (rows[0] == 0).all() and (rows[1] == 2).all() and given.tolist() == [False, True]
    with pytest.raises(ValueError):
        design_rows(np.ones(4), base, 2)
    assert lane_valid(np.array([[0.0, 0.0], [2.0, np.nan]]), -np.ones(2), np.ones(2)).tolist() == [True, False]


@pytest.mark.parametrize("env_id", ["Hopper-v4", "BallCatcher-Pitch", "SolarCleaner-Reach", "TruckUnload-Single"])
def test_per_lane_validate_rejected_shape_and_none_lanes(env_id):
    env, base = cg.make_vec(env_id, 2, **CPU)
    try:
        lo, hi = env.offset_bounds
        assert np.asarray(env.validate_embodiment(np.stack([base, hi + 1.0]))).tolist() == [True, False]
        assert np.asarray(env.validate_embodiment([None, hi])).all()
        _, rejected, _ = env.reset(embodiment=[None, hi])
        assert tuple(rejected.shape) == (2,)
        installed = (env._current_embodiment if hasattr(env, "_current_embodiment") else env._embodiment_buf).cpu().numpy()
        np.testing.assert_allclose(installed[0], base)                      # None lane -> base design
        np.testing.assert_allclose(installed[1], hi)
    finally:
        env.close()


def test_single_env_validate_stays_a_bool():
    env, base = cg.make("Hopper-v4", **CPU)
    assert env.validate_embodiment(base) is True
    env.close()


@pytest.mark.parametrize("env_id", ["RotateInHand-Cube", "HandPickUp-Cube"])
def test_hand_step_none_lane_keeps_its_design(env_id):
    env, base = cg.make_vec(env_id, 2)
    try:
        lo, hi = env.offset_bounds
        mid = (0.5 * (lo + hi)).astype(np.float32)
        env.reset(embodiment=np.stack([mid, mid]))
        a = torch.zeros(2, env.act_dim(), device=env.device)
        env.step(a, embodiment=[None, hi])
        buf = env._reset_act_buf.cpu().numpy()
        np.testing.assert_allclose(buf[0], mid, atol=1e-6)                  # None lane kept its design
        np.testing.assert_allclose(buf[1], hi, atol=1e-6)
        assert tuple(env.reset()[1].shape) == (2,)
    finally:
        env.close()


def test_nerograsp_validates_per_env_and_flags_per_env():
    pytest.importorskip("mujoco_warp")
    env, base = cg.make_vec("NeroGrasp-Thin", 2, use_cpu=True)
    try:
        valid = np.asarray(env.validate_embodiment(base))
        assert valid.shape == (2,) and valid.all()
        assert np.asarray(env.validate_embodiment([None] * len(base))).all()   # None entries are the base design
        _, rejected, _ = env.reset(embodiment=[None] * len(base))
        assert tuple(rejected.shape) == (2,) and not bool(rejected.any())
    finally:
        env.close()
