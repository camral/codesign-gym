"""Changing action cadence must retain the physical timestep and episode time."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("gymnasium")

import codesign_gym as cg
from envs.soft_walker.engine.warp_native.solid_model import STATE_FIELDS
from envs.soft_walker import config as C
from envs.soft_walker.vector_env import SoftWalkerVectorEnv

DEVICE = os.environ.get("CODESIGN_ACTION_HOLD_TEST_DEVICE", "cpu")


@pytest.fixture
def tiny_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a short fixed physical budget to test action-horizon rounding and autoreset."""
    monkeypatch.setattr(C, "BASE_GRID", {2: (2, 1), 3: (2, 1, 1)})
    monkeypatch.setattr(C, "CONTROL_DT", 0.0004)
    monkeypatch.setattr(C, "SUBSTEPS", 4)
    monkeypatch.setattr(C, "SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(C, "HORIZON", 6)


@pytest.mark.parametrize("driver", C.DRIVERS)
def test_all_drivers_keep_physics_dt_and_settling_with_ten_step_hold(driver: str) -> None:
    default = SoftWalkerVectorEnv(1, driver=driver, device="cpu")
    held = SoftWalkerVectorEnv(1, driver=driver, device="cpu", physics_steps_per_action=10)
    try:
        assert default.physics_steps_per_action == 40
        assert default.control_dt == pytest.approx(0.01)
        assert default.horizon == 240
        assert held.physics_steps_per_action == 10
        assert held.physics_dt == default.physics_dt == 0.00025
        assert held.control_dt == pytest.approx(0.0025)
        assert held.horizon == 960
        assert held.target_episode_duration_s == pytest.approx(2.4)
        assert held.episode_duration_s == pytest.approx(default.episode_duration_s)
        assert held.settling_duration_s == pytest.approx(0.2)
        assert held.lanes[0].settling_steps == default.lanes[0].settling_steps == 800
        assert held.lanes[0].template.config == default.lanes[0].template.config
        np.testing.assert_array_equal(
            held.lanes[0].template.fields["reference"],
            default.lanes[0].template.fields["reference"],
        )
        for name in (
            "physics_steps_per_action",
            "physics_dt",
            "control_dt",
            "horizon",
            "target_episode_duration_s",
            "episode_duration_s",
            "settling_duration_s",
        ):
            assert held.metadata[name] == getattr(held, name)
    finally:
        default.close()
        held.close()


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 10.0, 1.5, "10", np.int64(10)])
def test_public_builder_requires_a_positive_plain_int(invalid: object) -> None:
    with pytest.raises(ValueError, match="physics_steps_per_action"):
        cg.make_vec("SoftWalker-Spring2D", num_envs=1, device="cpu", physics_steps_per_action=invalid)


def test_horizon_rounds_up_without_changing_physics_dt() -> None:
    env = SoftWalkerVectorEnv(1, driver="spring", device="cpu", physics_steps_per_action=7)
    try:
        assert env.horizon == (240 * 40 + 6) // 7
        assert env.physics_dt == 0.00025
        assert env.target_episode_duration_s == pytest.approx(2.4)
        assert env.episode_duration_s == pytest.approx(env.horizon * 7 * 0.00025)
        assert 0 <= env.episode_duration_s - env.target_episode_duration_s < env.control_dt
    finally:
        env.close()


def test_ten_physics_steps_equal_ten_single_steps_and_summed_rewards(tiny_task):
    held, _ = cg.make_vec("SoftWalker-Spring2D", num_envs=2, device=DEVICE, physics_steps_per_action=10)
    single, _ = cg.make_vec("SoftWalker-Spring2D", num_envs=2, device=DEVICE, physics_steps_per_action=1)
    try:
        assert not held.reset(seed=13)[1].any()
        assert not single.reset(seed=13)[1].any()
        action = np.zeros(held.action_space.shape)
        action[:, :2] = [[0.15, -0.05], [-0.1, 0.12]]
        result = held.step(action)
        reward_sum = np.zeros(2)
        for _ in range(10):
            reference = single.step(action)
            reward_sum += reference[1]
            assert not reference[2].any() and not reference[3].any()
        assert not result[2].any() and not result[3].any()
        np.testing.assert_allclose(result[0], reference[0], rtol=2e-8, atol=2e-10)
        np.testing.assert_allclose(result[1], reward_sum, rtol=2e-8, atol=2e-10)
        for whole, increment in zip(held.lanes, single.lanes):
            for field in STATE_FIELDS:
                np.testing.assert_allclose(
                    getattr(whole.owner.initial, field).numpy(),
                    getattr(increment.owner.initial, field).numpy(),
                    rtol=2e-8,
                    atol=2e-10,
                )
            assert whole._steps == 1 and increment._steps == 10
            assert whole.owner.initial.ledger.numpy()[3] == pytest.approx(held.control_dt)
        assert held.physics_dt == single.physics_dt
        assert held.control_dt == pytest.approx(10 * single.control_dt)
    finally:
        held.close()
        single.close()


def test_action_horizon_autoresets_after_rounded_physical_duration(tiny_task: None) -> None:
    env = SoftWalkerVectorEnv(2, driver="spring", device=DEVICE, physics_steps_per_action=10)
    try:
        env.reset(seed=2)
        assert env.horizon == 3
        assert env.target_episode_duration_s == pytest.approx(0.0024)
        assert env.episode_duration_s == pytest.approx(0.003)
        for index in range(env.horizon):
            result = env.step(np.zeros(env.action_space.shape))
            assert not result[2].any()
            assert result[3].all() == (index == env.horizon - 1)
        for lane in env.lanes:
            assert lane.owner.initial.ledger.numpy()[3] == pytest.approx(env.episode_duration_s)
        reset = env.step(np.zeros(env.action_space.shape))
        assert reset[4]["autoreset"].all()
        assert not reset[1].any() and not reset[2].any() and not reset[3].any()
        assert all(lane._steps == 0 for lane in env.lanes)
        assert all(lane.owner.initial.ledger.numpy()[3] == 0.0 for lane in env.lanes)
    finally:
        env.close()


def test_default_hold_preserves_existing_env_kwargs_customizations(
    tiny_task: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    timing = C.env_kwargs("spring") | {"substeps": 2, "horizon": 5}
    monkeypatch.setattr(C, "env_kwargs", lambda driver: timing)
    env = SoftWalkerVectorEnv(1, driver="spring", device="cpu")
    try:
        assert env.physics_steps_per_action == 2 and env.horizon == 5
        assert env.control_dt == pytest.approx(0.0002)
        assert env.episode_duration_s == env.target_episode_duration_s == pytest.approx(0.001)
    finally:
        env.close()
