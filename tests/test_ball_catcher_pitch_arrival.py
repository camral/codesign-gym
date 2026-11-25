"""Native Pitch launches cross the calibrated low, broad band."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pitch_crossing_distribution_and_reset_reproducibility():
    from codesign_gym import make_vec

    env, base = make_vec("BallCatcher-Pitch", 512, device="cuda", use_graph=True)
    try:
        env.reset(seed=11)
        source = env.qpos_torch()[:, env._ball_qadr:env._ball_qadr+3].clone()
        velocity = env.qvel_torch()[:, env._ball_dadr:env._ball_dadr+3].clone()
        torch.testing.assert_close(source, torch.tensor([8., 2., 1.], device="cuda").expand_as(source))
        time = (.4648899-source[:, 0])/velocity[:, 0]
        gravity = torch.as_tensor(env.m.opt.gravity, device="cuda")
        crossing = source+velocity*time[:, None]+.5*gravity*time[:, None]**2
        assert ((time >= .375-1e-6) & (time <= .425+1e-6)).all()
        assert ((crossing[:, 1] >= -1.3-1e-5) & (crossing[:, 1] <= 1.7+1e-5)).all()
        assert ((crossing[:, 2] >= 1.35-1e-5) & (crossing[:, 2] <= 1.65+1e-5)).all()
        assert crossing[:, 1].std() > .75 and crossing[:, 2].std() > .075
        assert time.std() > .012
        assert not torch.as_tensor(base).any()
        env.reset(seed=11, embodiment=env.embodiment_space.high)
        torch.testing.assert_close(env.qvel_torch()[:, env._ball_dadr:env._ball_dadr+3], velocity, rtol=0, atol=0)
        mask = torch.arange(env.num_envs, device="cuda") % 2 == 0
        untouched = env.qpos_torch()[~mask].clone()
        env.reset(seed=12, options={"reset_mask": mask})
        torch.testing.assert_close(env.qpos_torch()[~mask], untouched, rtol=0, atol=0)
        updated = env.qvel_torch()[:, env._ball_dadr:env._ball_dadr+3]
        torch.testing.assert_close(updated[~mask], velocity[~mask], rtol=0, atol=0)
        assert not torch.equal(updated[mask], velocity[mask])
        with pytest.raises(ValueError, match="arrival-time"):
            env.set_pitch(vel=10)
        obs, reward, _, _, _ = env.step(torch.zeros_like(env._act_buf))
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    finally:
        env.close()


@pytest.mark.parametrize("kwargs", [
    {"pitch_arrival_time": 0}, {"pitch_arrival_time": .02, "pitch_arrival_spread": .025},
    {"pitch_arrival_time": float("nan")}, {"pitch_arrival_spread": -1},
    {"pitch_target_half_widths": [0, -1, 0]}, {"pitch_target_half_widths": [0, 1]},
])
def test_invalid_arrival_configuration(kwargs):
    from envs.ball_catcher.ball_catcher_env import BallCatcherParallelEnv

    with pytest.raises(ValueError):
        BallCatcherParallelEnv(**kwargs)
