"""MuJoCo envs autoreset as Gymnasium NEXT_STEP: the step after a copy ends respawns it, returns its first observation
with reward 0 and no termination, ignores its action, and leaves every other copy's trajectory unchanged."""
import gymnasium as gym
import numpy as np
import pytest
import torch

from codesign_gym import make_vec

pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

IDS = ["HalfCheetah-v4", "Swimmer-v4", "Hopper-v4", "Walker2d-v4", "Ant-v4", "Humanoid-v4", "HumanoidStandup-v4",
       "BallCatcher-Pitch", "SolarCleaner-Reach", "TruckUnload-Single",
       "RotateInHand-Cube", "HandPickUp-Cube"]


def _actions(envs, seed):
    g = torch.Generator().manual_seed(seed)
    low, high = (torch.as_tensor(b, dtype=torch.float32) for b in (envs.single_action_space.low, envs.single_action_space.high))
    low, high = low.clamp(min=-1.0), high.clamp(max=1.0)
    return low + (high - low) * torch.rand((envs.num_envs, *envs.single_action_space.shape), generator=g)


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


@pytest.mark.slow
@pytest.mark.parametrize("env_id,kw", [(i, {}) for i in IDS] + [("Walker2d-v4", {"physics": "native"})],
                         ids=[*IDS, "Walker2d-v4-native"])
def test_next_step_autoreset(env_id, kw):
    """Copy 0 is made to end after step 3 in env A only. On step 4 it respawns (reward 0, no flags, a finite first
    observation, its step count back to 0); copy 1 sees the same observations and rewards in both envs."""
    a, _ = make_vec(env_id, 2, device="cpu", **kw)
    b, _ = make_vec(env_id, 2, device="cpu", **kw)
    assert a.metadata["autoreset_mode"] == gym.vector.AutoresetMode.NEXT_STEP
    a.reset(seed=0)
    b.reset(seed=0)
    for t in range(6):
        act = _actions(a, t).to(a.device)
        if t == 3:
            a.reset_mask[0] = True                  # as if copy 0 ended on step 3
            a._reset_pending = True
        obs_a, r_a, te_a, tr_a, _ = a.step(act)
        obs_b, r_b, _, _, _ = b.step(act.clone())
        if t == 3:
            assert float(r_a[0]) == 0.0 and not bool(te_a[0]) and not bool(tr_a[0])
            assert int(a.world_steps[0]) == 0
            assert np.isfinite(_np(obs_a[0])).all()
        np.testing.assert_allclose(_np(obs_a[1]), _np(obs_b[1]), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(float(r_a[1]), float(r_b[1]), rtol=1e-5, atol=1e-6)
    assert int(a.world_steps[0]) == 2 * a.phys_substeps   # two real steps since the respawn
    a.close()
    b.close()


@pytest.mark.slow
def test_final_steps_in_control_steps_and_reset_step():
    envs, _ = make_vec("Hopper-v4", 2, device="cpu", max_episode_steps=5)
    envs.reset(seed=0)
    zero = torch.zeros(envs.action_space.shape)
    for _ in range(4):
        envs.step(zero)
    _, _, _, truncated, info = envs.step(zero)
    assert truncated.all() and _np(info["final_steps"]).tolist() == [5, 5]
    _, reward, terminated, truncated, info = envs.step(zero)
    assert _np(reward).tolist() == [0.0, 0.0] and not terminated.any() and not truncated.any() and "final_obs" not in info
    envs.close()
