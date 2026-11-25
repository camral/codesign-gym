"""Tests for the Ant-v4 co-design port (CPU / no CUDA graph)."""

import warnings

import numpy as np
import pytest

pytest.importorskip("warp")
pytest.importorskip("mujoco_warp")

import torch  # noqa: E402
import codesign_gym as cg  # noqa: E402
from codesign_gym.single_env import SingleEnv  # noqa: E402

ENV_ID = "Ant-v4"
KW = dict(device="cpu", use_graph=False)
N_SLOTS = 36    # 12 leg links (hip+thigh+ankle x 4) x (x, y, z)
ACT_DIM, OBS_DIM = 8, 27


def _make_vec(n=4):
    return cg.make_vec(ENV_ID, num_envs=n, **KW)


def test_registry_lists_env():
    assert ENV_ID in cg.list_envs()


def test_base_embodiment_and_space():
    env, base = _make_vec(2)
    assert np.array_equal(base, env.base_embodiment)
    assert base.shape == (N_SLOTS,) and base.dtype == np.float32
    assert np.array_equal(base, np.zeros(N_SLOTS, dtype=np.float32))
    assert env.embodiment_space.contains(base)
    assert env.validate_embodiment(base).all()      # one flag per lane
    assert env.single_action_space.shape == (ACT_DIM,)
    assert env.single_observation_space.shape == (OBS_DIM,)


def test_make_vec_reset_is_3tuple_and_step_is_5tuple():
    env, _ = _make_vec(4)
    out = env.reset(seed=0)
    assert len(out) == 3
    obs, rejected, info = out
    assert tuple(obs.shape) == (4, OBS_DIM)
    assert not bool(np.asarray(rejected).any())
    assert env.observation_space.contains(obs.cpu().numpy())
    obs, reward, terminated, truncated, info = env.step(torch.zeros(4, ACT_DIM))
    assert tuple(obs.shape) == (4, OBS_DIM)
    assert tuple(reward.shape) == (4,)
    assert bool(torch.isfinite(reward).all())
    assert terminated.shape == (4,) and truncated.shape == (4,)


def test_make_single_env():
    env, _ = cg.make(ENV_ID, **KW)
    assert isinstance(env, SingleEnv)
    obs, rejected, info = env.reset(seed=0)
    assert np.asarray(obs).shape == (OBS_DIM,)
    assert rejected is False or rejected == False  # noqa: E712
    obs, reward, terminated, truncated, info = env.step(np.zeros(ACT_DIM, dtype=np.float32))
    assert np.ndim(reward) == 0 and np.isfinite(float(reward))
    assert np.ndim(terminated) == 0


def test_healthy_step_gets_healthy_reward():
    env, _ = _make_vec(4)
    env.reset(seed=0)
    obs, reward, term, trunc, info = env.step(torch.zeros(4, ACT_DIM))
    assert not bool(term.any())            # upright ant is healthy
    assert bool((reward > 0.5).all())      # dominated by the +1 healthy reward


def test_rollout_stays_finite():
    env, _ = _make_vec(4)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs, reward, term, trunc, info = env.step(rng.uniform(-1, 1, (4, ACT_DIM)).astype(np.float32))
        assert bool(torch.isfinite(reward).all())
        assert bool(torch.isfinite(obs).all())


@pytest.mark.parametrize("bad,exc", [
    (np.zeros(N_SLOTS - 1, dtype=np.float32), ValueError),
    (np.full(N_SLOTS, 2.0, dtype=np.float32), ValueError),
    (np.full(N_SLOTS, np.nan, dtype=np.float32), ValueError),
])
def test_embodiment_rejection(bad, exc):
    env, _ = _make_vec(2)
    with pytest.raises(exc):
        env.reset(embodiment=bad)


def test_transition_embodiment_is_frozen_with_warning():
    env, _ = _make_vec(2)
    env.reset(seed=0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env.step(torch.zeros(2, ACT_DIM), embodiment=np.zeros(N_SLOTS, dtype=np.float32))
    assert any("Transition" in str(w.message) for w in caught)


def test_asymmetric_designs_change_mass():
    import warp as wp
    env, _ = _make_vec(3)
    emb = np.zeros((3, N_SLOTS), dtype=np.float32)
    emb[1] = 0.9                 # all links longer -> heaviest
    emb[2, :N_SLOTS // 4] = 0.9  # only one leg longer (asymmetric, 3D)
    env.reset(embodiment=emb, seed=0)
    mass = wp.to_torch(env.gpu_model.body_mass).sum(dim=1)
    assert float(mass[1]) > float(mass[2]) > float(mass[0])


def test_seed_determinism():
    env, _ = _make_vec(4)
    o1, _, _ = env.reset(seed=123)
    o2, _, _ = env.reset(seed=123)
    assert torch.allclose(o1, o2)
