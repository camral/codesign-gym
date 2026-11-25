"""Tests for the vectorized mobile-env co-design env and the central codesign_gym API."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mobile_env")

import codesign_gym as cg
from codesign_gym import SingleEnv
from envs.mobile_env import embodiment as E
from envs.mobile_env.config import make_config
from envs.mobile_env.vector_env import MobileEnvVectorEnv

MOBILE_IDS = ["Network-Urban", "Network-Suburban", "Network-Motorway"]


def _design(cfg, positions, tx=None):
    tx = cfg.seed_tx if tx is None else tx
    stations = np.zeros((cfg.max_bs, 3), dtype=np.float32)
    stations[:, 2] = cfg.tx_min
    for i, (x, y) in enumerate(positions):
        stations[i] = (x, y, tx)
    return {"num_bs": len(positions), "stations": stations}


# --- central registry API ---------------------------------------------------------

def test_registry_lists_all_mobile_envs():
    envs = cg.list_envs()
    assert set(MOBILE_IDS) <= set(envs)


def test_all_ids_use_one_builder_module():
    for env_id in MOBILE_IDS:
        path, _ = cg.REGISTRY[env_id]
        assert path == "envs.mobile_env.registration:build_vec"


@pytest.mark.parametrize("env_id", MOBILE_IDS)
def test_make_vec_returns_env_and_base(env_id):
    env, base = cg.make_vec(env_id, num_envs=3)
    assert isinstance(env, MobileEnvVectorEnv)
    assert set(base) == {"num_bs", "stations"}
    assert np.array_equal(base["stations"], env.base_embodiment["stations"])
    env.close()


@pytest.mark.parametrize("env_id", MOBILE_IDS)
def test_make_returns_squeezed_single_env(env_id):
    env, base = cg.make(env_id)
    assert isinstance(env, SingleEnv)
    obs, rejected, info = env.reset(seed=0)
    assert rejected is False
    assert env.observation_space.contains(obs)
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    assert np.isscalar(r) or np.ndim(r) == 0
    assert env.observation_space.contains(obs)
    env.close()


# --- vectorized reset / step ------------------------------------------------------

def test_reset_returns_obs_rejected_info():
    env, _ = cg.make_vec("Network-Urban", num_envs=4)
    obs, rejected, info = env.reset(seed=0)
    assert obs.shape[0] == 4
    assert rejected.shape == (4,) and rejected.dtype == bool
    assert not rejected.any()
    assert info["action_mask"].shape[0] == 4
    env.close()


def test_step_returns_batched_five_tuple():
    env, _ = cg.make_vec("Network-Urban", num_envs=4)
    env.reset(seed=0)
    actions = env.action_space.sample()
    obs, r, term, trunc, info = env.step(actions)
    assert obs.shape[0] == 4
    assert r.shape == (4,)
    assert term.shape == (4,) and trunc.shape == (4,)
    assert np.all(np.isfinite(r))
    env.close()


def test_reset_with_designs_reports_rejection_per_env():
    env, base = cg.make_vec("Network-Urban", num_envs=3)
    cfg = make_config("dense_urban")
    good = _design(cfg, [(10, 10), (20, 20), (30, 30), (40, 40)])
    bad = _design(cfg, [(10, 10)])  # below min_bs
    obs, rejected, info = env.reset(embodiment=[good, None, bad])
    assert list(rejected) == [False, False, True]
    env.close()


def test_validate_embodiment_batch():
    env, _ = cg.make_vec("Network-Urban", num_envs=2)
    cfg = make_config("dense_urban")
    good = _design(cfg, [(10, 10), (20, 20), (30, 30)])
    bad = _design(cfg, [(10, 10)])
    out = env.validate_embodiment([good, bad])
    assert list(out) == [True, False]
    env.close()


def test_step_embodiment_ignored_with_warning():
    env, base = cg.make_vec("Network-Urban", num_envs=1)
    env.reset(seed=0)
    cfg = make_config("dense_urban")
    other = _design(cfg, [(11, 11), (22, 22), (33, 33), (44, 44), (55, 55)])
    with pytest.warns(UserWarning):
        env.step(env.action_space.sample(), embodiment=[other])
    env.close()


def test_vector_render_returns_frame_tuple():
    env, _ = cg.make_vec("Network-Urban", num_envs=2, render_mode="rgb_array")
    env.reset(seed=0)
    env.step(env.action_space.sample())
    frames = env.render()
    assert isinstance(frames, tuple) and len(frames) == 2
    for f in frames:
        assert f.ndim == 3 and f.shape[2] == 3
    env.close()


def test_autoreset_after_truncation():
    # Run past the horizon; finished sub-envs autoreset on the next step.
    env, _ = cg.make_vec("Network-Urban", num_envs=2)
    env.reset(seed=0)
    ep = make_config("dense_urban").ep_max_time
    saw_trunc = False
    for _ in range(ep + 3):
        actions = env.action_space.sample()
        obs, r, term, trunc, info = env.step(actions)
        if trunc.any():
            saw_trunc = True
        assert np.all(np.isfinite(r))
    assert saw_trunc
    env.close()


def test_partial_reset_touches_only_the_masked_lanes():
    """reset(options={"reset_mask": m}) resets only the masked lanes; the others keep
    their obs and any pending autoreset, and a full reset must come first."""
    env, _ = cg.make_vec("Network-Urban", num_envs=2)
    with pytest.raises(ValueError, match="full reset"):
        env.reset(options={"reset_mask": np.array([True, False])})
    env.reset(seed=0)
    for _ in range(3):
        obs, *_ = env.step(env.action_space.sample())
    kept = np.asarray(obs[1]).copy()
    env._autoreset[1] = True                       # as a terminal step leaves it
    obs, rejected, _ = env.reset(seed=5, options={"reset_mask": np.array([True, False])})
    np.testing.assert_array_equal(np.asarray(obs[1]), kept)
    assert not np.asarray(rejected).any() and env._autoreset.tolist() == [False, True]
    env.close()


def test_unseeded_lanes_and_episodes_differ_but_seeded_ones_replay():
    """Without a seed each lane builds its own scenario and every episode draws a new
    one; reset(seed=s) still replays exactly."""
    env, _ = cg.make_vec("Network-Urban", num_envs=3)
    o, _, _ = env.reset()
    assert not np.array_equal(o[0], o[1]) and not np.array_equal(o[1], o[2])
    o2, _, _ = env.reset()
    assert not np.array_equal(o, o2)
    a, _, _ = env.reset(seed=4)
    b, _, _ = env.reset(seed=4)
    np.testing.assert_array_equal(a, b)
    env.close()
