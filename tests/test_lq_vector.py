"""Tests for the vectorized LQ env and the central codesign_gym.make / make_vec API."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import codesign_gym as cg
from codesign_gym import SingleEnv
from envs.lq_structure.vector_env import LQStructureVectorEnv


LQ_IDS = ["LQStructure-LQR", "LQStructure-LQG", "LQStructure-QP"]


# --- central API -------------------------------------------------------------------

def test_registry_lists_all_envs():
    envs = cg.list_envs()
    assert set(LQ_IDS) <= set(envs)
    assert "BallCatcher-Pitch" in envs  # registered without importing its heavy deps


def test_unknown_env_id_raises():
    with pytest.raises(KeyError):
        cg.make("NoSuchEnv-v0")


def test_make_returns_squeezed_single_env():
    env, base = cg.make("LQStructure-LQR", n=4)
    assert isinstance(env, SingleEnv)
    obs, rejected, info = env.reset(seed=0)
    assert obs.shape == (16,)             # unbatched, no leading dim
    assert env.action_space.shape == (4,)
    # info is unwrapped to the single env (scalar/array, not per-env batched)
    assert rejected is False
    assert info["state"].shape == (8,)
    assert set(base) == {"masses", "stiffness", "damping", "actuators", "sensors"}


def test_make_step_is_unbatched():
    env, _ = cg.make("LQStructure-QP", n=4)
    env.reset(seed=0)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == (16,)
    assert isinstance(r, float)
    assert isinstance(term, bool) and isinstance(trunc, bool)
    # applied_control unwrapped to a single 1-D array
    assert np.all(np.abs(info["applied_control"]) <= env.cfg.u_max + 1e-9)


def test_make_wraps_make_vec_num_envs_1():
    e1, b1 = cg.make("LQStructure-LQR", n=4)
    e2, b2 = cg.make_vec("LQStructure-LQR", num_envs=1, n=4)
    o1, _, _ = e1.reset(seed=0)           # squeezed single env -> (16,)
    o2, _, _ = e2.reset(seed=0)           # vector env -> (1, 16)
    assert o1.shape == (16,)
    assert o2.shape == (1, 16)
    assert np.allclose(o1, o2[0])       # same underlying env, just squeezed
    assert set(b1) == set(b2)


def test_make_validate_embodiment_unbatched():
    env, base = cg.make("LQStructure-LQR", n=4)
    assert env.validate_embodiment(base) is True
    bad = {k: v.copy() for k, v in base.items()}
    bad["actuators"] = np.zeros(4, dtype=np.int8)
    assert env.validate_embodiment(bad) is False


def test_make_returns_base_embodiment_for_all_ids():
    for eid in LQ_IDS:
        _, base = cg.make(eid, n=4)
        assert set(base) == {"masses", "stiffness", "damping", "actuators", "sensors"}


# --- vector behavior ---------------------------------------------------------------

@pytest.mark.parametrize("eid", LQ_IDS)
def test_reset_step_shapes(eid):
    venv, _ = cg.make_vec(eid, num_envs=4, n=5, horizon=6)
    obs, rejected, info = venv.reset(seed=0)
    assert obs.shape == (4, venv.single_observation_space.shape[0])
    assert rejected.shape == (4,)
    assert rejected.dtype == bool
    obs, r, term, trunc, info = venv.step(venv.action_space.sample())
    assert obs.shape == (4, venv.single_observation_space.shape[0])
    assert r.shape == (4,) and term.shape == (4,) and trunc.shape == (4,)


def test_batched_spaces():
    venv, _ = cg.make_vec("LQStructure-LQR", num_envs=3, n=6)
    assert venv.observation_space.shape == (3, 24)
    assert venv.action_space.shape == (3, 6)


def test_seed_determinism():
    venv, _ = cg.make_vec("LQStructure-LQG", num_envs=3, n=4)
    o1, _, _ = venv.reset(seed=7)
    r1 = [venv.step(np.zeros(venv.action_space.shape))[1] for _ in range(4)]
    o2, _, _ = venv.reset(seed=7)
    r2 = [venv.step(np.zeros(venv.action_space.shape))[1] for _ in range(4)]
    assert np.allclose(o1, o2) and np.allclose(r1, r2)


def test_per_env_seeds_differ():
    venv, _ = cg.make_vec("LQStructure-LQR", num_envs=2, n=4)
    obs, _, _ = venv.reset(seed=0)
    assert not np.allclose(obs[0], obs[1])


def test_stacked_dict_embodiment():
    venv, base = cg.make_vec("LQStructure-LQR", num_envs=3, n=4)
    stacked = {k: np.stack([base[k]] * 3) for k in base}
    stacked["masses"][1] = np.full(4, 1.7)
    obs, rejected, info = venv.reset(embodiment=stacked)
    assert not rejected.any()
    assert not np.allclose(info["plant"][0]["Ad"], info["plant"][1]["Ad"])


def test_list_embodiment_with_none_entries():
    venv, base = cg.make_vec("LQStructure-LQR", num_envs=3, n=4)
    one = {k: v.copy() for k, v in base.items()}
    one["masses"] = np.full(4, 1.9)
    obs, rejected, info = venv.reset(embodiment=[None, one, None])
    assert not rejected.any()
    assert not np.allclose(info["plant"][0]["Ad"], info["plant"][1]["Ad"])


def test_per_env_rejection_array():
    venv, base = cg.make_vec("LQStructure-LQR", num_envs=3, n=4)
    bad = {k: v.copy() for k, v in base.items()}
    bad["masses"] = bad["masses"] + 100.0
    obs, rejected, info = venv.reset(embodiment=[None, bad, None])
    assert rejected.tolist() == [False, True, False]


def test_validate_embodiment_batch():
    venv, base = cg.make_vec("LQStructure-QP", num_envs=3, n=4)
    over = {k: v.copy() for k, v in base.items()}
    over["actuators"] = np.ones(4, dtype=np.int8)
    assert venv.validate_embodiment([None, over, base]).tolist() == [True, False, True]


def test_autoreset_next_step_semantics():
    venv, _ = cg.make_vec("LQStructure-LQR", num_envs=2, n=4, horizon=3)
    venv.reset(seed=0)
    truncs, rewards = [], []
    for _ in range(4):
        _, r, _, trunc, _ = venv.step(np.zeros(venv.action_space.shape))
        truncs.append(trunc.copy())
        rewards.append(r.copy())
    assert truncs[2].all()
    assert not truncs[3].any()
    assert np.allclose(rewards[3], 0.0)


def test_transition_level_change_in_step():
    venv, base = cg.make_vec("LQStructure-LQR", num_envs=2, n=4)
    _, _, info0 = venv.reset(seed=0)
    Ad_before = info0["plant"][0]["Ad"].copy()
    new = {k: v.copy() for k, v in base.items()}
    new["masses"] = np.full(4, 1.6)
    _, _, _, _, info1 = venv.step(np.zeros(venv.action_space.shape), embodiment=[new, None])
    assert not np.allclose(Ad_before, info1["plant"][0]["Ad"])


def test_step_raises_on_invalid_transition_embodiment():
    venv, base = cg.make_vec("LQStructure-LQR", num_envs=2, n=4)
    venv.reset(seed=0)
    bad = {k: v.copy() for k, v in base.items()}
    bad["actuators"] = np.zeros(4, dtype=np.int8)
    with pytest.raises(ValueError):
        venv.step(np.zeros(venv.action_space.shape), embodiment=[bad, None])


def test_eval_env_freezes_embodiment_with_warning():
    venv, base = cg.make_vec("LQStructure-LQR", num_envs=2, n=4, is_eval_env=True)
    _, _, info0 = venv.reset(seed=0)
    Ad_before = info0["plant"][0]["Ad"].copy()
    new = {k: v.copy() for k, v in base.items()}
    new["masses"] = np.full(4, 1.6)
    with pytest.warns(UserWarning, match="not permitted"):
        _, _, _, _, info1 = venv.step(np.zeros(venv.action_space.shape), embodiment=[new, None])
    assert np.allclose(Ad_before, info1["plant"][0]["Ad"])


def test_bad_batch_size_raises():
    venv, base = cg.make_vec("LQStructure-LQR", num_envs=3, n=4)
    with pytest.raises(ValueError):
        venv.reset(embodiment=[None, None])


def test_vector_info_exposes_state_control_cost():
    venv, _ = cg.make_vec("LQStructure-QP", num_envs=3, n=4)
    venv.reset(seed=0)
    _, r, _, _, info = venv.step(np.full(venv.action_space.shape, 50.0))
    assert info["state"].shape == (3, 8)
    assert info["stage_cost"].shape == (3,)
    assert len(info["applied_control"]) == 3
    for u in info["applied_control"]:
        assert np.all(np.abs(u) <= venv.cfg.u_max + 1e-9)


def test_vector_render_returns_frame_tuple():
    venv, _ = cg.make_vec("LQStructure-LQR", num_envs=3, n=4, render_mode="rgb_array")
    venv.reset(seed=0)
    frames = venv.render()
    assert isinstance(frames, tuple) and len(frames) == 3
    for f in frames:
        assert f.ndim == 3 and f.shape[2] == 3
    venv.close()


@pytest.mark.parametrize("is_eval_env", [False, True])
def test_autoreset_keeps_each_lanes_design(is_eval_env):
    """A lane that autoresets keeps the design it was reset with (an eval env cannot
    re-supply it at step time), instead of reverting to the base design."""
    env, base = cg.make_vec("LQStructure-LQR", num_envs=2, n=4, horizon=2, is_eval_env=is_eval_env)
    design = {k: v.copy() for k, v in base.items()}
    design["masses"] = np.full_like(base["masses"], 1.9)
    env.reset(seed=0, embodiment=[design, None])
    a = np.zeros(env.action_space.shape)
    for _ in range(7):                      # three episodes of horizon 2, autoresets between them
        _, _, _, _, info = env.step(a)
        np.testing.assert_allclose(info["embodiment"][0]["masses"], 1.9)
        np.testing.assert_allclose(info["embodiment"][1]["masses"], base["masses"])


def test_partial_reset_touches_only_the_masked_lanes():
    """reset(options={"reset_mask": m}) resets only the masked lanes; the others keep
    their obs and any pending autoreset, and a full reset must come first."""
    env, _ = cg.make_vec("LQStructure-LQR", num_envs=2, n=4)
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
