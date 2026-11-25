"""Tests for the F1TENTH vehicle setup co-design environment."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("f1tenth_gym_jax")

import codesign_gym as cg
from codesign_gym import SingleEnv
from envs.f1tenth import embodiment as E
from envs.f1tenth.config import make_config
from envs.f1tenth.vector_env import F1TenthVectorEnv

IDS = ["Racing-Spielberg", "Racing-Levine"]


# --- embodiment + couplings -------------------------------------------------------

def test_registry_lists_ids():
    assert set(IDS) <= set(cg.list_envs())


def test_space_and_base_valid():
    cfg = make_config("spielberg")
    base = E.base_embodiment(cfg)
    assert base.shape == (5,)
    assert E.validate_embodiment(base, cfg)
    assert E.make_embodiment_space(cfg).contains(base)


def test_base_design_recovers_stock_mass():
    cfg = make_config("spielberg")
    dvec = E.design_to_param_vector(E.base_embodiment(cfg), cfg)[0]
    m, mu_eff, I, lf, lr, a_max, csf, csr = dvec
    assert m == pytest.approx(cfg.m_base, abs=1e-4)          # base spec -> base mass
    assert mu_eff == pytest.approx(cfg.mu_base, abs=1e-4)    # no load shift at base
    assert (lf + lr) == pytest.approx(cfg.wheelbase, abs=1e-4)


def test_couplings_have_tradeoffs():
    cfg = make_config("spielberg")
    base = E.base_embodiment(cfg)
    # max grip + max power design
    hot = base.copy(); hot[0] = cfg.mu_max; hot[1] = cfg.csf_max; hot[2] = cfg.csr_max; hot[3] = cfg.fdrive_max
    b = E.design_to_param_vector(base, cfg)[0]
    h = E.design_to_param_vector(hot, cfg)[0]
    assert h[0] > b[0]                       # upgrades cost mass
    assert h[1] < cfg.mu_max                 # tyre load-sensitivity: effective mu < mu0
    assert h[2] > b[2]                        # heavier -> larger yaw inertia
    # a_max = F/m: extra mass eats into the accel gain from extra force
    assert h[5] < cfg.fdrive_max / cfg.m_base


def test_validate_structural_only():
    cfg = make_config("spielberg")
    base = E.base_embodiment(cfg)
    assert E.validate_embodiment(base, cfg)
    oob = base.copy(); oob[0] = 99.0
    assert not E.validate_embodiment(oob, cfg)
    assert not E.validate_embodiment(base[:-1], cfg)


# --- live env (builds a Showdown-free JAX sim + loads a track) --------------------

def test_make_vec_and_rollout():
    env, base = cg.make_vec("Racing-Spielberg", num_envs=4)
    assert isinstance(env, F1TenthVectorEnv)
    assert np.array_equal(base, env.base_embodiment)
    obs, rejected, info = env.reset(seed=0)
    assert obs.shape[0] == 4 and not rejected.any()
    for _ in range(5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs.astype(np.float32))
        assert r.shape == (4,) and np.all(np.isfinite(r))
    env.close()


def test_make_single_env():
    env, base = cg.make("Racing-Spielberg")
    assert isinstance(env, SingleEnv)
    obs, rejected, info = env.reset(seed=0)
    assert rejected is False
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert np.ndim(r) == 0 and np.isfinite(r)
    env.close()


def test_levine_uses_packaged_map_and_restores_lookup_env(monkeypatch):
    monkeypatch.setenv("F1TENTH_GYM_JAX_MAP_DIR", "/tmp/original-map-dir")
    env, _ = cg.make_vec("Racing-Levine", num_envs=2)
    obs, rejected, _ = env.reset(seed=0)
    assert obs.shape[0] == 2 and not rejected.any()
    assert Path(env._env.track.filepath).name == "levine.png"
    assert "envs/f1tenth/assets/maps/levine" in str(env._env.track.filepath)
    assert os.environ["F1TENTH_GYM_JAX_MAP_DIR"] == "/tmp/original-map-dir"
    env.close()


def test_heterogeneous_design_batch():
    env, _ = cg.make_vec("Racing-Spielberg", num_envs=3)
    cfg = make_config("spielberg")
    designs = np.stack([
        E.base_embodiment(cfg),
        np.array([cfg.mu_max, cfg.csf_max, cfg.csr_max, cfg.fdrive_max, 0.55], np.float32),
        np.array([cfg.mu_min, cfg.csf_min, cfg.csr_min, cfg.fdrive_min, 0.47], np.float32),
    ])
    obs, rejected, info = env.reset(embodiment=designs)
    assert not rejected.any()
    a = np.tile(np.array([0.0, 3.0], np.float32), (3, 1))
    for _ in range(5):
        obs, r, term, trunc, info = env.step(a)
    assert np.all(np.isfinite(r))
    env.close()


def test_transition_level_change_supported():
    env, base = cg.make_vec("Racing-Spielberg", num_envs=2)
    env.reset(seed=0)
    cfg = make_config("spielberg")
    other = np.array([cfg.mu_min, cfg.csf_min, cfg.csr_min, cfg.fdrive_min, 0.47], np.float32)
    # supports_transition_embodiment=True: a mid-episode design swap is applied silently
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obs, r, term, trunc, info = env.step(env.action_space.sample(), embodiment=other)
    assert np.all(np.isfinite(r))
    env.close()


def test_vector_render_returns_frame_tuple():
    env, _ = cg.make_vec("Racing-Spielberg", num_envs=2, render_mode="rgb_array")
    env.reset(seed=0)
    env.step(env.action_space.sample())
    frames = env.render()
    assert isinstance(frames, tuple) and len(frames) == 2
    for f in frames:
        assert f.ndim == 3 and f.shape[2] == 3 and f.dtype == np.uint8
    env.close()


def test_partial_reset_preserves_unselected_worlds():
    import jax

    env, base = cg.make_vec("Racing-Spielberg", num_envs=4, render_mode="rgb_array")
    try:
        obs, rejected, _ = env.reset(seed=0)
        assert env.metadata["supports_partial_reset"] and not rejected.any()
        action = np.tile(np.array([0.0, 0.0], np.float32), (4, 1))
        for _ in range(2):
            obs, _, _, _, _ = env.step(action)
        old_states = [np.asarray(leaf).copy() for leaf in jax.tree_util.tree_leaves(env._states)]
        old_designs, old_dvecs = env._designs.copy(), np.asarray(env._dvecs).copy()
        old_auto, old_trail, old_start = env._autoreset.copy(), [list(t) for t in env._trail], env._start_s.copy()
        mask = np.array([True, False, True, False])
        designs = np.tile(base, (4, 1))
        designs[0, 0] = env.cfg.mu_max
        designs[2, 3] = env.cfg.fdrive_max
        fresh, rejected, info = env.reset(options={"reset_mask": mask}, embodiment=designs)
        assert not rejected.any() and np.array_equal(fresh[~mask], obs[~mask])
        assert np.array_equal(info["design"][mask], designs[mask])
        assert np.array_equal(env._designs[~mask], old_designs[~mask])
        assert np.array_equal(np.asarray(env._dvecs)[~mask], old_dvecs[~mask])
        assert np.array_equal(env._autoreset[~mask], old_auto[~mask])
        assert all(env._trail[i] == old_trail[i] and env._start_s[i] == old_start[i] for i in np.flatnonzero(~mask))
        assert all(len(env._trail[i]) == 1 for i in np.flatnonzero(mask))
        assert np.array_equal(info["step_count"][mask], np.zeros(2, np.int32))
        for before, after in zip(old_states, jax.tree_util.tree_leaves(env._states)):
            assert np.array_equal(np.asarray(after)[~mask], before[~mask])
        with pytest.raises(ValueError, match="reset_mask"):
            env.reset(options={"reset_mask": np.array([1, 0, 1, 0])})
        no_op, rejected, _ = env.reset(options={"reset_mask": np.zeros(4, bool)})
        assert np.array_equal(no_op, fresh) and not rejected.any()
        env._autoreset[0] = True
        old_design = env._designs.copy()
        same_design, rejected, _ = env.reset(options={"reset_mask": np.array([True, False, False, False])})
        assert not rejected.any() and not env._autoreset[0]
        assert np.array_equal(env._designs[0], base) and np.array_equal(env._designs[1:], old_design[1:])   # None -> base
        assert np.array_equal(same_design[1:], no_op[1:])
        obs, reward, term, trunc, _ = env.step(action)
        assert obs.shape == fresh.shape and reward.shape == term.shape == trunc.shape == (4,)
    finally:
        env.close()


def test_partial_reset_rejection_only_applies_to_selected_worlds():
    env, base = cg.make_vec("Racing-Spielberg", num_envs=3)
    try:
        env.reset(seed=1)
        mask = np.array([True, False, False])
        designs = np.tile(base, (3, 1))
        designs[0, 0] = 99.0
        designs[1, 0] = 99.0
        _, rejected, info = env.reset(options={"reset_mask": mask}, embodiment=designs)
        assert np.array_equal(rejected, mask)
        assert np.array_equal(info["design"], np.tile(base, (3, 1)))
    finally:
        env.close()


def test_determinism_with_seed():
    env, _ = cg.make_vec("Racing-Spielberg", num_envs=2)
    o1, _, _ = env.reset(seed=7)
    a = np.tile(np.array([0.1, 2.0], np.float32), (2, 1))
    r1 = [env.step(a)[1] for _ in range(4)]
    o2, _, _ = env.reset(seed=7)
    r2 = [env.step(a)[1] for _ in range(4)]
    assert np.allclose(o1, o2) and np.allclose(r1, r2)
    env.close()


def test_eval_episode_limit_exceeds_training_limit():
    train, _ = cg.make_vec("Racing-Spielberg", num_envs=2, time_limit=2)
    evaluation, _ = cg.make_vec("Racing-Spielberg", num_envs=2, time_limit=2, is_eval_env=True)
    try:
        assert train._base.max_steps == 2 and evaluation._base.max_steps == 10000
        train.reset(seed=9)
        evaluation.reset(seed=9)
        action = np.zeros((2, 2), np.float32)
        for _ in range(2):
            _, _, train_term, train_trunc, _ = train.step(action)
            _, _, eval_term, eval_trunc, _ = evaluation.step(action)
        assert not train_term.any() and train_trunc.all()
        assert not eval_term.any() and not eval_trunc.any()
        for _ in range(6):
            _, _, eval_term, eval_trunc, _ = evaluation.step(action)
        assert not eval_term.any() and not eval_trunc.any()
    finally:
        train.close()
        evaluation.close()


def test_eval_limits_for_both_racing_presets():
    for env_id, limit in (("Racing-Spielberg", 10000), ("Racing-Levine", 10000)):
        env, _ = cg.make_vec(env_id, num_envs=1, is_eval_env=True)
        try:
            assert env._base.max_steps == limit
            assert env._base.max_num_laps == 1
        finally:
            env.close()


def test_lap_success_distinguishes_collision_and_step_limit(monkeypatch):
    import jax.numpy as jnp

    env, _ = cg.make_vec("Racing-Spielberg", num_envs=3)
    try:
        env.reset(seed=3)
        states = env._states.replace(num_laps=jnp.asarray([[1], [1], [0]], dtype=env._states.num_laps.dtype))
        obs = jnp.asarray(env._last_obs)
        def fake_advance(*args):
            return states, obs, jnp.zeros(3, jnp.float32), jnp.array([False, True, False]), jnp.array([True, False, True]), jnp.zeros(3, bool)
        monkeypatch.setattr(env, "_advance", fake_advance)
        _, _, terminated, truncated, info = env.step(np.zeros((3, 2), np.float32))
        assert np.array_equal(terminated, [False, True, False])
        assert np.array_equal(truncated, [True, False, True])
        assert info["success"].shape == (3,) and info["success"].dtype == np.bool_
        assert np.array_equal(info["success"], [True, False, False])
        assert np.array_equal(env.last_success, info["success"])
        env.reset(options={"reset_mask": np.array([False, True, True])})
        assert np.array_equal(env.last_success, [True, False, False])
        env.reset()
        assert not env.last_success.any()
    finally:
        env.close()


def test_nonfinite_world_terminates_without_poisoning_batch():
    env, _ = cg.make_vec("Racing-Levine", num_envs=3, is_eval_env=True)
    try:
        initial, rejected, _ = env.reset(seed=4)
        assert not rejected.any() and np.isfinite(initial).all()
        actions = np.zeros((3, 2), np.float32)
        actions[1, 0] = np.nan
        obs, reward, terminated, truncated, info = env.step(actions)
        assert np.isfinite(obs).all() and np.isfinite(reward).all()
        assert np.array_equal(terminated, [False, True, False])
        assert not truncated.any()
        assert np.array_equal(env.last_diverged, [False, True, False])
        assert np.array_equal(info["diverged"], env.last_diverged)
        assert np.array_equal(obs[1], np.zeros_like(obs[1])) and reward[1] == 0.0
        obs, reward, terminated, truncated, info = env.step(np.zeros((3, 2), np.float32))
        assert np.isfinite(obs).all() and np.isfinite(reward).all()
        assert not env.last_diverged.any() and not terminated.any() and not truncated.any()
    finally:
        env.close()


def test_step_embodiment_none_keeps_and_invalid_raises():
    """A None lane keeps its design; any invalid lane raises before anything changes."""
    env, base = cg.make_vec("Racing-Spielberg", num_envs=2)
    space = env.embodiment_space
    design = np.asarray(space.low + 0.5 * (space.high - space.low), dtype=np.float32)
    env.reset(seed=0, embodiment=np.stack([design, design]))
    a = np.zeros(env.action_space.shape, np.float32)
    env.step(a, embodiment=[None, design])
    np.testing.assert_allclose(env._designs[0], design)
    before = env._designs.copy()
    with pytest.raises(ValueError, match=r"lanes \[1\]"):
        env.step(a, embodiment=[base, np.asarray(space.high, np.float32) + 10.0])
    np.testing.assert_array_equal(env._designs, before)
    env.close()


def test_action_box_covers_the_strongest_design():
    """The longitudinal action bound is the largest a_max in the design space (the
    dynamics clip each car to its own), so drive-power upgrades are usable."""
    env, _ = cg.make_vec("Racing-Spielberg", num_envs=1)
    sp = env.embodiment_space
    rows = np.random.default_rng(0).uniform(sp.low, sp.high, size=(20000, 5)).astype(np.float32)
    a_max = E.design_to_param_vector(rows, env.cfg)[:, 5]
    bound = float(env.single_action_space.high[1])
    assert bound == pytest.approx(E.max_accel(env.cfg)) and env.single_action_space.low[1] == -bound
    assert a_max.max() <= bound + 1e-5 and bound > float(E.design_to_param_vector(env.base_embodiment[None], env.cfg)[0, 5])
    env.close()


def test_step_refuses_a_rejected_lane_until_reset():
    env, base = cg.make_vec("Racing-Levine", 2)
    bad = np.asarray(base, np.float32) * 100.0
    _, rejected, _ = env.reset(seed=0, embodiment=[base, bad])
    assert rejected.tolist() == [False, True]
    with pytest.raises(RuntimeError, match="rejected"):
        env.step(np.zeros((2, 2), np.float32))
    env.reset(seed=0, embodiment=[base, base])
    env.step(np.zeros((2, 2), np.float32))
    env.close()


def test_none_at_partial_reset_is_the_base_design_and_design_is_in_step_info():
    env, base = cg.make_vec("Racing-Levine", 2)
    other = np.asarray(base, np.float32).copy()
    other[0] = float(E.make_embodiment_space(make_config("levine")).high[0])
    env.reset(seed=0, embodiment=[other, other])
    _, _, _, _, info = env.step(np.zeros((2, 2), np.float32))
    np.testing.assert_allclose(info["design"], [other, other])
    env.reset(options={"reset_mask": np.array([True, False])})              # None -> base for lane 0
    _, _, _, _, info = env.step(np.zeros((2, 2), np.float32))
    np.testing.assert_allclose(info["design"], [base, other])
    env.close()


def test_render_one_lane():
    env, _ = cg.make_vec("Racing-Levine", 2, render_mode="rgb_array")
    env.reset(seed=0)
    frames, one = env.render(), env.render(env_idx=1)
    assert len(frames) == 2 and one.shape == frames[1].shape
    with pytest.raises(IndexError):
        env.render(env_idx=2)
    env.close()


def test_base_car_has_the_stock_yaw_inertia():
    cfg = make_config("levine")
    params = E.design_to_param_vector(np.asarray(E.base_embodiment(cfg), np.float32)[None], cfg)[0]
    assert params[2] == pytest.approx(0.04712, rel=1e-3)          # [m, mu_eff, I, lf, lr, a_max, C_Sf, C_Sr]
