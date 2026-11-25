"""Tests for the vectorized RobotWarehouse env and the central codesign_gym API."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jumanji")

import codesign_gym as cg
from codesign_gym import SingleEnv
from envs.robot_warehouse import embodiment as E
from envs.robot_warehouse.config import make_config
from envs.robot_warehouse.generator import CoDesignWarehouseGenerator
from envs.robot_warehouse.vector_env import RobotWarehouseVectorEnv
from jumanji.environments.routing.robot_warehouse.constants import _SHELVES

RW_IDS = ["Warehouse-Small", "Warehouse-Medium", "Warehouse-Congested"]


def _base(setup):
    cfg = make_config(setup)
    return E.base_embodiment(cfg, CoDesignWarehouseGenerator(cfg))


# --- central registry API ---------------------------------------------------------

def test_registry_lists_all_rw_envs():
    assert set(RW_IDS) <= set(cg.list_envs())


def test_all_ids_use_one_builder_module():
    for env_id in RW_IDS:
        path, _ = cg.REGISTRY[env_id]
        assert path == "envs.robot_warehouse.registration:build_vec"


@pytest.mark.parametrize("env_id", RW_IDS)
def test_make_vec_returns_env_and_base(env_id):
    env, base = cg.make_vec(env_id, num_envs=3)
    assert isinstance(env, RobotWarehouseVectorEnv)
    assert np.array_equal(base, env.base_embodiment)
    env.close()


@pytest.mark.parametrize("env_id", RW_IDS)
def test_make_returns_squeezed_single_env(env_id):
    env, base = cg.make(env_id)
    assert isinstance(env, SingleEnv)
    obs, rejected, info = env.reset(seed=0)
    assert rejected is False
    assert env.observation_space.contains(obs)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert np.ndim(r) == 0 and np.isfinite(r)
    assert env.observation_space.contains(obs)
    env.close()


# --- vectorized reset / step ------------------------------------------------------

def test_reset_returns_obs_rejected_info():
    env, _ = cg.make_vec("Warehouse-Small", num_envs=4)
    obs, rejected, info = env.reset(seed=0)
    assert obs.shape[0] == 4
    assert rejected.shape == (4,) and rejected.dtype == bool and not rejected.any()
    assert info["action_mask"].shape == (4, env.num_agents, 5)
    env.close()


def test_step_returns_batched_five_tuple_and_shapes_design_invariant():
    env, _ = cg.make_vec("Warehouse-Small", num_envs=4)
    obs0, _, _ = env.reset(seed=0)
    actions = env.action_space.sample()
    obs, r, term, trunc, info = env.step(actions)
    assert obs.shape == obs0.shape          # obs size independent of the design
    assert r.shape == (4,) and term.shape == (4,) and trunc.shape == (4,)
    assert np.all(np.isfinite(r))
    env.close()


def test_heterogeneous_design_batch():
    # A batch of *different* occupancy vectors resets in one call (the whole point of
    # the fixed-grid design: homogeneous shapes -> vmap over heterogeneous placements).
    env, _ = cg.make_vec("Warehouse-Small", num_envs=3)
    cfg = make_config("small")
    n, K = E.n_eligible(cfg), cfg.num_shelves
    designs = np.zeros((3, n), dtype=np.int8)
    for i in range(3):
        designs[i, i:i + K] = 1               # K shelves, shifted -> distinct valid layouts
    obs, rejected, info = env.reset(embodiment=designs)
    assert not rejected.any()
    # different layouts generally give different observations
    assert not np.allclose(obs[0], obs[1])
    env.close()


def test_variable_count_above_floor_is_accepted():
    env, _ = cg.make_vec("Warehouse-Small", num_envs=2)
    cfg = make_config("small")
    K = cfg.num_shelves
    denser = _base("small").copy(); denser[np.flatnonzero(denser == 0)[:20]] = 1   # K+20 shelves
    obs, rejected, info = env.reset(embodiment=[_base("small"), denser])
    assert not rejected.any()
    # the denser design actually places more shelves on the grid
    g = np.asarray(env._states.grid)          # (N, 2, H, W); layer _SHELVES holds shelves
    counts = (g[:, _SHELVES] > 0).reshape(2, -1).sum(1)
    assert counts[1] == K + 20 and counts[0] == K
    env.close()


def test_agents_never_spawn_on_shelves():
    env, _ = cg.make_vec("Warehouse-Small", num_envs=8)
    env.reset(seed=1, embodiment=np.stack([_base("small")] * 8))
    g = np.asarray(env._states.grid)          # layer 0 shelves, layer 1 agents
    overlap = ((g[:, 0] > 0) & (g[:, 1] > 0)).sum()
    assert overlap == 0
    env.close()


def test_replenishment_never_requests_inactive_shelf():
    # The masked replenishment must only ever request active (on-grid) shelves,
    # otherwise inactive padded slots poison the queue and reward stalls.
    env, _ = cg.make_vec("Warehouse-Small", num_envs=6)
    env.reset(seed=0, embodiment=np.stack([_base("small")] * 6))
    for _ in range(80):
        env.step(np.stack([env.single_action_space.sample() for _ in range(6)]))
    rq = np.asarray(env._states.request_queue)
    grid = np.asarray(env._states.grid[:, _SHELVES])
    for e in range(6):
        on_grid = set((grid[e][grid[e] > 0] - 1).tolist())
        assert set(rq[e].tolist()).issubset(on_grid)
    env.close()


def test_reset_with_designs_reports_rejection_per_env():
    env, base = cg.make_vec("Warehouse-Small", num_envs=3)
    good = _base("small")
    bad = good.copy(); bad[np.flatnonzero(bad)[0]] = 0     # below the floor -> invalid
    obs, rejected, info = env.reset(embodiment=[good, None, bad])
    assert list(rejected) == [False, False, True]
    env.close()


def test_validate_embodiment_batch():
    env, _ = cg.make_vec("Warehouse-Small", num_envs=2)
    good = _base("small")
    bad = good.copy(); bad[np.flatnonzero(bad)[0]] = 0     # below the floor
    assert list(env.validate_embodiment([good, bad])) == [True, False]
    env.close()


def test_step_embodiment_ignored_with_warning():
    env, _ = cg.make_vec("Warehouse-Small", num_envs=1)
    env.reset(seed=0)
    with pytest.warns(UserWarning):
        env.step(env.action_space.sample(), embodiment=_base("small"))
    env.close()


def test_determinism_with_seed():
    env, _ = cg.make_vec("Warehouse-Small", num_envs=2)
    o1, _, _ = env.reset(seed=7)
    a = env.action_space.sample()
    r1 = [env.step(a)[1] for _ in range(4)]
    o2, _, _ = env.reset(seed=7)
    r2 = [env.step(a)[1] for _ in range(4)]
    assert np.allclose(o1, o2) and np.allclose(r1, r2)
    env.close()


def test_vector_render_returns_frame_tuple():
    env, _ = cg.make_vec("Warehouse-Small", num_envs=2, render_mode="rgb_array")
    env.reset(seed=0)
    env.step(env.action_space.sample())
    frames = env.render()
    assert isinstance(frames, tuple) and len(frames) == 2
    for f in frames:
        assert f.ndim == 3 and f.shape[2] == 3
        assert f.shape[0] % 2 == f.shape[1] % 2 == 0
    env.close()


def test_autoreset_after_truncation():
    env, _ = cg.make_vec("Warehouse-Small", num_envs=2)
    env.reset(seed=0)
    tl = make_config("small").time_limit
    saw_done = False
    for _ in range(tl + 3):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        saw_done = saw_done or bool(term.any() or trunc.any())
        assert np.all(np.isfinite(r))
    assert saw_done
    env.close()


@pytest.mark.parametrize("is_eval_env", [False, True])
def test_partial_reset_preserves_unselected_worlds(is_eval_env):
    import jax

    env, base = cg.make_vec("Warehouse-Small", num_envs=3, is_eval_env=is_eval_env)
    try:
        env.reset(seed=7)
        before_obs, _, _, _, _ = env.step(np.zeros((3, env.num_agents), np.int32))
        before = jax.tree_util.tree_map(lambda x: np.asarray(x).copy(), env._states)
        env._returns[:] = [3, 4, 5]
        env._autoreset[:] = True
        denser = base.copy()
        denser[np.flatnonzero(denser == 0)[0]] = 1
        obs, rejected, info = env.reset(
            embodiment=[denser, np.zeros_like(base), np.zeros_like(base)],
            options={"reset_mask": [True, False, True]},
        )
        assert rejected.tolist() == [False, False, True]
        for old, new in zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(env._states)):
            np.testing.assert_array_equal(np.asarray(new)[1], old[1])
        np.testing.assert_array_equal(obs[1], before_obs[1])
        np.testing.assert_array_equal(env._design, np.stack([denser, base, base]))
        assert info["step_count"].tolist() == [0, 1, 0]
        assert env._returns.tolist() == [0, 4, 0]
        assert env._autoreset.tolist() == [False, True, False]
        np.testing.assert_array_equal(info["action_mask"], env._states.action_mask)
        key = np.asarray(env._key).copy()
        same, rejected, _ = env.reset(options={"reset_mask": [False] * 3})
        np.testing.assert_array_equal(same, obs)
        np.testing.assert_array_equal(env._key, key)
        assert not rejected.any()
    finally:
        env.close()


def test_partial_reset_requires_full_reset_and_correct_mask_shape():
    env, _ = cg.make_vec("Warehouse-Small", num_envs=2)
    try:
        with pytest.raises(ValueError, match="shape"):
            env.reset(options={"reset_mask": [True]})
        with pytest.raises(RuntimeError, match="earlier full reset"):
            env.reset(options={"reset_mask": [True, False]})
    finally:
        env.close()


def test_display_state_filters_padding_without_changing_simulation():
    import jax

    env, base = cg.make_vec("Warehouse-Small", num_envs=2)
    try:
        env.reset(seed=0)
        display = env._display_state(1)
        assert all(isinstance(x, np.ndarray) for x in jax.tree_util.tree_leaves(display))
        assert len(display.shelves.position.x) == int(base.sum())
        assert env._states.shelves.position.x.shape == (2, len(base))
        grid = np.asarray(display.grid[_SHELVES])
        assert np.all(grid[display.shelves.position.x, display.shelves.position.y] > 0)
    finally:
        env.close()


def test_step_refuses_a_rejected_lane_until_reset():
    env, base = cg.make_vec("Warehouse-Small", 2)
    base = np.asarray(base[0] if isinstance(base, list) else base)
    empty = np.zeros_like(base)                       # below the shelf minimum: impossible
    _, rejected, _ = env.reset(seed=0, embodiment=[base, empty])
    assert rejected.tolist() == [False, True]
    actions = np.zeros(env.action_space.shape, dtype=np.int32)
    with pytest.raises(RuntimeError, match="rejected"):
        env.step(actions)
    env.reset(seed=0, embodiment=[base, base])
    env.step(actions)
    env.close()
