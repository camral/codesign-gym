"""Tests for the vectorized microgrid env and the central codesign_gym API."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import codesign_gym as cg
from codesign_gym import SingleEnv
from envs.microgrid.env import MicrogridCodesignEnv
from envs.microgrid.registration import SETUP_CONFIGS

MICRO_IDS = [
    "Microgrid-Random",
    "Microgrid-Forecast",
    "Microgrid-Campus",
    "Microgrid-OffGridSingle",
    "Microgrid-OffGridCampus",
]
REMOVED_IDS = [
    "MicrogridSizing-Random-v0",
    "MicrogridSizing-Forecast-v0",
    "MicrogridSizing-Campus-v0",
    "MicrogridSizing-OffGrid-v0",
    "MicrogridSizing-OffGridCampus-v0",
    "MobileEnv-DenseUrban-v0",
    "MobileEnv-Suburban-v0",
    "MobileEnv-RuralHighway-v0",
    "RobotWarehouse-Small-v0",
    "RobotWarehouse-Medium-v0",
    "RobotWarehouse-Congested-v0",
    "F1Tenth-TimeTrial-Spielberg-v0",
    "F1Tenth-TimeTrial-Levine-v0",
]
EMB_KEYS = {"pv_capacity_kw", "battery_capacity_kwh", "genset_rating_kw"}


def single(setup="random", **kw):
    cfg = {**SETUP_CONFIGS[setup], **kw}
    env = MicrogridCodesignEnv(**cfg)
    return env, env.base_embodiment


# --- central API -------------------------------------------------------------------

def test_registry_lists_all_envs():
    envs = cg.list_envs()
    assert set(MICRO_IDS) <= set(envs)


def test_removed_preset_ids_are_not_registered():
    assert not set(REMOVED_IDS) & set(cg.list_envs())
    for env_id in REMOVED_IDS:
        with pytest.raises(KeyError):
            cg.make_vec(env_id, num_envs=1)


def test_make_returns_squeezed_single_env():
    env, base = cg.make("Microgrid-Random")
    assert isinstance(env, SingleEnv)
    obs, rejected, info = env.reset(seed=0)
    assert obs.shape == (5,)                       # unbatched, no leading dim
    assert env.action_space.shape == (3,)
    assert not rejected
    assert "capex" in info
    assert set(base) == EMB_KEYS


def test_make_step_is_unbatched():
    env, _ = cg.make("Microgrid-Random")
    env.reset(seed=0)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == (5,)
    assert isinstance(r, float)
    assert isinstance(term, bool) and isinstance(trunc, bool)


def test_make_wraps_make_vec_num_envs_1():
    e1, b1 = cg.make("Microgrid-Random")
    e2, b2 = cg.make_vec("Microgrid-Random", num_envs=1)
    o1, _, _ = e1.reset(seed=0)            # squeezed single env -> (5,)
    o2, _, _ = e2.reset(seed=0)            # vector env -> (1, 5)
    assert o1.shape == (5,)
    assert o2.shape == (1, 5)
    assert np.allclose(o1, o2[0])
    assert set(b1) == set(b2)


def test_make_validate_embodiment_unbatched():
    env, base = cg.make("Microgrid-Random")
    assert env.validate_embodiment(base) is True
    bad = {**base, "pv_capacity_kw": 1e9}
    assert env.validate_embodiment(bad) is False


def test_make_returns_base_embodiment_for_all_ids():
    for eid in MICRO_IDS:
        _, base = cg.make(eid)
        assert set(base) == EMB_KEYS


def test_unknown_env_id_raises():
    with pytest.raises(KeyError):
        cg.make("NoSuchEnv-v0")


# --- vector behavior ---------------------------------------------------------------

def test_batched_shapes_and_reset():
    vec, _ = cg.make_vec("Microgrid-Random", num_envs=4)
    assert vec.observation_space.shape == (4, 5)
    assert vec.action_space.shape == (4, 3)
    obs, rejected, info = vec.reset(seed=0)
    assert obs.shape == (4, 5)
    assert rejected.shape == (4,)
    assert rejected.dtype == bool
    assert not rejected.any()


def test_batched_embodiment_rejection_mask():
    vec, _ = cg.make_vec("Microgrid-Random", num_envs=4)
    emb = {
        "pv_capacity_kw": np.array([100, 200, 300, 9e9]),  # last out of range
        "battery_capacity_kwh": np.array([100, 200, 300, 100]),
        "genset_rating_kw": np.array([50, 100, 150, 50]),
    }
    _, rejected, info = vec.reset(seed=1, embodiment=emb)
    assert rejected.tolist() == [False, False, False, True]


def test_single_embodiment_broadcast():
    vec, base = cg.make_vec("Microgrid-Random", num_envs=3)
    _, rejected, info = vec.reset(seed=0, embodiment=base)
    assert not rejected.any()


def test_list_embodiment_with_none_entries():
    vec, base = cg.make_vec("Microgrid-Random", num_envs=3)
    one = {**base, "pv_capacity_kw": 123.0}
    _, rejected, info = vec.reset(embodiment=[None, one, None])
    assert not rejected.any()
    assert info["embodiment"][1]["pv_capacity_kw"] == pytest.approx(123.0)


def test_validate_embodiment_batch():
    vec, base = cg.make_vec("Microgrid-Random", num_envs=3)
    over = {**base, "pv_capacity_kw": 9e9}
    assert vec.validate_embodiment([None, over, base]).tolist() == [True, False, True]


def test_autoreset_next_step_semantics():
    vec, _ = cg.make_vec("Microgrid-Random", num_envs=2, horizon=3)
    vec.reset(seed=0)
    truncs, rewards = [], []
    for _ in range(4):
        _, r, _, trunc, _ = vec.step(vec.action_space.sample())
        truncs.append(trunc.copy())
        rewards.append(r.copy())
    assert truncs[2].all()          # horizon=3 -> truncates on the 3rd step
    assert not truncs[3].any()      # 4th step is the autoreset
    assert np.allclose(rewards[3], 0.0)


def test_vec_of_one_matches_single():
    env, _ = single("random")
    vec, _ = cg.make_vec("Microgrid-Random", num_envs=1)
    o1, _, _ = env.reset(seed=42)
    o2, _, _ = vec.reset(seed=[42])          # a per-lane list is used as given; an int would be spawned into lane seeds
    assert np.allclose(o1, o2[0])
    a = env.action_space.sample()
    s_obs, s_r, *_ = env.step(a)
    v_obs, v_r, *_ = vec.step(a[None])
    assert np.allclose(s_obs, v_obs[0])
    assert np.isclose(s_r, v_r[0])


def test_transition_level_change_in_step():
    vec, base = cg.make_vec("Microgrid-Random", num_envs=2)
    vec.reset(seed=0, embodiment=base)
    new = {**base, "pv_capacity_kw": base["pv_capacity_kw"] + 10.0}
    _, _, _, _, info = vec.step(vec.action_space.sample(), embodiment=[new, None])
    assert info["embodiment"][0]["pv_capacity_kw"] == pytest.approx(new["pv_capacity_kw"])
    assert info["embodiment"][1]["pv_capacity_kw"] == pytest.approx(base["pv_capacity_kw"])


def test_step_raises_on_invalid_transition_embodiment():
    vec, base = cg.make_vec("Microgrid-Random", num_envs=2)
    vec.reset(seed=0)
    bad = {**base, "pv_capacity_kw": 9e9}
    with pytest.raises(ValueError):
        vec.step(vec.action_space.sample(), embodiment=[bad, None])


def test_eval_env_freezes_embodiment_with_warning():
    vec, base = cg.make_vec("Microgrid-Random", num_envs=2, is_eval_env=True)
    vec.reset(seed=0, embodiment=base)
    new = {**base, "pv_capacity_kw": base["pv_capacity_kw"] + 10.0}
    with pytest.warns(UserWarning, match="Transition-level embodiment changes are not permitted"):
        _, _, _, _, info = vec.step(vec.action_space.sample(), embodiment=[new, None])
    assert info["embodiment"][0]["pv_capacity_kw"] == pytest.approx(base["pv_capacity_kw"])


def test_bad_batch_size_raises():
    vec, _ = cg.make_vec("Microgrid-Random", num_envs=3)
    with pytest.raises(ValueError):
        vec.reset(embodiment={
            "pv_capacity_kw": np.array([1, 2]),
            "battery_capacity_kwh": np.array([1, 2]),
            "genset_rating_kw": np.array([1, 2]),
        })


def test_vector_render_returns_frame_tuple():
    vec, _ = cg.make_vec("Microgrid-Random", num_envs=3, render_mode="rgb_array")
    vec.reset(seed=0)
    vec.step(vec.action_space.sample())     # render needs a step of history
    frames = vec.render()
    assert isinstance(frames, tuple) and len(frames) == 3
    for f in frames:
        assert f.ndim == 3 and f.shape[2] == 3
    vec.close()


def test_partial_reset_touches_only_the_masked_lanes():
    """reset(options={"reset_mask": m}) resets only the masked lanes; the others keep
    their obs and any pending autoreset, and a full reset must come first."""
    env, _ = cg.make_vec("Microgrid-Random", num_envs=2)
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


@pytest.mark.parametrize("setup", ["random", "campus", "offgrid"])
def test_vectorized_numpy_matches_pymgrid_full_episode(setup):
    """Vectorized NumPy engine reproduces pymgrid MicrogridCodesignEnv across a full
    24-hour rollout including mid-episode transition-level embodiment changes."""
    single_env, base = single(setup, horizon=24)
    vec_env = cg.make_vec(
        {"random": "Microgrid-Random", "campus": "Microgrid-Campus", "offgrid": "Microgrid-OffGridSingle"}[setup],
        num_envs=1,
        horizon=24,
    )[0]
    rng = np.random.default_rng(123)
    o_s, rj_s, _ = single_env.reset(seed=7, embodiment=base)
    o_v, rj_v, _ = vec_env.reset(seed=[7], embodiment=base)
    assert not rj_s and not bool(rj_v[0])
    np.testing.assert_allclose(o_v[0], o_s, rtol=1e-5, atol=1e-5)

    for step_i in range(24):
        act = rng.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
        emb = None
        if step_i == 10:
            emb = {
                "pv_capacity_kw": base["pv_capacity_kw"] * 1.3,
                "battery_capacity_kwh": base["battery_capacity_kwh"] * 0.6,
                "genset_rating_kw": base["genset_rating_kw"] * 0.8,
            }
        os_s, rs_s, ts_s, tr_s, inf_s = single_env.step(act, embodiment=emb)
        os_v, rs_v, ts_v, tr_v, inf_v = vec_env.step(act[None], embodiment=emb)
        np.testing.assert_allclose(os_v[0], os_s, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(float(rs_v[0]), float(rs_s), rtol=1e-4, atol=1e-3)
        assert bool(ts_v[0]) == bool(ts_s) and bool(tr_v[0]) == bool(tr_s)
        np.testing.assert_allclose(float(inf_v["soc"][0]), float(inf_s["soc"]), rtol=1e-5, atol=1e-5)
    single_env.close()
    vec_env.close()
