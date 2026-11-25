"""Tests for the microgrid sizing-and-dispatch co-design environment (single env).

The single env is built directly (``codesign_gym.make`` returns the squeezed
num_envs=1 vector env; see test_microgrid_vector.py for the batched + central
API). These tests poke sub-env internals (``_window``, ``_amort_capex`` ...), so
they construct :class:`MicrogridCodesignEnv` directly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.codesign_env import CoDesignEnv
from envs.microgrid import embodiment as E
from envs.microgrid.env import MicrogridCodesignEnv
from envs.microgrid.registration import SETUP_CONFIGS

SETUPS = list(SETUP_CONFIGS)
EMB_KEYS = {"pv_capacity_kw", "battery_capacity_kwh", "genset_rating_kw"}


def single(setup="random", **kw):
    """Build a single MicrogridCodesignEnv for a setup, plus its base embodiment."""
    cfg = {**SETUP_CONFIGS[setup], **kw}
    env = MicrogridCodesignEnv(**cfg)
    return env, env.base_embodiment


# --- base class -------------------------------------------------------------------

def test_subclasses_codesign_env():
    env, _ = single("random")
    assert isinstance(env, CoDesignEnv)


# --- embodiment module ------------------------------------------------------------

def test_space_and_base():
    sp = E.make_embodiment_space()
    base = E.base_embodiment(sp)
    assert set(base) == set(E.EMBODIMENT_KEYS)
    assert E.validate_embodiment(base, sp)


@pytest.mark.parametrize(
    "emb,ok",
    [
        ({"pv_capacity_kw": 250, "battery_capacity_kwh": 500, "genset_rating_kw": 250}, True),
        ({"pv_capacity_kw": -1, "battery_capacity_kwh": 500, "genset_rating_kw": 250}, False),
        ({"pv_capacity_kw": 1e9, "battery_capacity_kwh": 500, "genset_rating_kw": 250}, False),
        ({"pv_capacity_kw": 250, "battery_capacity_kwh": 500}, False),  # missing key
        ({"pv_capacity_kw": np.nan, "battery_capacity_kwh": 500, "genset_rating_kw": 250}, False),
    ],
)
def test_validate(emb, ok):
    sp = E.make_embodiment_space()
    assert E.validate_embodiment(emb, sp) is ok


def test_capex_linear_and_amortization():
    emb = {"pv_capacity_kw": 100, "battery_capacity_kwh": 200, "genset_rating_kw": 50}
    total = E.capex(emb)
    assert total == 100 * 1000 + 200 * 300 + 50 * 500
    amort = E.amortized_capex(emb, horizon=24)
    assert amort == pytest.approx(total * 24 / (10 * 8760))
    assert 0 < amort < total


# --- reset / step -----------------------------------------------------------------

@pytest.mark.parametrize("setup", SETUPS)
def test_reset_step_rollout(setup):
    env, base = single(setup)
    assert set(base) == EMB_KEYS
    obs, rejected, info = env.reset(seed=0)
    assert rejected is False
    assert env.observation_space.contains(obs)
    assert "capex" in info
    for _ in range(env.horizon):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(r)
        assert "capex" in info  # capex reported every step
    assert trunc and not term


def test_determinism_with_seed():
    env, _ = single("random")
    o1, _, _ = env.reset(seed=42)
    r1 = [env.step(np.zeros(env.action_space.shape, np.float32))[1] for _ in range(5)]
    o2, _, _ = env.reset(seed=42)
    r2 = [env.step(np.zeros(env.action_space.shape, np.float32))[1] for _ in range(5)]
    assert np.allclose(o1, o2) and np.allclose(r1, r2)


def test_capex_charged_only_first_step():
    env, base = single("random")
    env.reset(seed=0, embodiment=base)
    charged = []
    for _ in range(env.horizon):
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
        charged.append(info["capex_charged"])
    assert charged[0] > 0
    assert all(c == 0 for c in charged[1:])
    assert charged[0] == pytest.approx(env._amort_capex)


def test_energy_balance_matches_theory():
    # A grid-balancing policy (import deficit / export surplus) should cost
    # exactly the imported energy times the tariff, plus the first-step capex.
    env, base = single("random", grid_max_import=2000.0, grid_max_export=2000.0)
    obs, _, _ = env.reset(seed=7)
    total = 0.0
    for _ in range(env.horizon):
        _, demand, pv, _, _ = obs[:5]
        a_grid = np.clip((demand - pv) / 2000.0, -1, 1)
        obs, r, _, _, _ = env.step(np.array([0.0, 0.0, a_grid], dtype=np.float32))
        total += r
    env.reset(seed=7)
    w = env._window
    pv_avail = base["pv_capacity_kw"] * w.solar_cf
    imp = np.maximum(w.demand_kw - pv_avail, 0).sum() * env.import_price
    cred = np.maximum(pv_avail - w.demand_kw, 0).sum() * env.export_price
    amort = env._amort_capex
    assert total == pytest.approx(-imp + cred - amort, rel=1e-4, abs=1e-2)


# --- embodiment_rejected path -----------------------------------------------------

def test_rejected_embodiment_then_step_raises():
    env, _ = single("random")
    obs, rejected, info = env.reset(embodiment={"pv_capacity_kw": 1e9, "battery_capacity_kwh": 1, "genset_rating_kw": 1})
    assert rejected is True
    with pytest.raises(RuntimeError):
        env.step(env.action_space.sample())


# --- transition-level embodiment --------------------------------------------------

def test_transition_soc_projection_and_no_reward_capex():
    env, _ = single("random")
    env.reset(seed=2, embodiment={"pv_capacity_kw": 100, "battery_capacity_kwh": 400, "genset_rating_kw": 100})
    for _ in range(5):
        env.step(np.array([-1.0, 0, 1.0], dtype=np.float32))  # charge up
    stored_before = env._current_stored_kwh()
    _, r, _, _, info = env.step(
        np.array([0, 0, 0.5], dtype=np.float32),
        embodiment={"pv_capacity_kw": 100, "battery_capacity_kwh": 50, "genset_rating_kw": 100},
    )
    # stored energy clamped to the new (smaller) capacity; capex not charged to reward
    assert env._current_stored_kwh() <= 50 + 1e-6
    assert env._current_stored_kwh() == pytest.approx(min(stored_before, 50.0), abs=1e-6)
    assert info["capex_charged"] == 0.0
    assert info["capex_delta"] >= 0.0


def test_eval_env_freezes_embodiment_with_warning():
    # CoDesignEnv nulls the step embodiment when is_eval_env, so the design is
    # frozen and warns that transition-level changes are not permitted.
    env, _ = single("random", is_eval_env=True)
    start = {"pv_capacity_kw": 100, "battery_capacity_kwh": 400, "genset_rating_kw": 100}
    env.reset(seed=0, embodiment=start)
    other = {"pv_capacity_kw": 50, "battery_capacity_kwh": 50, "genset_rating_kw": 50}
    with pytest.warns(UserWarning, match="Transition-level embodiment changes"):
        _, _, _, _, info = env.step(env.action_space.sample(), embodiment=other)
    assert info["embodiment"] == start  # transition frozen out


# --- forecast ---------------------------------------------------------------------

def test_forecast_obs_dim():
    env, _ = single("forecast")
    obs, _, _ = env.reset(seed=0)
    assert obs.shape[0] == 5 + 2 * env.forecast_horizon


# --- rendering --------------------------------------------------------------------

def test_render_rgb_array_returns_frame():
    env, _ = single("random", render_mode="rgb_array")
    assert env.render() is None            # no history yet
    env.reset(seed=0)
    assert env.render() is None            # still no step taken
    env.step(env.action_space.sample())
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8 and frame.shape[0] > 0 and frame.shape[1] > 0
    env.close()


def test_render_none_when_no_mode():
    env, _ = single("random")
    env.reset(seed=0)
    env.step(env.action_space.sample())
    assert env.render() is None


def test_render_mode_validation():
    with pytest.raises(ValueError):
        single("random", render_mode="bogus")


def _episode(env, seed, n=8):
    obs, rejected, _ = env.reset(seed=seed)
    assert not rejected
    soc0, rewards = float(obs[0]), []
    for _ in range(n):
        obs, r, *_ = env.step(env.action_space.sample())
        rewards.append(float(r))
    return soc0, rewards


def test_every_episode_starts_at_init_soc_and_same_seed_repeats():
    """pymgrid's reset keeps the battery charge; a same-design reset must still start
    each episode at BATTERY_INIT_SOC, so episodes are independent and seed-reproducible."""
    env, _ = single()
    env.action_space.seed(0)
    first = _episode(env, seed=3)
    env.action_space.seed(0)
    second = _episode(env, seed=3)
    assert first[0] == second[0] == pytest.approx(E.BATTERY_INIT_SOC)
    assert first[1] == second[1]
