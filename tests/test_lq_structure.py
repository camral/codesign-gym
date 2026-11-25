"""Tests for the LQ structure co-design environment (single-env + dynamics).

The single env is built directly (``codesign_gym.make`` returns the batched num_envs=1
vector env; see test_lq_vector.py for the batched API). Ground-truth solving lives in the
demo script and is not imported here.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.lq_structure import designs, dynamics
from envs.lq_structure.config import make_config
from envs.lq_structure.env import LQStructureEnv
from core.codesign_env import CoDesignEnv


ALL_SETUPS = ["lqr", "lqg", "qp"]


def single(setup="lqr", *, is_eval_env=False, render_mode=None, n=6, **cfg):
    """Build a single LQStructureEnv plus its base embodiment."""
    c = make_config(setup, n=n, **cfg)
    env = LQStructureEnv(c, is_eval_env=is_eval_env, render_mode=render_mode)
    return env, designs.base_embodiment(c)


# --- base class -------------------------------------------------------------------

def test_subclasses_codesign_env():
    env, _ = single("lqr", n=4)
    assert isinstance(env, CoDesignEnv)


# --- dynamics ----------------------------------------------------------------------

def test_tridiagonal_shape_and_values():
    K = dynamics.tridiagonal_from_links(np.array([2.0, 3.0, 4.0]))
    assert K[0, 0] == pytest.approx(5.0)
    assert K[1, 1] == pytest.approx(7.0)
    assert K[2, 2] == pytest.approx(4.0)
    assert K[0, 1] == pytest.approx(-3.0)
    assert K[1, 2] == pytest.approx(-4.0)
    assert np.allclose(K, K.T)


def test_continuous_state_space_shapes():
    n = 4
    A, B, G = dynamics.continuous_state_space(np.ones(n), np.ones(n), np.full(n, 0.5), np.ones(n))
    assert A.shape == (2 * n, 2 * n)
    assert B.shape == (2 * n, n)
    assert G.shape == (2 * n, n)
    assert np.allclose(A[:n, n:], np.eye(n))


def test_free_chain_is_stable():
    n = 5
    A, _, _ = dynamics.continuous_state_space(np.ones(n), np.full(n, 2.0), np.full(n, 0.5), np.zeros(n))
    eig = np.linalg.eigvals(A)
    assert np.all(eig.real < 0)


def test_actuator_matrix_columns_match_placement():
    masses = np.array([1.0, 2.0, 4.0])
    B = dynamics.actuator_matrix(masses, np.array([1, 0, 1]))
    assert B.shape == (6, 2)
    assert B[3, 0] == pytest.approx(1.0)
    assert B[5, 1] == pytest.approx(0.25)


def test_discretize_no_input():
    n = 3
    A, _, _ = dynamics.continuous_state_space(np.ones(n), np.ones(n), np.full(n, 0.3), np.zeros(n))
    Ad, Bd = dynamics.discretize(A, np.zeros((2 * n, 0)), 0.05)
    assert Ad.shape == (2 * n, 2 * n)
    assert Bd.shape == (2 * n, 0)


# --- spaces / embodiment -----------------------------------------------------------

@pytest.mark.parametrize("setup", ALL_SETUPS)
def test_embodiment_space_and_base(setup):
    env, base = single(setup, n=4)
    assert set(env.embodiment_space.spaces) == set(designs.EMB_KEYS)
    assert env.validate_embodiment(base)
    for key in ("masses", "stiffness", "damping"):
        assert base[key].shape == (4,)


@pytest.mark.parametrize("setup", ALL_SETUPS)
def test_spaces_dims(setup):
    env, _ = single(setup, n=6)
    assert env.observation_space.shape == (24,)  # 4n
    assert env.action_space.shape == (6,)         # one slot per possible actuator


def test_validate_rejects_out_of_bounds():
    env, base = single("lqr", n=4)
    bad = {k: v.copy() for k, v in base.items()}
    bad["masses"] = bad["masses"] + 100.0
    assert env.validate_embodiment(bad) is False


def test_validate_requires_actuator():
    env, base = single("lqr", n=4)
    bad = {k: v.copy() for k, v in base.items()}
    bad["actuators"] = np.zeros(4, dtype=np.int8)
    assert not env.validate_embodiment(bad)


def test_validate_actuator_budget():
    env, base = single("qp", n=4)  # actuator_budget = 2
    bad = {k: v.copy() for k, v in base.items()}
    bad["actuators"] = np.ones(4, dtype=np.int8)
    assert not env.validate_embodiment(bad)


def test_validate_mass_budget():
    env, base = single("qp", n=4)
    bad = {k: v.copy() for k, v in base.items()}
    bad["masses"] = np.full(4, env.cfg.mass_high)
    assert not env.validate_embodiment(bad)


def test_every_actuator_placement_fits_the_action_space():
    # lqg designs actuator placement (lqr pins it): the action space has a slot for every
    # possible actuator, so a full placement and a partial one are both valid.
    env, base = single("lqg", n=4)
    assert env.action_space.shape == (4,)
    for placement in (np.ones(4, dtype=np.int8), np.array([1, 1, 0, 0], dtype=np.int8)):
        design = {k: v.copy() for k, v in base.items()}
        design["actuators"] = placement
        assert env.validate_embodiment(design)


def test_lqg_requires_sensor():
    env, base = single("lqg", n=4)
    bad = {k: v.copy() for k, v in base.items()}
    bad["sensors"] = np.zeros(4, dtype=np.int8)
    assert not env.validate_embodiment(bad)


# --- reset / step ------------------------------------------------------------------

@pytest.mark.parametrize("setup", ALL_SETUPS)
def test_reset_step_shapes(setup):
    env, _ = single(setup, n=5)
    obs, rejected, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert rejected is False
    obs2, r, term, trunc, info2 = env.step(np.zeros(env.action_space.shape))
    assert obs2.shape == env.observation_space.shape
    assert isinstance(r, float)
    assert term is False


def test_reward_is_nonpositive_and_zero_at_origin():
    env, _ = single("lqr", n=4, process_noise_std=0.0)
    env.reset(seed=1)
    env._x = np.zeros(env.cfg.state_dim)
    _, r, _, _, _ = env.step(np.zeros(env.action_space.shape))
    assert r == pytest.approx(0.0)


def test_determinism_with_seed():
    env, _ = single("lqg", n=4)
    o1, _, _ = env.reset(seed=42)
    r1 = [env.step(np.zeros(env.action_space.shape))[1] for _ in range(5)]
    o2, _, _ = env.reset(seed=42)
    r2 = [env.step(np.zeros(env.action_space.shape))[1] for _ in range(5)]
    assert np.allclose(o1, o2)
    assert np.allclose(r1, r2)


def test_horizon_truncation():
    env, _ = single("lqr", n=3, horizon=10)
    env.reset(seed=0)
    truncs = []
    for _ in range(10):
        _, _, term, trunc, _ = env.step(np.zeros(env.action_space.shape))
        truncs.append(trunc)
    assert truncs[-1] is True
    assert not any(truncs[:-1])


def test_impulse_reset_mode():
    env, _ = single("lqr", n=4, reset_mode="impulse")
    _, _, info = env.reset(seed=0)
    x = info["state"]
    assert np.allclose(x[:4], 0.0)
    assert x[4] == pytest.approx(1.0)
    assert np.allclose(x[5:], 0.0)


# --- transition-level embodiment ---------------------------------------------------

def test_transition_level_change_rebuilds_plant():
    env, base = single("lqr", n=4)
    _, _, info0 = env.reset(seed=0)
    Ad_before = info0["plant"]["Ad"].copy()
    new = {k: v.copy() for k, v in base.items()}
    new["masses"] = np.full(4, env.cfg.mass_high)
    _, _, _, _, info1 = env.step(np.zeros(env.action_space.shape), embodiment=new)
    assert not np.allclose(Ad_before, info1["plant"]["Ad"])


def test_transition_level_placement_change():
    env, base = single("lqg", n=4)
    env.reset(seed=0)
    new = {k: v.copy() for k, v in base.items()}
    new["actuators"] = np.array([1, 0, 0, 0], dtype=np.int8)
    _, _, _, _, info = env.step(np.zeros(env.action_space.shape), embodiment=new)
    assert info["plant"]["n_actuators"] == 1


def test_step_rejects_invalid_embodiment():
    env, base = single("lqr", n=4)
    env.reset(seed=0)
    bad = {k: v.copy() for k, v in base.items()}
    bad["actuators"] = np.zeros(4, dtype=np.int8)
    with pytest.raises(ValueError):
        env.step(np.zeros(env.action_space.shape), embodiment=bad)


def test_eval_env_freezes_embodiment_with_warning():
    # CoDesignEnv nulls the step embodiment when is_eval_env, so the plant is unchanged,
    # and warns that this is an evaluation environment (base-managed freeze).
    env, base = single("lqr", n=4, is_eval_env=True)
    _, _, info0 = env.reset(seed=0)
    Ad_before = info0["plant"]["Ad"].copy()
    new = {k: v.copy() for k, v in base.items()}
    new["masses"] = np.full(4, env.cfg.mass_high)
    with pytest.warns(UserWarning, match="not permitted"):
        _, _, _, _, info1 = env.step(np.zeros(env.action_space.shape), embodiment=new)
    assert np.allclose(Ad_before, info1["plant"]["Ad"])


# --- embodiment_rejected path ------------------------------------------------------

def test_reset_rejects_invalid_embodiment_and_step_raises():
    env, base = single("lqr", n=4)
    bad = {k: v.copy() for k, v in base.items()}
    bad["masses"] = bad["masses"] + 100.0
    obs, rejected, info = env.reset(seed=0, embodiment=bad)
    assert rejected is True
    with pytest.raises(RuntimeError):
        env.step(np.zeros(env.action_space.shape))


def test_reset_with_valid_embodiment():
    env, base = single("qp", n=4)
    emb = {k: v.copy() for k, v in base.items()}
    emb["actuators"] = np.array([0, 1, 0, 0], dtype=np.int8)
    obs, rejected, info = env.reset(seed=0, embodiment=emb)
    assert rejected is False
    assert info["plant"]["n_actuators"] == 1


# --- info exposures ----------------------------------------------------------------

def test_step_info_exposes_state_control_cost():
    env, _ = single("lqr", n=4, process_noise_std=0.0)
    _, _, rinfo = env.reset(seed=0)
    x0 = rinfo["state"]
    _, r, _, _, info1 = env.step(np.zeros(env.action_space.shape))
    assert np.allclose(info1["state"], x0)
    _, _, _, _, info2 = env.step(np.zeros(env.action_space.shape))
    assert not np.allclose(info1["state"], info2["state"])
    assert "applied_control" in info1 and "stage_cost" in info1
    assert r == pytest.approx(-info1["stage_cost"] * env.cfg.dt)


def test_applied_control_is_saturated():
    env, _ = single("qp", n=4)  # u_max = 1.0
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.full(env.action_space.shape, 100.0))
    assert np.all(np.abs(info["applied_control"]) <= env.cfg.u_max + 1e-9)
    assert info["applied_control"].shape == (info["n_actuators"],)


def test_info_exposes_embodiment_copy():
    env, base = single("lqr", n=4)
    _, _, info = env.reset(seed=0)
    emb = info["embodiment"]
    assert set(emb) == set(base)
    emb["masses"][0] = 999.0
    _, _, _, _, info2 = env.step(np.zeros(env.action_space.shape))
    assert info2["embodiment"]["masses"][0] != 999.0


def test_info_exposes_plant_matrices():
    env, _ = single("lqr", n=4)
    _, _, info = env.reset(seed=0)
    plant = info["plant"]
    assert plant["Ad"].shape == (8, 8)
    assert plant["Cs"].shape[1] == 8
    assert {"A_cont", "Ad", "Bd", "Gd", "Cs", "n_actuators", "n_sensors"} <= set(plant)


# --- rendering ---------------------------------------------------------------------

def test_render_rgb_array_returns_frame():
    env, _ = single("lqg", n=4, render_mode="rgb_array")
    assert env.render() is None
    env.reset(seed=0)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8 and frame.shape[0] > 0 and frame.shape[1] > 0
    env.step(np.zeros(env.action_space.shape))
    assert env.render().shape == frame.shape
    env.close()


def test_render_none_when_no_mode():
    env, _ = single("lqr", n=4)
    env.reset(seed=0)
    assert env.render() is None


def test_render_mode_validation():
    with pytest.raises(ValueError):
        single("lqr", n=4, render_mode="bogus")


@pytest.mark.parametrize("setup,frozen", [("lqr", ("actuators", "sensors")), ("lqg", ()), ("qp", ("sensors",))])
def test_frozen_placements_are_pinned(setup, frozen):
    """design_actuators/design_sensors=False freeze that placement at its base value:
    the space only samples it, and validation rejects any other placement."""
    env, base = single(setup, n=4)
    env.embodiment_space.seed(0)
    for _ in range(20):
        sample = env.embodiment_space.sample()
        for key in frozen:
            np.testing.assert_array_equal(sample[key], base[key])
    for key in frozen:
        moved = {k: v.copy() for k, v in base.items()}
        moved[key][-1] = 1 - moved[key][-1]
        assert env.validate_embodiment(moved) is False
    for key in {"actuators", "sensors"} - set(frozen):
        moved = {k: v.copy() for k, v in base.items()}
        moved[key][0] = 1 - moved[key][0]
        assert designs.validate_embodiment(moved, env.cfg)[1] != f"{key} placement is fixed in this setup (must equal the base placement)"
