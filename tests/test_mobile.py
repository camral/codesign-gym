"""Tests for the mobile-env placement-and-connection co-design environment (single env).

These construct :class:`MobileEnvCoDesign` directly to poke internals (padding,
action masking, MComCore rebuild). The batched + central API is covered in
test_mobile_vector.py.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("mobile_env")

from core.codesign_env import CoDesignEnv
from envs.mobile_env import embodiment as E
from envs.mobile_env.config import PRESETS, make_config
from envs.mobile_env.env import MobileEnvCoDesign

SETUPS = list(PRESETS)
EMB_KEYS = {"num_bs", "stations"}


def single(setup="dense_urban", **kw):
    """Build a single MobileEnvCoDesign for a preset, plus its base embodiment."""
    cfg = make_config(setup)
    env = MobileEnvCoDesign(cfg, **kw)
    return env, env.base_embodiment


def design(cfg, positions, tx=None):
    """Build a valid embodiment with the given active BS positions."""
    tx = cfg.seed_tx if tx is None else tx
    stations = np.zeros((cfg.max_bs, 3), dtype=np.float32)
    stations[:, 2] = cfg.tx_min
    for i, (x, y) in enumerate(positions):
        stations[i] = (x, y, tx)
    return {"num_bs": len(positions), "stations": stations}


# --- base class + embodiment module -----------------------------------------------

def test_subclasses_codesign_env():
    env, _ = single()
    assert isinstance(env, CoDesignEnv)


def test_space_and_base_valid():
    cfg = make_config("dense_urban")
    base = E.base_embodiment(cfg)
    assert set(base) == EMB_KEYS
    assert E.validate_embodiment(base, cfg)
    # base embodiment lies inside the declared embodiment_space
    assert E.make_embodiment_space(cfg).contains(base)


@pytest.mark.parametrize("setup", SETUPS)
def test_base_embodiment_num_bs_in_budget(setup):
    cfg = make_config(setup)
    base = E.base_embodiment(cfg)
    assert cfg.min_bs <= base["num_bs"] <= cfg.max_bs


def test_validate_rejects_bad_designs():
    cfg = make_config("dense_urban")  # min_bs=3, max_bs=8, map 150x150, tx 23-33
    assert E.validate_embodiment(design(cfg, [(10, 10), (20, 20), (30, 30)]), cfg)
    # too few base stations (below min_bs)
    assert not E.validate_embodiment(design(cfg, [(10, 10)]), cfg)
    # station off the map
    assert not E.validate_embodiment(design(cfg, [(10, 10), (20, 20), (999, 20)]), cfg)
    # tx power out of range
    assert not E.validate_embodiment(design(cfg, [(10, 10), (20, 20), (30, 30)], tx=100), cfg)
    # missing key
    assert not E.validate_embodiment({"num_bs": 3}, cfg)


def _line_positions(cfg, n):
    """``n`` distinct on-map positions for a design of ``n`` active stations."""
    return [(5.0 + 3.0 * i, 5.0) for i in range(n)]


@pytest.mark.parametrize("setup", SETUPS)
def test_tx_budget_rejects_full_fleet_at_max(setup):
    cfg = make_config(setup)
    budget = E.tx_budget_watts(cfg)
    # budget is a fraction (<1) of the whole fleet at tx_max, so all max_bs
    # stations at tx_max must exceed it and be rejected.
    assert cfg.tx_budget_fraction < 1.0
    full = design(cfg, _line_positions(cfg, cfg.max_bs), tx=cfg.tx_max)
    assert not E.validate_embodiment(full, cfg)
    # the seed deployment stays within budget
    assert E.validate_embodiment(E.base_embodiment(cfg), cfg)
    # a scarce deployment can afford tx_max: min_bs stations fit under budget
    lean = design(cfg, _line_positions(cfg, cfg.min_bs), tx=cfg.tx_max)
    assert E.validate_embodiment(lean, cfg)


def test_tx_budget_boundary():
    cfg = make_config("dense_urban")
    budget = E.tx_budget_watts(cfg)  # max_bs * W(tx_max) * 0.8
    # tx that spreads the budget evenly across all max_bs stations
    tx_edge = 10.0 * np.log10((budget / cfg.max_bs) * 1000.0)  # W -> dBm
    # a hair under the even-split budget -> valid; a hair over -> rejected
    # (0.2 dB ~ 4.7% power, comfortably clear of float32 rounding)
    under = design(cfg, _line_positions(cfg, cfg.max_bs), tx=tx_edge - 0.2)
    assert E.validate_embodiment(under, cfg)
    over = design(cfg, _line_positions(cfg, cfg.max_bs), tx=tx_edge + 0.2)
    assert not E.validate_embodiment(over, cfg)


# --- reset / step -----------------------------------------------------------------

@pytest.mark.parametrize("setup", SETUPS)
def test_reset_step_rollout(setup):
    env, base = single(setup)
    obs, rejected, info = env.reset(seed=0)
    assert rejected is False
    assert env.observation_space.contains(obs)
    assert info["num_bs"] == base["num_bs"]
    assert info["action_mask"].shape == (env.num_ues, env.max_bs + 1)
    trunc = False
    for _ in range(env.cfg.ep_max_time + 5):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        assert np.isfinite(r)
        if trunc:
            break
    assert trunc and not term


def test_determinism_with_seed():
    env, _ = single("dense_urban")
    o1, _, _ = env.reset(seed=42)
    a = env.action_space.sample()
    r1 = [env.step(a)[1] for _ in range(5)]
    o2, _, _ = env.reset(seed=42)
    r2 = [env.step(a)[1] for _ in range(5)]
    assert np.allclose(o1, o2) and np.allclose(r1, r2)


# --- padding / masking ------------------------------------------------------------

def test_variable_num_bs_padding_and_mask():
    env, _ = single("dense_urban")  # num_ues=15, max_bs=8
    emb = design(env.cfg, [(10, 10), (20, 20), (30, 30), (40, 40)])  # num_bs=4
    obs, rejected, info = env.reset(seed=1, embodiment=emb)
    assert rejected is False
    # padded obs shape uses max_bs, not the active count
    assert obs.shape == (env.num_ues * (2 * env.max_bs + 1),)
    mask = info["action_mask"]
    # NOOP + 4 real stations valid; the 4 padded station columns are not
    assert mask[:, 0].all()
    assert mask[:, 1:5].all()
    assert not mask[:, 5:].any()


def test_sanitize_action_maps_padded_to_noop():
    # Choosing a non-existent (padded) station must not crash mobile-env's indexing.
    env, _ = single("dense_urban")
    emb = design(env.cfg, [(10, 10), (20, 20), (30, 30)])  # num_bs=3
    env.reset(seed=0, embodiment=emb)
    bad = np.full(env.num_ues, env.max_bs, dtype=int)  # all pick the last padded slot
    obs, r, term, trunc, info = env.step(bad)
    assert np.isfinite(r)
    assert env.observation_space.contains(obs)


# --- embodiment_rejected path -----------------------------------------------------

def test_rejected_embodiment_then_step_raises():
    env, _ = single("dense_urban")
    bad = design(env.cfg, [(10, 10)])  # below min_bs
    obs, rejected, info = env.reset(embodiment=bad)
    assert rejected is True
    with pytest.raises(RuntimeError):
        env.step(env.action_space.sample())


# --- transition-level embodiment (unsupported) ------------------------------------

def test_step_embodiment_is_ignored_with_warning():
    env, base = single("dense_urban")
    env.reset(seed=0, embodiment=base)
    other = design(env.cfg, [(11, 11), (22, 22), (33, 33), (44, 44), (55, 55)])
    with pytest.warns(UserWarning):
        _, _, _, _, info = env.step(env.action_space.sample(), embodiment=other)
    # design unchanged mid-episode
    assert info["num_bs"] == base["num_bs"]


def test_eval_env_freezes_embodiment_with_warning():
    # CoDesignEnv nulls the step embodiment when is_eval_env, warning that this is an
    # evaluation environment (the eval reason takes priority over the env's own).
    env, base = single("dense_urban", is_eval_env=True)
    env.reset(seed=0, embodiment=base)
    other = design(env.cfg, [(11, 11), (22, 22), (33, 33), (44, 44), (55, 55)])
    with pytest.warns(UserWarning, match="not permitted"):
        _, _, _, _, info = env.step(env.action_space.sample(), embodiment=other)
    assert info["num_bs"] == base["num_bs"]


# --- rendering --------------------------------------------------------------------

def test_render_mode_validation():
    with pytest.raises(ValueError):
        single("dense_urban", render_mode="bogus")


def test_render_rgb_with_overlay():
    env, _ = single("dense_urban", render_mode="rgb_array")
    assert env.render() is None                     # no underlying env yet
    env.reset(seed=0)
    env.step(env.action_space.sample())
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
    # the co-design overlay draws a dark panel in the top-left corner, so that
    # region is no longer mobile-env's white background
    # semi-transparent dark panel in the bottom-left margin: white (255) -> grey
    corner = frame[-55:-10, 10:110]
    assert corner.min() < 150
    assert frame[10:55, 10:110].min() > 200   # top-left is now clear of the panel
    env.close()


def test_render_supports_station_on_map_boundary():
    env, _ = single("dense_urban", render_mode="rgb_array")
    emb = design(env.cfg, [(0, 20), (50, 50), (100, 100)])
    env.reset(seed=0, embodiment=emb)
    env.step(env.action_space.sample())
    frame = env.render()
    assert frame.ndim == 3 and np.isfinite(frame).all()
    env.close()


def test_overlay_lines_report_design():
    env, _ = single("dense_urban")
    emb = design(env.cfg, [(10, 10), (20, 20), (30, 30), (40, 40)], tx=27)
    env.reset(seed=0, embodiment=emb)
    lines = env._overlay_lines()
    assert "4/8 BS" in lines[0]
    assert "27" in lines[1]                         # per-BS tx power reported
    assert "4 live + 4 padded" in lines[3]
