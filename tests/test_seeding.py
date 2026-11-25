"""Seeding contract across env families: reset(seed=...) takes an integer (int or a
NumPy integer) or a per-lane sequence, and a seeded reset replays its episode."""

from __future__ import annotations

import numpy as np
import pytest

import codesign_gym as cg
from core.seeding import lane_seeds, parse_seed

CPU = dict(device="cpu", use_graph=False)


def test_parse_seed_forms():
    assert parse_seed(None, 3) is None
    assert parse_seed(np.int64(7), 3) == 7 and parse_seed(np.array(7), 3) == 7
    assert parse_seed([1, np.int32(2), 3], 3) == [1, 2, 3]
    spawned = lane_seeds(5, 3)
    assert len(set(spawned)) == 3 and spawned == lane_seeds(5, 3) and all(0 <= s < 2**31 for s in spawned)
    assert not set(spawned) & set(lane_seeds(6, 3))                  # neighbouring run seeds share no lane
    assert lane_seeds([9, 1, 4], 3) == [9, 1, 4] and lane_seeds(None, 2) == [None, None]
    assert len(lane_seeds(-1, 2)) == 2                                # negative seeds are valid entropy too
    for bad in (True, 1.5, "7", [1, 2]):
        with pytest.raises((TypeError, ValueError)):
            parse_seed(bad, 3)


def _rollout(env, seed, steps=8):
    obs, *_ = env.reset(seed=seed)
    frames = [np.asarray(obs.cpu() if hasattr(obs, "cpu") else obs).copy()]
    rng = np.random.default_rng(0)
    for _ in range(steps):
        a = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
        obs, *_ = env.step(a)
        frames.append(np.asarray(obs.cpu() if hasattr(obs, "cpu") else obs).copy())
    return np.stack(frames)


@pytest.mark.parametrize("env_id,kw", [("Hopper-v4", CPU), ("BallCatcher-Pitch", CPU), ("SolarCleaner-Stability", CPU),
                                       ("SolarCleaner-Traverse", CPU), ("TruckUnload-Single", CPU)])
def test_batched_warp_envs_accept_every_seed_form_and_replay(env_id, kw):
    env, _ = cg.make_vec(env_id, 2, **kw)
    try:
        a = _rollout(env, 7)
        np.testing.assert_array_equal(a, _rollout(env, 7))                 # a seeded reset replays its episode
        np.testing.assert_array_equal(a, _rollout(env, np.int64(7)))       # a NumPy integer is the same seed
        lanes = _rollout(env, [7, 11])
        np.testing.assert_array_equal(lanes, _rollout(env, [7, 11]))
        np.testing.assert_array_equal(lanes[:, 0], a[:, 0])                # lane 0 of [7, 11] is lane 0 of 7
    finally:
        env.close()


@pytest.mark.parametrize("env_id,kw", [("LQStructure-LQR", {"n": 4}), ("Microgrid-Random", {}), ("Network-Urban", {}),
                                       ("Warehouse-Small", {}), ("Racing-Levine", {})])
def test_lane_envs_accept_every_seed_form(env_id, kw):
    if env_id.startswith("Racing"):
        pytest.importorskip("f1tenth_gym_jax")
    env, _ = cg.make_vec(env_id, 2, **kw)
    try:
        a, _, _ = env.reset(seed=np.int64(3))
        b, _, _ = env.reset(seed=3)
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        c, _, _ = env.reset(seed=[3, 4])
        d, _, _ = env.reset(seed=[3, 4])
        np.testing.assert_array_equal(np.asarray(c), np.asarray(d))
    finally:
        env.close()


def test_ball_catcher_shoves_replay_under_a_seed():
    """With shoves on, two resets with one seed replay the same shove directions."""
    env, _ = cg.make_vec("BallCatcher-Pitch", 2, force_mag=50.0, force_every=4, force_hold=2, **CPU)
    try:
        np.testing.assert_array_equal(_rollout(env, 5, steps=30), _rollout(env, 5, steps=30))
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ["RotateInHand-Cube", "HandPickUp-Cube"])
def test_hand_envs_accept_every_seed_form_and_reproduce_the_spawn(env_id):
    # The seed draws the spawn and goal; compare that state exactly. Later steps are
    # not bitwise reproducible on CUDA (atomics in the contact solve), seed or not.
    env, _ = cg.make_vec(env_id, 2)
    try:
        a = _rollout(env, 7, steps=0)
        np.testing.assert_array_equal(a, _rollout(env, np.int64(7), steps=0))
        lanes = _rollout(env, [7, 11], steps=0)
        np.testing.assert_array_equal(lanes, _rollout(env, [7, 11], steps=0))
        np.testing.assert_array_equal(lanes[:, 0], a[:, 0])
        assert not np.array_equal(lanes[:, 1], a[:, 1])                     # lane 1 took its own seed
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ["Hopper-v4", "BallCatcher-Pitch", "TruckUnload-Single"])
def test_neighbouring_seeds_do_not_replay_each_other_a_step_apart(env_id):
    """Streams are keyed by (seed, step) hashed together. With seed + step, a run seeded 0 respawned two steps later
    drew exactly what a fresh reset seeded 2 draws."""
    import torch
    a, _ = cg.make_vec(env_id, 2, device="cpu")
    b, _ = cg.make_vec(env_id, 2, device="cpu")
    try:
        a.reset(seed=0)
        a.step(torch.zeros(a.action_space.shape))
        a.reset_mask[:] = True                               # as if both copies ended on that step
        a._reset_pending = True
        respawned = a.step(torch.zeros(a.action_space.shape))[0]
        for seed in (1, 2, 3):                                # with seed + step it matched seed 2: two ticks later
            assert not torch.allclose(respawned, b.reset(seed=seed)[0])
        torch.testing.assert_close(b.reset(seed=2)[0], b.reset(seed=2)[0])   # a seed still replays exactly
    finally:
        a.close()
        b.close()
