"""Tests for the TruckUnload co-design environment.

Everything runs on the CPU warp device (``device="cpu", use_graph=False``): the
env is GPU-parallel in production, but the codesign contract, the length floor,
the per-stacker targets and the unobserved design are device-independent.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")

import torch  # noqa: E402
import warp as wp  # noqa: E402

import codesign_gym as cg  # noqa: E402

CPU = dict(device="cpu", use_graph=False)
PRESET_IDS = ["TruckUnload-Single", "TruckUnload-Multi"]
# Column + two arm links per robot: loader + 1 stacker, loader + 2 stackers.
N_OFFSETS = {"TruckUnload-Single": 6, "TruckUnload-Multi": 9}
TIME_LIMITS = {"TruckUnload-Single": 120.0, "TruckUnload-Multi": 300.0}


def _make_vec(env_id="TruckUnload-Single", num_envs=2, **overrides):
    return cg.make_vec(env_id, num_envs, **{**CPU, **overrides})


def _link_lengths(env) -> np.ndarray:
    """Deformed length of every offsettable link, per world (links are boxes)."""
    g = env._segs["geom"]
    return 2.0 * wp.to_torch(env.gpu_model.geom_size)[:, g, 2].cpu().numpy()


def _zero(env):
    return torch.zeros((env.num_envs, env.act_dim()))


# --------------------------------------------------------------------------- #
# Registry + build
# --------------------------------------------------------------------------- #
def test_registry_lists_truck_unload_envs():
    ids = cg.list_envs()
    for env_id in PRESET_IDS:
        assert env_id in ids


@pytest.mark.parametrize("env_id", PRESET_IDS)
def test_preset_builds_resets_steps(env_id):
    env, base = _make_vec(env_id, 2)
    try:
        assert base.shape == (N_OFFSETS[env_id],)
        assert env.embodiment_space.shape == (N_OFFSETS[env_id],)
        out = env.reset(embodiment=base)
        assert len(out) == 3, "vectorized reset must be (obs, rejected, info)"
        obs, rejected, _ = out
        assert obs.shape == (2, env.obs_dim())
        assert rejected.dtype == torch.bool and not rejected.any()
        assert env.time_limit_s == TIME_LIMITS[env_id]
        assert env._episode_ctrl_steps == int(TIME_LIMITS[env_id] / (env.timestep * env.phys_substeps))
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(_zero(env))
        assert reward.shape == (2,) and terminated.shape == (2,)
        assert info["reward_per_robot"].shape == (2, env.n_robots)
        assert torch.allclose(info["reward_per_robot"].sum(dim=1), reward)
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# The design is not observed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("env_id", PRESET_IDS)
def test_design_not_in_observation(env_id):
    env, _ = _make_vec(env_id, 2)
    try:
        per_robot = sum(len(a) + len(d) for a, d in zip(env._rob_qadr, env._rob_dadr))
        assert env.obs_dim() == per_robot + 37 * env.n_robots + int(env.time_in_obs)
        # Two very different bodies from the same state: the start observations
        # differ only through what the body does, never by a design block.
        lo, hi = env.offset_bounds
        obs, _, _ = env.reset(embodiment=np.stack([lo, hi]))
        assert obs.shape[1] == env.obs_dim()
        assert env.single_observation_space.shape == (env.obs_dim(),)
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Shortening, and the length floor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("max_offset", [0.1, 0.5])
def test_lower_bound_stops_at_floor_plus_pad(max_offset):
    env, _ = _make_vec("TruckUnload-Single", 3, max_offset=max_offset)
    try:
        lo, hi = env.offset_bounds
        # Every link can shorten, and none past min_link_length + min_length_pad.
        assert np.all(lo < 0.0) and np.all(hi == 1.0)
        floor = env.min_link_length + env.min_length_pad
        env.reset(embodiment=np.stack([lo, hi, np.zeros_like(lo)]))
        L = _link_lengths(env)
        L0 = 2.0 * env.m.geom_size[env._segs["geom"], 2]
        np.testing.assert_allclose(L[2], L0, atol=1e-6)
        assert np.all(L[0] >= floor - 1e-6)
        # A link whose budget fits inside [-1, 0] lands exactly on the floor.
        exact = lo > -1.0
        np.testing.assert_allclose(L[0][exact], floor, atol=1e-5)
        np.testing.assert_allclose(L[1], L0 + max_offset, atol=1e-5)
        assert not np.any(env.validate_embodiment(np.where(exact, lo - 0.05, lo)))
    finally:
        env.close()


def test_device_clip_enforces_floor_on_raw_offsets():
    """The raw engine path (no validation) still cannot cross the floor."""
    env, _ = _make_vec("TruckUnload-Single", 1, max_offset=0.5)
    try:
        env.reset()
        env.reset_mask.fill_(True)
        from envs.truck_unload.truck_unload_env import TruckUnloadParallelEnv
        TruckUnloadParallelEnv.step(env, _zero(env), reset_actions=-torch.ones((1, env.n_offsets)))
        assert np.all(_link_lengths(env) >= env.min_link_length + env.min_length_pad - 1e-6)
    finally:
        env.close()


def test_change_max_offset_moves_bounds():
    env, _ = _make_vec("TruckUnload-Single", 1, max_offset=0.1)
    try:
        lo_small = env.embodiment_space.low.copy()
        env.change_max_offset(0.4)
        lo_big = env.embodiment_space.low
        assert np.all(lo_big >= lo_small) and np.any(lo_big > lo_small)
        np.testing.assert_allclose(env._seg.axis_lo.numpy(), env._segs["axis_lo"], atol=1e-6)
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Codesign contract
# --------------------------------------------------------------------------- #
def test_independent_robots():
    env, _ = _make_vec("TruckUnload-Multi", 1)
    try:
        names = [env.m.body(int(b)).name for b in env._segs["body"]]
        assert len(set(env._segs["axis_act"].tolist())) == env.n_offsets == len(names)
        assert sum(n.startswith("stacker0_") for n in names) == 3
        assert sum(n.startswith("stacker1_") for n in names) == 3
    finally:
        env.close()


def test_design_survives_autoreset():
    env, _ = _make_vec("TruckUnload-Single", 2)
    try:
        lo, hi = env.offset_bounds
        design = np.stack([hi, lo]).astype(np.float32)
        env.reset(embodiment=design)
        before = _link_lengths(env)
        env.step(_zero(env))
        # Force world 0 to respawn on the next step, as an autoreset would.
        env.reset_mask[0] = True
        env.step(_zero(env))
        np.testing.assert_allclose(_link_lengths(env), before, atol=1e-6)
    finally:
        env.close()


def test_single_success_requires_loader_and_stacker_completion():
    env, _ = _make_vec("TruckUnload-Single", 2)
    try:
        assert env._success_deliveries.tolist() == [4, 5]
        env._delivered_total.copy_(torch.tensor([[4, 5], [4, 4]]))
        env._update_success()
        assert env.last_success.tolist() == [True, False]
        env.reset(options={"reset_mask": np.array([True, False])})
        assert env.last_success.tolist() == [False, False]
    finally:
        env.close()


def test_step_embodiment_is_dropped_with_warning():
    env, base = _make_vec("TruckUnload-Single", 2)
    try:
        env.reset(embodiment=base)
        before = _link_lengths(env)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            env.step(_zero(env), embodiment=np.broadcast_to(env.offset_bounds[1], (2, env.n_offsets)))
        assert w, "a design handed to step must warn"
        np.testing.assert_allclose(_link_lengths(env), before, atol=1e-6)
    finally:
        env.close()


def test_validate_rejects_impossible_designs():
    env, _ = _make_vec("TruckUnload-Single", 2)
    try:
        assert not np.any(env.validate_embodiment(np.zeros(env.n_offsets + 1, np.float32)))
        nan = np.zeros(env.n_offsets, np.float32)
        nan[0] = np.nan
        assert not np.any(env.validate_embodiment(nan))
        assert not np.any(env.validate_embodiment(np.full(env.n_offsets, 1.5, np.float32)))
        with pytest.raises(ValueError):
            env.reset(embodiment=np.full(env.n_offsets, 1.5, np.float32))
        assert np.all(env.validate_embodiment(env.offset_bounds[0]))
    finally:
        env.close()


def test_make_single_env():
    env, base = cg.make("TruckUnload-Single", **CPU)
    try:
        obs, rejected, _ = env.reset(embodiment=base)
        assert obs.shape == (env.obs_dim(),)  # leading batch axis stripped
        assert rejected is False
        obs, reward, terminated, truncated, _ = env.step(np.zeros(env.act_dim(), np.float32))
        assert obs.shape == (env.obs_dim(),) and isinstance(reward, float)
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Multi: a red shelf stacker and a blue floor-area stacker
# --------------------------------------------------------------------------- #
def _place_target(env, r, pos):
    """Teleport robot ``r``'s current target box to ``pos``, at rest."""
    b = int(env._tgt[0, r])
    assert b >= 0, "robot has no target"
    qpos, qvel = wp.to_torch(env.gpu_data.qpos), wp.to_torch(env.gpu_data.qvel)
    a, d = int(env._box_qadr[b]), int(env._box_dadr[b])
    qpos[:, a:a + 3] = torch.as_tensor(pos, dtype=qpos.dtype)
    qpos[:, a + 3:a + 7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    qvel[:, d:d + 6] = 0.0
    return b


def test_multi_colours_and_targets():
    from envs.truck_unload.container import BOX_RGB, COLOR_NAMES

    env, _ = _make_vec("TruckUnload-Multi", 1)
    try:
        lay = env.layout
        assert [p.kind for p in lay.platforms] == ["site", "region"]
        red, blue = BOX_RGB[COLOR_NAMES["red"]], BOX_RGB[COLOR_NAMES["blue"]]
        for name, owner in zip(lay.box_names, lay.box_color):
            rgb = tuple(np.round(env.m.geom(f"{name}_geom").rgba[:3], 6))
            assert rgb == (red if owner == 0 else blue)
        # 2x2x2 container split evenly, plus one waiting box per stacker.
        assert np.sum(lay.box_color == 0) == 5 and np.sum(lay.box_color == 1) == 5
    finally:
        env.close()


def test_floor_area_delivery_anywhere_on_it():
    env, _ = _make_vec("TruckUnload-Multi", 1)
    try:
        env.reset()
        env.step(_zero(env))
        region = env.layout.platforms[1]
        lo, hi = region.region_lower, region.region_upper
        # Off-centre, but on the area: region-only delivery.
        pos = [lo[0] + 0.3 * (hi[0] - lo[0]), lo[1] + 0.7 * (hi[1] - lo[1]), lo[2] + env.box_size / 2.0 + 1e-3]
        b = _place_target(env, 2, pos)
        for _ in range(env.goal_dwell_steps + 5):
            env.step(_zero(env))
        assert int(env.boxes_delivered[0, 2]) >= 1
        assert int(env._box_state[0, b]) == 1
        # The area's goal site does not move.
        np.testing.assert_allclose(wp.to_torch(env.gpu_model.site_pos)[0, env._goal_sid[2]].numpy(), env.m.site_pos[env._goal_sid[2]], atol=1e-6)
    finally:
        env.close()


def test_shelf_goal_rises_after_delivery():
    env, _ = _make_vec("TruckUnload-Multi", 1)
    try:
        env.reset()
        env.step(_zero(env))
        sid = env._goal_sid[1]
        goal0 = wp.to_torch(env.gpu_data.site_xpos)[0, sid].numpy().copy()
        _place_target(env, 1, goal0)
        for _ in range(env.goal_dwell_steps + 5):
            env.step(_zero(env))
        assert int(env.boxes_delivered[0, 1]) >= 1
        rise = wp.to_torch(env.gpu_model.site_pos)[0, sid].numpy() - env.m.site_pos[sid]
        np.testing.assert_allclose(rise, [0.0, 0.0, env.layout.platforms[0].goal_step], atol=1e-5)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PRESET_IDS)
def test_every_corner_design_spawns_inside_its_field(env_id):
    """Fields are fixed; the presets size them so no design starts out of bounds."""
    import itertools
    corners = np.array(list(itertools.product([0, 1], repeat=3)))
    env, _ = _make_vec(env_id, len(corners))
    try:
        lo, hi = env.offset_bounds
        design = np.stack([np.where(np.tile(c, env.n_robots) == 1, hi, lo) for c in corners]).astype(np.float32)
        env.reset(embodiment=design)
        _, _, terminated, _, _ = env.step(_zero(env))
        assert not bool(env.last_left_field.any())
        assert not bool(terminated.any())
    finally:
        env.close()


def test_floor_area_height_limit():
    env, _ = _make_vec("TruckUnload-Multi", 1)
    try:
        region = env.layout.platforms[1]
        top = region.region_lower[2]
        np.testing.assert_allclose(region.region_upper[2] - top, 3.0 * env.box_size)
    finally:
        env.close()


def test_floor_area_claws_back_a_box_that_leaves():
    env, _ = _make_vec("TruckUnload-Multi", 1)
    try:
        env.reset()
        env.step(_zero(env))
        region = env.layout.platforms[1]
        lo, hi = region.region_lower, region.region_upper
        b = _place_target(env, 2, [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2] + env.box_size / 2.0 + 1e-3])
        paid = 0.0
        for _ in range(env.goal_dwell_steps + 5):
            _, _, terminated, _, info = env.step(_zero(env))
            paid += float(info["reward_per_robot"][0, 2])
        assert int(env.boxes_delivered[0, 2]) == 1 and int(env._box_state[0, b]) == 1
        # Knock it out of the area onto the belt: still in the stacker's field,
        # so live again, not lost.
        booked = float(env._paid[0, 2, b])                            # what the stacker was paid for this box
        qpos, qvel = wp.to_torch(env.gpu_data.qpos), wp.to_torch(env.gpu_data.qvel)
        a, d = int(env._box_qadr[b]), int(env._box_dadr[b])
        belt_top = env.layout.conveyor.upper[2]
        qpos[:, a:a + 3] = torch.tensor([float(env.layout.field_lo[2][0]) + 0.5, float(env.layout.conveyor.goal_pos[1]), float(belt_top) + env.box_size / 2.0 + 1e-3])
        qvel[:, d:d + 6] = 0.0
        _, _, terminated, _, info = env.step(_zero(env))
        assert not bool(terminated[0]), "a box knocked back onto the belt is live, not lost"
        assert int(info["boxes_undelivered"][0, 2]) == 1
        assert int(env._box_state[0, b]) == 0
        assert int(env.boxes_delivered[0, 2]) == 0
        # Exactly what delivering it paid comes back out: the pair nets zero.
        share = float(env._box_return[2])
        assert float(info["reward_per_robot"][0, 2]) == pytest.approx(-booked, abs=1e-6)
        assert float(env._paid[0, 2, b]) == 0.0
        assert paid <= share + 1e-5
    finally:
        env.close()


def _box_on_goal_one_step_from_scoring(env):
    """Put every world's loader box on its goal with dwell one short of the limit."""
    z = torch.full((env.num_envs, env.act_dim()), -1.0)   # grasp off
    for _ in range(3):
        env.step(z)
    q = env.qpos_torch()
    for w in range(env.num_envs):
        t = int(env._tgt[w, 0])
        a = int(env._box_qadr[t])
        q[w, a:a + 3] = env._goal_pos[w, 0] + torch.tensor([0.1, 0.0, 0.0])
        env._dwell[w, 0] = int(env.goal_dwell_steps) - 1
    return z


def test_partial_reset_leaves_other_worlds_task_state_alone():
    """reset(reset_mask) runs the task kernels only on the reset worlds: an untouched
    world's dwell must not tick during the reset call, so the delivery it completes on
    the next step is scored and paid rather than silently consumed."""
    env, _ = _make_vec("TruckUnload-Single", 2)
    try:
        env.reset()
        z = _box_on_goal_one_step_from_scoring(env)
        before = (env._dwell[1].clone(), env._scored[1].clone(), int(env._loader_idx[1]), env._tgt[1].clone())
        env.reset(options={"reset_mask": np.array([True, False])})
        after = (env._dwell[1], env._scored[1], int(env._loader_idx[1]), env._tgt[1])
        assert torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])
        assert before[2] == after[2] and torch.equal(before[3], after[3])
        _, _, _, _, info = env.step(z)
        assert int(info["boxes_delivered"][1].sum()) == 1
    finally:
        env.close()


def test_partial_reset_keeps_pending_autoresets():
    """A world that ended on the last step still autoresets on the next one when a
    partial reset names only other worlds."""
    env, _ = _make_vec("TruckUnload-Single", 2)
    try:
        env.reset()
        z = torch.zeros(2, env.act_dim())
        for _ in range(3):
            env.step(z)
        env.reset_mask[1] = True                      # as a terminal step leaves it
        env.reset(options={"reset_mask": np.array([True, False])})
        assert env.reset_mask.tolist() == [False, True]            # reset world set up; the ended one still owed
        _, reward, _, _, _ = env.step(z)
        # NEXT_STEP: world 0 takes its first step, world 1 spends this one respawning
        assert int(env.world_steps[0]) == env.phys_substeps and int(env.world_steps[1]) == 0
        assert float(reward[1]) == 0.0
    finally:
        env.close()


def test_carry_pays_only_new_records():
    """Carry shaping pays for a box getting closer to its goal than it has been since
    the target was set; carry -> release -> push back -> re-carry pays nothing more
    until the old record is passed."""
    env, _ = _make_vec("TruckUnload-Single", 1)
    try:
        env.reset()
        for t in (env._undeliv, env._close, env._away):
            t.zero_()
        env._has_tgt.fill_(1.0)
        env._d_tip.fill_(0.1)
        scale = float(env.carry_weight * env._box_return[0])        # per robot; robot 0 is the loader
        paid = []
        for rebase, grasped, d_goal in ((1, 1, 1.0), (0, 1, 0.5), (0, 0, 1.0), (0, 1, 0.5), (0, 1, 0.3)):
            env._rebase_t.fill_(rebase)
            env._grasped.fill_(float(grasped))
            env._d_goal.fill_(d_goal)
            env._reward_terminated_truncated()
            paid.append(float(env.last_reward_per_robot[0, 0]))
        assert paid[1] == pytest.approx(0.5 * scale)          # first carry: 1.0 -> 0.5
        assert paid[2] == pytest.approx(0.0, abs=1e-6)        # pushed back while not grasped: reach term only (tip still)
        assert paid[3] == pytest.approx(0.0, abs=1e-6)        # re-carry over covered ground: nothing
        assert paid[4] == pytest.approx(0.2 * scale)          # past the record: 0.5 -> 0.3
    finally:
        env.close()


def test_observations_stay_inside_the_declared_space():
    """The declared obs bounds are the true ones (finite only for quaternions and
    flags), so real observations -- including a stretched design's -- fit them."""
    env, base = _make_vec("TruckUnload-Multi", 2)
    try:
        _, hi = env.offset_bounds
        obs, _, _ = env.reset(seed=0, embodiment=np.stack([base, hi]))
        rng = np.random.default_rng(0)
        for _ in range(40):
            a = torch.as_tensor(rng.uniform(-1, 1, (2, env.single_action_space.shape[0])), dtype=torch.float32)
            obs, *_ = env.step(a)
            assert all(env.single_observation_space.contains(row) for row in obs.cpu().numpy().astype(np.float32))
    finally:
        env.close()


def test_observation_space_declares_the_true_bounds():
    env, _ = _make_vec("TruckUnload-Single", 2)
    try:
        low, high = env._obs_bounds()
        sp = env.single_observation_space
        np.testing.assert_array_equal(sp.low, low)
        np.testing.assert_array_equal(sp.high, high)
    finally:
        env.close()


def test_sample_observations_returns_rows():
    env, _ = _make_vec("TruckUnload-Single", 2)
    try:
        rows = env.sample_observations(5)
        assert tuple(rows.shape) == (5, env.single_observation_space.shape[0])
    finally:
        env.close()


def test_env_offsets_use_the_max_offset_each_world_was_built_with():
    """A change_max_offset reaches a world only at its next respawn; until then its
    reported offsets are in the metres it was built with."""
    env, _ = _make_vec("TruckUnload-Single", 2)
    try:
        lo, hi = env.offset_bounds
        env.reset(embodiment=np.stack([hi, hi]))
        before = env.env_offsets.copy()
        env.change_max_offset(0.5 * env.max_offset)
        np.testing.assert_allclose(env.env_offsets, before)           # nothing rebuilt yet
    finally:
        env.close()


def test_reward_per_robot_is_zeroed_with_the_reward_on_divergence():
    """A diverged world earns nothing, so its per-robot split is zero too and still
    sums to the reward (a claw-back pending on it must not leak into info)."""
    env, _ = _make_vec("TruckUnload-Single", 2)
    try:
        env.reset()
        env._diverged[0] = True
        env._undeliv[0, 0] = 1                                        # a claw-back on the diverged world
        reward, _, _ = env._reward_terminated_truncated()
        split = env.last_reward_per_robot
        assert float(reward[0]) == 0.0 and torch.equal(split[0], torch.zeros_like(split[0]))
        assert torch.allclose(split.sum(1), reward)
    finally:
        env.close()


def test_a_delivered_box_knocked_out_of_its_field_is_clawed_back_before_it_is_lost():
    """Knocking a delivered box clean out of the field ends the episode (lost), but
    what delivering it paid still comes back first."""
    env, _ = _make_vec("TruckUnload-Multi", 1)
    try:
        env.reset()
        env.step(_zero(env))
        region = env.layout.platforms[1]
        lo, hi = region.region_lower, region.region_upper
        b = _place_target(env, 2, [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, lo[2] + env.box_size / 2.0 + 1e-3])
        for _ in range(env.goal_dwell_steps + 5):
            env.step(_zero(env))
        assert int(env._box_state[0, b]) == 1
        booked = float(env._paid[0, 2, b])
        qpos = wp.to_torch(env.gpu_data.qpos)
        a = int(env._box_qadr[b])
        qpos[:, a:a + 3] = torch.tensor([float(env.layout.field_lo[2][0]) + 0.5, 50.0, 1.0])   # far outside every field in y
        _, _, terminated, _, info = env.step(_zero(env))
        assert bool(terminated[0]) and int(env._box_state[0, b]) == 2
        assert float(info["reward_per_robot"][0, 2]) == pytest.approx(-booked, abs=1e-6)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ["TruckUnload-Single", "TruckUnload-Multi"])
def test_first_step_after_reset_pays_only_its_own_progress(env_id):
    """reset() starts the shaping from the respawn pose, so the first zero-action step pays a tiny reward (no jump
    from stale distances), and the same after a later reset."""
    env, _ = _make_vec(env_id, 2)
    try:
        z = torch.zeros(2, env.act_dim())
        env.reset(seed=0)
        first = env.step(z)[1]
        for _ in range(3):
            env.step(z)
        env.reset(seed=0)
        again = env.step(z)[1]
        assert first.abs().max() < 0.01
        torch.testing.assert_close(first, again)
    finally:
        env.close()


def _scene(env):
    belt = wp.to_torch(env.gpu_model.geom_surfacevel)[:, torch.as_tensor(env._belt_gids, dtype=torch.long), 0].clone()
    boxes = env.qpos_torch()[:, torch.as_tensor(env._box_qadr[:env._consts.n_grid])].clone()
    return belt, boxes


def test_seeded_scene_variation():
    """Each respawn draws its world's belt speed (one speed for the belt and both rollers, within the spread) and
    nudges the container's boxes in x and y by less than half the clearance. The seed replays the scene."""
    env, _ = _make_vec("TruckUnload-Multi", 4)
    try:
        env.reset(seed=0)
        belt, boxes = _scene(env)
        assert torch.allclose(belt, belt[:, :1].expand_as(belt))
        assert ((belt / env.belt_velocity - 1).abs() <= env.belt_velocity_spread + 1e-6).all()
        assert belt[:, 0].unique().numel() == 4
        grid = torch.as_tensor(env._init_qpos[env._box_qadr[:env._consts.n_grid]])
        nudge = boxes - grid
        assert nudge.abs().max() <= env.box_jitter + 1e-6 and nudge.abs().max() > 0
        env.reset(seed=0)
        again = _scene(env)
        assert torch.equal(belt, again[0]) and torch.equal(boxes, again[1])
        env.reset(seed=1)
        assert not torch.equal(belt, _scene(env)[0])
    finally:
        env.close()


def test_box_jitter_must_keep_boxes_apart():
    with pytest.raises(ValueError, match="box_jitter"):
        _make_vec("TruckUnload-Single", 1, box_jitter=0.5)


def test_constructor_defaults_are_the_single_preset():
    """Building the class directly gives the same env as make("TruckUnload-Single")."""
    import inspect
    from envs.truck_unload.config import SINGLE
    from envs.truck_unload.codesign_env import TruckUnloadCoDesignEnv
    from envs.truck_unload.truck_unload_env import TruckUnloadParallelEnv

    params = inspect.signature(TruckUnloadParallelEnv.__init__).parameters
    assert {k: params[k].default for k in SINGLE if k in params} == {k: v for k, v in SINGLE.items() if k in params}
    direct = TruckUnloadCoDesignEnv(2, device="cpu")
    made, _ = _make_vec("TruckUnload-Single", 2)
    try:
        torch.testing.assert_close(direct.reset(seed=0)[0], made.reset(seed=0)[0])
    finally:
        direct.close()
        made.close()


def test_step_needs_every_copy_reset_but_resets_may_be_partial():
    env, base = _make_vec()
    try:
        actions = torch.zeros((2, env.single_action_space.shape[0]))
        _, rejected, _ = env.reset(embodiment=base, options={"reset_mask": torch.tensor([True, False])})
        assert not rejected.any()
        with pytest.raises(RuntimeError, match=r"call reset\(\) before step\(\): copies \[1\]"):
            env.step(actions)
        env.reset(embodiment=base, options={"reset_mask": torch.tensor([False, True])})
        env.step(actions)
    finally:
        env.close()


def test_a_rejected_copy_can_be_reset_on_its_own(monkeypatch):
    env, base = _make_vec()
    try:
        compute = env._compute_obs

        def diverge_copy_1(*args, **kwargs):     # stands in for physics that blew up at the respawn
            obs = compute(*args, **kwargs)
            env._diverged[1] = True
            return obs

        monkeypatch.setattr(env, "_compute_obs", diverge_copy_1)
        _, rejected, _ = env.reset(embodiment=base)
        assert rejected.tolist() == [False, True]
        monkeypatch.undo()
        actions = torch.zeros((2, env.single_action_space.shape[0]))
        with pytest.raises(RuntimeError, match=r"copies \[1\]"):
            env.step(actions)
        _, rejected, _ = env.reset(embodiment=base, options={"reset_mask": torch.tensor([False, True])})
        assert not rejected.any()
        env.step(actions)
    finally:
        env.close()
