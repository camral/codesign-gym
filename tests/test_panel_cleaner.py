"""Tests for the PanelCleaner co-design environment.

Everything runs on the CPU warp device (``device="cpu", use_graph=False``): the
env is GPU-parallel in production, but its correctness -- the codesign contract,
the independent-leg embodiment, the buffer scaling and the body-panel fouling --
is device-independent and testable without CUDA. Panels are kept tiny so the
warp kernels compile once and the suite stays quick.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")

import torch  # noqa: E402
import warnings  # noqa: E402

import codesign_gym as cg  # noqa: E402

# Small, flat, single-cell panels: the embodiment and the machinery under test
# do not depend on the array's size, and this keeps every build cheap.
SMALL = dict(
    device="cpu", use_graph=False,
    panel_rows=1, panel_cols=1, n_sensor_rows=1, n_sensor_cols=1,
)

PRESET_IDS = ["SolarCleaner-Reach", "SolarCleaner-Stability", "SolarCleaner-Traverse"]

# Dog quadruped: 8 leg links (4 legs x upper+knee) x xyz = 24, plus the full arm
# (8 links) x xyz = 24.
N_OFFSETS = 48


def _make_vec(env_id="SolarCleaner-Reach", num_envs=2, **overrides):
    cfg = {**SMALL, **overrides}
    return cg.make_vec(env_id, num_envs, **cfg)


# --------------------------------------------------------------------------- #
# Registry + build
# --------------------------------------------------------------------------- #
def test_registry_lists_panel_cleaner_envs():
    ids = cg.list_envs()
    for env_id in PRESET_IDS:
        assert env_id in ids


@pytest.mark.parametrize("env_id", PRESET_IDS)
def test_preset_builds_resets_steps(env_id):
    env, base = _make_vec(env_id, 2)
    try:
        assert base.shape == (N_OFFSETS,)
        assert env.embodiment_space.shape == (N_OFFSETS,)
        out = env.reset(embodiment=base)
        assert len(out) == 3, "vectorized reset must be (obs, rejected, info)"
        obs, rejected, info = out
        assert obs.shape[0] == 2
        assert rejected.dtype == torch.bool and rejected.shape == (2,)
        assert not rejected.any()
        actions = torch.zeros((2, env.act_dim()))
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(actions)
        assert reward.shape == (2,)
        assert terminated.shape == (2,)
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Independent-leg embodiment
# --------------------------------------------------------------------------- #
def test_independent_legs_embodiment():
    import mujoco
    from envs.panel_cleaner.offsettable_logic import n_slots

    env, base = _make_vec("SolarCleaner-Reach", 1)
    try:
        assert env.n_offsets == N_OFFSETS
        seg = env._segs
        names = [mujoco.mj_id2name(env.m, mujoco.mjtObj.mjOBJ_BODY, int(b))
                 for b in seg["body"]]
        legs = [n for n in names if "leg" in n or "knee" in n]
        arm = [n for n in names if "leg" not in n and "knee" not in n]
        assert len(legs) == 8, "each of 4 dog legs contributes an upper + knee link"
        assert len(arm) == 8   # the whole arm: mast, link0, shoulder, link1-5
        # Every offsettable link moves its end along all three axes (bidirectional), none shared.
        assert (seg["axis_num"] == 3).all()
        assert all(n.endswith("_offsettable_xyz") for n in names)
        assert all("_equ" not in n for n in names)
        assert n_slots(seg) == N_OFFSETS   # 16 links x 3 axes = 48
        lo, hi = env.offset_bounds
        assert (lo < 0.0).any() and (hi > 0.0).all()
    finally:
        env.close()


def test_default_robot_is_dog_quadruped():
    import mujoco

    env, base = _make_vec("SolarCleaner-Reach", 1)
    try:
        m = env.m
        assert env.act_dim() == 18          # 12 leg + 6 arm velocity actuators
        assert m.geom("torso_geom").id >= 0       # dog chassis, not the ant sphere
        for foot in ("fl_knee_geom", "fr_knee_geom", "bl_knee_geom", "br_knee_geom"):
            assert m.geom(foot).id >= 0     # the four knee feet
        with pytest.raises(KeyError):
            m.geom("base_geom")             # the ant's base geom is gone
    finally:
        env.close()


def test_ant_base_still_selectable():
    # The swap changes the default; ant.xml stays available via kwarg.
    env, base = _make_vec("SolarCleaner-Reach", 1, ant_xml="panel_clean/ant.xml")
    try:
        out = env.reset(embodiment=base)
        assert len(out) == 3
        env.step(torch.zeros((1, env.act_dim())))
    finally:
        env.close()


def test_per_world_designs_differ():
    env, base = _make_vec("SolarCleaner-Reach", 2)
    try:
        lo, hi = env.offset_bounds
        emb = np.stack([
            np.zeros(env.n_offsets, np.float32),
            np.clip(0.6 * np.ones(env.n_offsets, np.float32), lo, hi),
        ])
        env.reset(embodiment=emb)
        applied = env.env_offsets_norm
        assert not np.allclose(applied[0], applied[1])
    finally:
        env.close()


def test_masked_reset_changes_only_selected_worlds():
    env, base = _make_vec("SolarCleaner-Reach", 2)
    try:
        lo, hi = env.offset_bounds
        initial = np.stack([base, np.clip(base + 0.25, lo, hi)])
        env.reset(embodiment=initial)
        env.world_steps[:] = torch.tensor([8, 16])
        changed = initial.copy()
        changed[0] = np.clip(base - 0.25, lo, hi)
        obs, rejected, _ = env.reset(
            embodiment=changed, options={"reset_mask": [True, False]})
        assert obs.shape[0] == 2 and not rejected.any()
        assert np.allclose(env._current_embodiment.cpu()[0], changed[0])
        assert np.allclose(env._current_embodiment.cpu()[1], initial[1])
        assert env.world_steps.tolist() == [0, 16]
        with pytest.raises(ValueError, match="reset_mask"):
            env.reset(embodiment=changed, options={"reset_mask": [[True, False]]})
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Codesign contract: freeze + structural validation
# --------------------------------------------------------------------------- #
def test_step_embodiment_is_frozen_with_warning():
    env, base = _make_vec("SolarCleaner-Reach", 2)
    try:
        assert env.metadata.get("supports_transition_embodiment") is False
        env.reset(embodiment=base)
        actions = torch.zeros((2, env.act_dim()))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            env.step(actions, embodiment=np.broadcast_to(base, (2, env.n_offsets)))
        assert any("Transition" in str(w.message) for w in caught)
    finally:
        env.close()


def test_validate_embodiment_rejects_impossible():
    env, base = _make_vec("SolarCleaner-Reach", 2)
    try:
        lo, hi = env.offset_bounds
        # In-bounds design is accepted.
        assert np.all(env.validate_embodiment(base))
        # Out-of-bounds violates embodiment_space -> invalid; reset refuses it.
        assert not np.any(env.validate_embodiment(hi + 1.0))
        with pytest.raises(ValueError):
            env.reset(embodiment=hi + 1.0)
        # Wrong shape -> invalid.
        assert not np.any(env.validate_embodiment(np.zeros(env.n_offsets + 1, np.float32)))
        # Non-finite -> invalid, per world.
        nan = np.zeros((2, env.n_offsets), np.float32)
        nan[0, 0] = np.nan
        assert env.validate_embodiment(nan).tolist() == [False, True]
    finally:
        env.close()


def test_bad_but_in_bounds_design_is_accepted():
    # A design that cleans poorly (legs driven to their length floor) is valid;
    # only impossible designs are rejected.
    env, base = _make_vec("SolarCleaner-Reach", 2)
    try:
        lo, hi = env.offset_bounds
        bad = lo.copy()  # every link shrunk as far as the bounds allow
        assert np.all(env.validate_embodiment(bad))
        obs, rejected, info = env.reset(embodiment=bad)
        assert not rejected.any()
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Buffer scaling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("num_envs", [1, 2, 4])
def test_contact_pool_scales_with_num_envs(num_envs):
    env, base = _make_vec("SolarCleaner-Reach", num_envs)
    try:
        pool = env.gpu_data.contact.geom.shape[0]
        assert pool % num_envs == 0
        assert pool // num_envs == pool // num_envs  # integer per-world share
        # The total pool is per-world * num_envs, so it grows with the batch.
        assert pool >= 64 * num_envs
    finally:
        env.close()


def test_contact_pool_grows_when_body_collides():
    # Making the panels solid obstacles (body collision) raises the per-world
    # contact budget, so the pool is larger for the same batch.
    roller_only, _ = _make_vec("SolarCleaner-Reach", 1, collide_roller_only=True)
    body, _ = _make_vec("SolarCleaner-Reach", 1, collide_roller_only=False,
                        panel_rows=1, panel_cols=1)
    try:
        assert (body.gpu_data.contact.geom.shape[0]
                > roller_only.gpu_data.contact.geom.shape[0])
    finally:
        roller_only.close()
        body.close()


# --------------------------------------------------------------------------- #
# Body-panel fouling
# --------------------------------------------------------------------------- #
def _foul_env(num_envs=2):
    from envs.panel_cleaner.codesign_env import PanelCleanerCoDesignEnv
    return PanelCleanerCoDesignEnv(
        num_envs, device="cpu", use_graph=False,
        panel_rows=1, panel_cols=1, n_sensor_rows=1, n_sensor_cols=1,
        collide_roller_only=False, terminate_on_body_contact=True,
        body_contact_penalty=-5.0)


def test_body_contact_requires_body_collision():
    from envs.panel_cleaner.codesign_env import PanelCleanerCoDesignEnv
    with pytest.raises(ValueError):
        PanelCleanerCoDesignEnv(
            1, device="cpu", use_graph=False,
            collide_roller_only=True, terminate_on_body_contact=True)


def test_body_contact_kernel_detects_only_non_roller_plate():
    import warp as wp
    env = _foul_env(2)
    try:
        env.reset(embodiment=env.base_embodiment)
        plate_gid = int(np.flatnonzero(env._plate_mask_np)[0])
        base_gid = int(env.m.geom("torso_geom").id)
        geom = wp.to_torch(env.gpu_data.contact.geom)
        wid = wp.to_torch(env.gpu_data.contact.worldid)
        dist = wp.to_torch(env.gpu_data.contact.dist)
        nacon = wp.to_torch(env.gpu_data.nacon)
        deep = -(env.body_contact_depth + 0.01)   # firmly penetrating

        # A firm base-plate contact in world 1 is a foul.
        geom[0, 0], geom[0, 1], wid[0], dist[0] = plate_gid, base_gid, 1, deep
        nacon[0] = 1
        env._launch_body_contact()
        assert env._bad_contact.tolist() == [0, 1]

        # A roller-plate contact is cleaning, not a foul.
        geom[0, 1], dist[0] = env._roller_gid, deep
        nacon[0] = 1
        env._launch_body_contact()
        assert env._bad_contact.tolist() == [0, 0]

        # A mere graze (shallower than the threshold) is not a foul.
        geom[0, 1], dist[0] = base_gid, -(env.body_contact_depth * 0.1)
        nacon[0] = 1
        env._launch_body_contact()
        assert env._bad_contact.tolist() == [0, 0]
    finally:
        env.close()


def test_no_false_foul_during_normal_stepping():
    env = _foul_env(2)
    try:
        env.reset(embodiment=env.base_embodiment)
        actions = torch.zeros((2, env.act_dim()))
        for _ in range(10):
            env.step(actions)
            assert not bool(env.last_body_contact.any())
    finally:
        env.close()


def test_traverse_foul_terminates_without_penalty():
    # The Traverse preset ends the episode on a foul but adds no reward penalty
    # (deterrent = forfeiting the rest of the sweep, not a flat cost), so an
    # attempt never scores below doing nothing.
    env, base = _make_vec("SolarCleaner-Traverse", 2, panel_rows=1, panel_cols=1,
                          n_sensor_rows=1, n_sensor_cols=1)
    try:
        assert env.terminate_on_body_contact is True
        assert env.body_contact_penalty == 0.0
        env.reset(embodiment=base)
        env.step(torch.zeros((2, env.act_dim())))
        env._bad_contact[1] = 1  # world 1 fouls; world 0 does not
        reward, terminated, truncated = env._reward_terminated_truncated()
        assert bool(terminated[1]) and not bool(terminated[0])
        # No penalty: the fouled world's reward matches the identical unfouled
        # world's (both zero-action), i.e. nothing was subtracted for the foul.
        assert float(reward[1]) == float(reward[0])
    finally:
        env.close()


def test_foul_terminates_and_penalizes():
    env = _foul_env(2)
    try:
        env.reset(embodiment=env.base_embodiment)
        env.step(torch.zeros((2, env.act_dim())))
        env._bad_contact[1] = 1  # world 1 fouled this step
        reward, terminated, truncated = env._reward_terminated_truncated()
        assert bool(terminated[1]) and not bool(terminated[0])
        assert float(reward[1]) <= env.body_contact_penalty + 1e-5
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Terrain
# --------------------------------------------------------------------------- #
def _terrain_env(num_envs=2, **overrides):
    # The Stability preset with a small array + coarse grid, terrain kept on.
    cfg = dict(panel_rows=1, panel_cols=1, n_sensor_rows=1, n_sensor_cols=1,
               terrain_res=(16, 16))
    cfg.update(overrides)
    return _make_vec("SolarCleaner-Stability", num_envs, **cfg)


def _hfield_slice(env, w):
    import warp as wp
    nr, nc = int(env.terrain_res[0]), int(env.terrain_res[1])
    hd = wp.to_torch(env.gpu_model.hfield_data)
    adr = int(wp.to_torch(env.gpu_model.hfield_adr)[w])
    return hd[adr:adr + nr * nc].clone()


def test_terrain_is_per_world_hfield():
    import mujoco
    env, base = _terrain_env(2)
    try:
        # CPU render model remains compact (compiled once), while the one uploaded model owns
        # the per-world asset union and points each world at its own slice.
        assert env.m.nhfield == 1
        assert env.gpu_model.nhfield == 2
        assert tuple(env.gpu_model.hfield_adr.shape) == (2,)
        assert env.m.geom("terrain_geom").type == mujoco.mjtGeom.mjGEOM_HFIELD
        import warp as wp
        dataid = wp.to_torch(env.gpu_model.geom_dataid)[:, env._terrain_gid]
        assert dataid.tolist() == [0, 1]
        env.reset(embodiment=base)
        actions = torch.zeros((2, env.act_dim()))
        for _ in range(5):
            env.step(actions)
        w0, w1 = _hfield_slice(env, 0), _hfield_slice(env, 1)
        assert not torch.allclose(w0, w1)            # per-world different
        assert float(w0.max() - w0.min()) > 0.0      # genuinely uneven
    finally:
        env.close()


def test_stability_preset_uses_collision_safe_terrain_resolution():
    from envs.panel_cleaner.config import STABILITY

    assert STABILITY["terrain_res"] == (16, 16)


def test_terrain_regenerates_each_episode():
    env, base = _terrain_env(2)
    try:
        env.reset(embodiment=base)
        actions = torch.zeros((2, env.act_dim()))
        for _ in range(5):
            env.step(actions)
        before = _hfield_slice(env, 0)
        ep_before = int(env._ep_count[0])
        # Reset only world 0; its terrain must change, world 1's must not.
        w1_before = _hfield_slice(env, 1)
        env.reset_mask[0], env.reset_mask[1] = True, False
        env._reset_pending = True                       # as a terminal step leaves it
        env.step(actions)
        assert int(env._ep_count[0]) > ep_before
        assert not torch.allclose(before, _hfield_slice(env, 0))   # fresh terrain
        assert torch.allclose(w1_before, _hfield_slice(env, 1))    # others untouched
    finally:
        env.close()


def test_midpoints_follow_the_terrain_with_their_panels():
    """A midpoint stays halfway in z between the cells either side of it once the
    terrain has re-seated their panels, in every world and every episode."""
    import warp as wp
    env, base = _terrain_env(2, panel_cols=2, midpoints=True)
    try:
        order = env.grid.goals(env.panel_traversal)
        mids = [j for j, g in enumerate(order) if g.is_midpoint]
        assert mids
        for _ in range(2):
            env.reset(embodiment=base)
            xpos = wp.to_torch(env.gpu_data.site_xpos)
            for j in mids:
                sid, so, si = (env.m.site(order[k].site).id for k in (j, j - 1, j + 1))
                torch.testing.assert_close(xpos[:, sid, 2], 0.5 * (xpos[:, so, 2] + xpos[:, si, 2]), atol=1e-5, rtol=0)
                assert not torch.allclose(xpos[:, sid, 2], torch.full_like(xpos[:, sid, 2], float(env.m.site_pos[sid][2])))
    finally:
        env.close()


def test_roughness_raises_high_frequency_detail():
    # Pin a low base frequency and a fine grid so the octaves are well resolved
    # (a high base_freq near the grid's Nyquist blurs the gain -> detail link).
    common = dict(terrain_res=(32, 32), terrain_base_freq=2.0, terrain_octaves=5)
    smooth, _ = _terrain_env(1, terrain_roughness=0.2, **common)
    rough, _ = _terrain_env(1, terrain_roughness=0.85, **common)
    try:
        for e in (smooth, rough):
            e.reset(embodiment=e.base_embodiment)
            e.step(torch.zeros((1, e.act_dim())))
        nr, nc = 32, 32
        gs = _hfield_slice(smooth, 0).reshape(nr, nc)
        gr = _hfield_slice(rough, 0).reshape(nr, nc)
        # Higher fBm gain => more cell-to-cell variation (rockier).
        assert (gr.diff(dim=0).abs().mean() + gr.diff(dim=1).abs().mean()) > \
               (gs.diff(dim=0).abs().mean() + gs.diff(dim=1).abs().mean())
    finally:
        smooth.close()
        rough.close()


def test_grown_legs_spawn_clear_of_terrain():
    # The base is dropped high enough that even legs grown to the maximum spawn
    # above the ground, not buried in a hill -- and the world stays finite.
    import warp as wp
    env, base = _terrain_env(2, max_offset=0.4)
    try:
        lo, hi = env.offset_bounds
        emb = np.zeros((2, env.n_offsets), np.float32)
        emb[1, 0:24] = hi[0:24]                     # world 1: legs grown to max
        env.reset(embodiment=emb)
        env.step(torch.zeros((2, env.act_dim())))  # apply spawn lift
        m = env.m
        feet = [m.geom(n).id for n in ("fl_knee_geom", "fr_knee_geom",
                                       "bl_knee_geom", "br_knee_geom")]
        gp = wp.to_torch(env.gpu_data.geom_xpos)
        nr = nc = int(env.terrain_res[0])
        hd = wp.to_torch(env.gpu_model.hfield_data)
        adr = wp.to_torch(env.gpu_model.hfield_adr)
        size2 = wp.to_torch(env.gpu_model.hfield_size)
        (x0, x1), (y0, y1) = env.terrain_region

        def terr(w, x, y):
            c = int(np.clip((x - x0) / (x1 - x0), 0, 1) * (nc - 1) + 0.5)
            r = int(np.clip((y - y0) / (y1 - y0), 0, 1) * (nr - 1) + 0.5)
            elev = float(hd[int(adr[w]) + r * nc + c])
            return env._terrain_base_h + elev * float(size2[w][2])

        for w in range(2):
            clr = min(float(gp[w, f, 2]) - terr(w, float(gp[w, f, 0]), float(gp[w, f, 1]))
                      for f in feet)
            assert clr > 0.0, f"world {w} feet spawned buried (clearance {clr:.3f})"
        for _ in range(60):
            env.step(torch.zeros((2, env.act_dim())))
        assert bool(torch.isfinite(env.qpos_torch()).all())
    finally:
        env.close()


def test_panels_sit_on_terrain():
    import warp as wp
    env, base = _terrain_env(2)
    try:
        env.reset(embodiment=base)
        actions = torch.zeros((2, env.act_dim()))
        for _ in range(5):
            env.step(actions)
        bp = wp.to_torch(env.gpu_model.body_pos)
        pbid = wp.to_torch(env._panel_bid_wp)
        base_h, peak = env.terrain_height
        for w in range(2):
            for b in pbid:
                z = float(bp[w, int(b), 2])
                assert base_h <= z <= base_h + (peak - base_h) + 1e-4
        # At least one panel is actually lifted off the base plane.
        assert any(float(bp[w, int(b), 2]) > base_h + 1e-3
                   for w in range(2) for b in pbid)
    finally:
        env.close()


def test_support_posts_are_solid_not_ghosts():
    import mujoco
    env, base = _make_vec("SolarCleaner-Reach", 1)
    try:
        m = env.m
        for suffix in ("_column", "_elbow", "_stub"):
            gid = None
            for gi in range(m.ngeom):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
                if name.endswith(suffix):
                    gid = gi
                    break
            assert gid is not None, f"no {suffix} geom found"
            # Collidable (some contype/conaffinity bit set), so not a ghost...
            assert m.geom_contype[gid] or m.geom_conaffinity[gid]
            # ...but not a panel plate, so it never triggers a foul.
            assert env._plate_mask_np[gid] == 0
    finally:
        env.close()


def test_leg_noise_applies_random_force_to_legs():
    import warp as wp
    env, base = _terrain_env(2, leg_noise_mag=20.0)
    try:
        assert env._n_leg_noise == 4  # the four knee (lower-leg) bodies
        env.reset(embodiment=base)
        env.step(torch.zeros((2, env.act_dim())))
        x = wp.to_torch(env.gpu_data.xfrc_applied)
        # Every leg body carries a nonzero force, and the two worlds differ.
        w0 = [x[0, int(b), :3].abs().sum().item() for b in env._leg_noise_bids_np]
        assert all(v > 0 for v in w0)
        assert any(not torch.allclose(x[0, int(b)], x[1, int(b)])
                   for b in env._leg_noise_bids_np)
    finally:
        env.close()


def test_off_green_terminates():
    # Walking off the terrain patch ends the episode; staying on it does not.
    env, base = _terrain_env(2)
    try:
        assert env.terminate_off_green is True
        env.reset(embodiment=base)
        env.step(torch.zeros((2, env.act_dim())))
        q = env.qpos_torch()
        q[1, env._base_qadr] = 999.0     # shove world 1 far off the green
        reward, terminated, truncated = env._reward_terminated_truncated()
        assert bool(terminated[1]) and not bool(terminated[0])
        assert env.last_off_green.tolist() == [False, True]
    finally:
        env.close()


LEG_LENGTH = [2, 5, 8, 11, 14, 17, 20, 23]   # the slot of each leg link that lengthens it


def test_sunk_base_terminates_relative_to_design():
    # The reference is each design's standing height, so a longer-legged design
    # must hold its base higher; sinking below the fraction ends the episode.
    env, base = _make_vec("SolarCleaner-Reach", 2)
    try:
        assert env.min_base_height_frac == 0.75
        long_legs = base.copy()
        long_legs[LEG_LENGTH] = 1.0
        env.reset(embodiment=[base, long_legs])
        h0 = env._spawn_h.clone()
        assert h0[1] > h0[0] + 0.1
        env.step(torch.zeros((2, env.act_dim())))
        q = env.qpos_torch()
        # Both bases at the base design's standing height: fine for it, too low for the long legs
        q[:, env._base_qadr + 2] = 0.8 * h0[0]
        reward, terminated, truncated = env._reward_terminated_truncated()
        assert env.last_too_low.tolist() == [False, True]
        assert bool(terminated[1]) and not bool(terminated[0])
    finally:
        env.close()


def test_sunk_base_check_can_be_disabled():
    env, base = _make_vec("SolarCleaner-Reach", 1, min_base_height_frac=0.0)
    try:
        env.reset(embodiment=base)
        env.step(torch.zeros((1, env.act_dim())))
        env.qpos_torch()[0, env._base_qadr + 2] = 0.05
        env._reward_terminated_truncated()
        assert not bool(env.last_too_low[0])
    finally:
        env.close()


def test_sunk_base_measured_above_terrain():
    # On terrain the reference is still the flat-ground standing height, and the
    # height is taken above the hill under the base, not above z = 0.
    env, base = _terrain_env(1)
    flat, _ = _make_vec("SolarCleaner-Reach", 1)
    try:
        env.reset(embodiment=base)
        flat.reset(embodiment=base)
        assert torch.allclose(env._spawn_h, flat._spawn_h, atol=1e-4)
        q = env.qpos_torch()
        z = q[0, env._base_qadr + 2].item()
        ground = z - env._base_height()[0].item()
        assert ground >= env.terrain_height[0] - 1e-4      # the terrain, not the floor
        q[0, env._base_qadr + 2] = ground + 0.9 * env._spawn_h[0]
        env._reward_terminated_truncated()
        assert not bool(env.last_too_low[0])
        q[0, env._base_qadr + 2] = ground + 0.6 * env._spawn_h[0]
        env._reward_terminated_truncated()
        assert bool(env.last_too_low[0])
    finally:
        env.close()
        flat.close()


def test_terrain_valleys_sit_above_the_floor():
    import warp as wp
    env, base = _terrain_env(1)
    try:
        env.reset(embodiment=base)
        env.step(torch.zeros((1, env.act_dim())))
        nr = nc = int(env.terrain_res[0])
        adr = int(wp.to_torch(env.gpu_model.hfield_adr)[0])
        hd = wp.to_torch(env.gpu_model.hfield_data)[adr:adr + nr * nc]
        s2 = float(wp.to_torch(env.gpu_model.hfield_size)[0][2])
        valley = env._terrain_base_h + float(hd.min()) * s2
        assert valley > 0.05, f"lowest valley at z={valley:.3f}, not above the floor"
    finally:
        env.close()


def test_terrain_off_has_no_hfield():
    env, base = _make_vec("SolarCleaner-Reach", 2)
    try:
        assert env.m.nhfield == 0
        assert env._terrain_gid == -1
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Single-env wrapper + space padding
# --------------------------------------------------------------------------- #
def test_make_single_env_unbatched():
    env, base = cg.make("SolarCleaner-Reach", **SMALL)
    try:
        obs, rejected, info = env.reset(embodiment=base)
        assert obs.dim() == 1  # leading batch axis stripped
    finally:
        env.close()


def test_partial_reset_keeps_pending_autoresets():
    """A world that ended on the last step still autoresets on the next one when a
    partial reset names only other worlds."""
    env, _ = _make_vec("SolarCleaner-Reach", 2)
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


@pytest.mark.parametrize("env_id", ["SolarCleaner-Traverse", "SolarCleaner-Stability"])
def test_reset_returns_the_state_the_first_step_starts_from(env_id):
    """reset() sets the episode up completely (spawn jitter, terrain, seat on it), so
    the first step continues from the returned pose instead of respawning again."""
    env, _ = _make_vec(env_id, 2)
    try:
        env.reset(seed=0)
        assert not env.reset_mask.any()
        q0 = env.qpos_torch()[:, env._base_qadr:env._base_qadr + 3].clone()
        env.step(torch.zeros(2, env.act_dim()))
        q1 = env.qpos_torch()[:, env._base_qadr:env._base_qadr + 3]
        assert float((q1[:, :2] - q0[:, :2]).abs().max()) < 0.02      # no re-drawn spawn jitter
        assert float((q1[:, 2] - q0[:, 2]).abs().max()) < 0.1         # no re-seat on new terrain
    finally:
        env.close()


def test_observations_stay_inside_the_declared_space():
    """The declared obs bounds are the true ones (finite only for quaternions and
    flags), so real observations -- including a stretched design's -- fit them."""
    env, base = _make_vec("SolarCleaner-Stability", 2)
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


def test_sample_observations_returns_rows():
    env, _ = _make_vec("SolarCleaner-Reach", 2)
    try:
        rows = env.sample_observations(5)
        assert tuple(rows.shape) == (5, env.single_observation_space.shape[0])
    finally:
        env.close()


def test_obs_after_a_clean_shows_the_next_goal():
    """On the step a cell is cleaned, the returned obs already describes the goal the
    world moved on to (what the next reward pays for), matching a fresh obs."""
    env, _ = _make_vec("SolarCleaner-Reach", 2, panel_cols=2)             # two cells: the clean leaves a goal to show
    try:
        assert env.n_goals > 1
        env.reset(seed=0)
        env.step(torch.zeros(2, env.act_dim()))                            # clears the reset's rebase
        env._dwell[0] = int(env._goal_dwell[env._goal_idx[0].clamp(max=env.n_goals - 1)]) + 1   # dwell banked above decay: cleans this step
        cleaned_before = int(env._goal_idx[0])
        obs, _, terminated, truncated, _ = env.step(torch.zeros(2, env.act_dim()))
        if not bool(env.last_cleaned[0]) or bool(terminated[0] | truncated[0]):
            raise AssertionError("the scripted clean did not happen")
        assert int(env._goal_idx[0]) == cleaned_before + 1
        env.reset_mask.zero_()
        fresh = env._compute_obs()
        g = int(env.m.nq + env.m.nv) + 10
        np.testing.assert_allclose(obs[0, g:g + 9].cpu().numpy(), fresh[0, g:g + 9].cpu().numpy(), atol=1e-5)
    finally:
        env.close()


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
