"""Tests for the BallCatcher co-design environment (Rocky robot).

Everything runs on the CPU warp device (``device="cpu", use_graph=False``): the
env is GPU-parallel in production, but its correctness -- the codesign contract,
the 3D-offsettable embodiment, the space sizing and the robot swap -- is
device-independent and testable without CUDA. The two existing ball_catcher
suites (contact timing, pitch arrival) pin ``device="cuda"``; this one covers the
Rocky swap and the three-axis offset shape on the CPU device instead.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")

import torch  # noqa: E402
import warnings  # noqa: E402
import warp as wp  # noqa: E402

import codesign_gym as cg  # noqa: E402

SMALL = dict(device="cpu", use_graph=False)

PRESET_IDS = ["BallCatcher-Pitch", "BallCatcher-Arc", "BallCatcher-Fielding"]

# Rocky's offsettable links: 6 legs (3 x hip+ankle) + 1 torso + 8 arm links
# (link2/3/4/6 on both arms), each moving its far end along xyz -> 3 slots.
N_OFFSETS = 45
ACT_DIM = 19       # 6 leg + 13 arm velocity actuators
OBS_DIM = 70       # nq_robot(26) + nv_robot(25) + 19; the design is not observed


def _make_vec(env_id="BallCatcher-Pitch", num_envs=2, **overrides):
    cfg = {**SMALL, **overrides}
    return cg.make_vec(env_id, num_envs, **cfg)


# --------------------------------------------------------------------------- #
# Registry + build
# --------------------------------------------------------------------------- #
def test_registry_lists_ball_catcher_envs():
    ids = cg.list_envs()
    for env_id in PRESET_IDS:
        assert env_id in ids


@pytest.mark.parametrize("env_id", PRESET_IDS)
def test_preset_builds_resets_steps(env_id):
    env, base = _make_vec(env_id, 2)
    try:
        assert base.shape == (N_OFFSETS,)
        assert env.embodiment_space.shape == (N_OFFSETS,)
        assert env.act_dim() == ACT_DIM
        assert env.obs_dim() == OBS_DIM
        out = env.reset(embodiment=base)
        assert len(out) == 3, "vectorized reset must be (obs, rejected, info)"
        obs, rejected, info = out
        assert obs.shape == (2, OBS_DIM)
        rejected = np.asarray(rejected)
        assert rejected.shape == (2,)                 # one flag per world
        assert not rejected.any()
        actions = torch.zeros((2, env.act_dim()))
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(actions)
        assert reward.shape == (2,)
        assert terminated.shape == (2,)
        assert np.isfinite(np.asarray(obs)).all()
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# 3D-offsettable embodiment
# --------------------------------------------------------------------------- #
def test_rocky_links_are_three_axis_offsettable():
    import mujoco
    from envs.ball_catcher.offsettable_logic import n_slots

    env, base = _make_vec("BallCatcher-Pitch", 1)
    try:
        assert env.n_offsets == N_OFFSETS
        seg = env._segs
        # Every offsettable link moves its end along all three axes: 15 links,
        # 3 slots each, none sharing an _equ group.
        assert len(seg["body"]) == 15
        assert (seg["axis_num"] == 3).all()
        assert n_slots(seg) == N_OFFSETS
        names = [mujoco.mj_id2name(env.m, mujoco.mjtObj.mjOBJ_BODY, int(b))
                 for b in seg["body"]]
        legs = [n for n in names if "leg" in n or "knee" in n]
        torso = [n for n in names if "torso_arm" in n]
        arm = [n for n in names if n.startswith("link")]
        assert len(legs) == 6      # 3 legs x (upper + ankle)
        assert len(torso) == 1
        assert len(arm) == 8       # link2/3/4/6 on both arms
        assert all("_equ" not in n for n in names)
        assert all(n.endswith("_offsettable_xyz") for n in names)
    finally:
        env.close()


def test_offset_bounds_are_bidirectional():
    # Links shorten as well as lengthen, with length_safe_lo keeping every link
    # safely above min_link_length + min_length_pad.
    env, base = _make_vec("BallCatcher-Pitch", 1)
    try:
        lo, hi = env.offset_bounds
        assert lo.shape == (N_OFFSETS,) and hi.shape == (N_OFFSETS,)
        assert (lo < 0.0).any()
        np.testing.assert_array_equal(hi, np.ones(N_OFFSETS, np.float32))
    finally:
        env.close()


def test_per_world_designs_differ():
    env, base = _make_vec("BallCatcher-Pitch", 2)
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


def test_cross_axis_offset_turns_a_link():
    # The 3D shape's whole point: an offset perpendicular to a link's built axis
    # moves its end sideways, turning the link. The single-axis deform could only
    # ever lengthen, leaving every geom_quat untouched; here the fully-extended
    # world's link frames must differ from the undeformed world's.
    env, base = _make_vec("BallCatcher-Pitch", 2)
    try:
        _, hi = env.offset_bounds
        emb = np.stack([np.zeros(env.n_offsets, np.float32), hi])
        env.reset(embodiment=emb)
        g = env._segs["geom"]
        quat = wp.to_torch(env.gpu_model.geom_quat)      # [world, geom, 4]
        turned = (quat[1, g] - quat[0, g]).abs().max(dim=-1).values
        assert float(turned.max()) > 1e-3, "no link frame turned under an offset"
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Codesign contract: freeze + structural validation
# --------------------------------------------------------------------------- #
def test_step_embodiment_is_frozen_with_warning():
    env, base = _make_vec("BallCatcher-Pitch", 2)
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
    env, base = _make_vec("BallCatcher-Pitch", 2)
    try:
        _, hi = env.offset_bounds
        # ball_catcher rejects only physically diverged resets, never a design,
        # so the structural hook passes any in-bounds embodiment.
        assert env.validate_embodiment(base).all()
        assert not env.validate_embodiment(hi + 1.0).any()           # out of bounds
        with pytest.raises(ValueError):
            env.reset(embodiment=hi + 1.0)
        assert not env.validate_embodiment(np.zeros(env.n_offsets + 1, np.float32)).any()   # wrong shape
        nan = np.zeros(env.n_offsets, np.float32)
        nan[0] = np.nan
        assert not env.validate_embodiment(nan).any()                # non-finite
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Robot swap: the dog base is still selectable via kwargs
# --------------------------------------------------------------------------- #
def test_dog_base_still_builds():
    from envs.ball_catcher.ball_catcher_env import _GROUND_GEOMS_DOG

    env, base = _make_vec(
        "BallCatcher-Pitch", 1,
        ant_xml="ant_reacher/dog.xml", arm_xml="ant_reacher/ant_arm.xml",
        ground_geoms=_GROUND_GEOMS_DOG)
    try:
        out = env.reset(embodiment=base)
        assert len(out) == 3
        env.step(torch.zeros((1, env.act_dim())))
    finally:
        env.close()


def test_rocky_basalt_variant_is_visual_only():
    """The basalt-textured Rocky variant keeps identical bodies, geoms, collisions,
    inertias, spaces, and physics trajectories to stock rocky.xml + rocky_arm.xml."""
    env_orig, base0 = _make_vec("BallCatcher-Pitch", 2)
    env_tex, base1 = _make_vec(
        "BallCatcher-Pitch", 2,
        ant_xml="ant_reacher/rocky_basalt.xml",
        arm_xml="ant_reacher/rocky_basalt_arm.xml",
    )
    try:
        assert env_tex.m.nbody == env_orig.m.nbody
        assert env_tex.m.ngeom == env_orig.m.ngeom
        assert env_tex.act_dim() == env_orig.act_dim() == ACT_DIM
        assert env_tex.obs_dim() == env_orig.obs_dim() == OBS_DIM
        assert env_tex.n_offsets == env_orig.n_offsets == N_OFFSETS
        assert base1.shape == base0.shape

        np.testing.assert_array_equal(env_tex.m.geom_contype, env_orig.m.geom_contype)
        np.testing.assert_array_equal(env_tex.m.geom_conaffinity, env_orig.m.geom_conaffinity)
        np.testing.assert_allclose(env_tex.m.body_mass, env_orig.m.body_mass, atol=1e-9)
        np.testing.assert_allclose(env_tex.m.body_inertia, env_orig.m.body_inertia, atol=1e-9)

        _, hi = env_orig.offset_bounds
        emb = np.stack([base0, 0.3 * hi])
        obs0, _, _ = env_orig.reset(seed=11, embodiment=emb)
        obs1, rej1, _ = env_tex.reset(seed=11, embodiment=emb)
        assert not np.asarray(rej1).any()
        np.testing.assert_allclose(obs1.cpu().numpy(), obs0.cpu().numpy(), atol=1e-6)

        rng = np.random.default_rng(11)
        for _ in range(5):
            act = torch.as_tensor(rng.uniform(-1, 1, (2, ACT_DIM)), dtype=torch.float32)
            o0, r0, t0, tr0, _ = env_orig.step(act)
            o1, r1, t1, tr1, _ = env_tex.step(act)
            np.testing.assert_allclose(o1.cpu().numpy(), o0.cpu().numpy(), atol=1e-5)
            np.testing.assert_allclose(r1.cpu().numpy(), r0.cpu().numpy(), atol=1e-5)
            assert torch.equal(t1, t0) and torch.equal(tr1, tr0)
    finally:
        env_orig.close()
        env_tex.close()


def test_tip_sphere_rides_and_is_massed_with_its_link():
    """link6r carries a sphere on its far end: it follows the tip and is massed with
    the link (compiled values at design 0), instead of being folded into its density."""
    env, base = _make_vec("BallCatcher-Pitch", 2)
    try:
        seg = env._segs
        k = next(i for i, g in enumerate(seg["tip_geom"]) if g >= 0)
        b, g, tg = int(seg["body"][k]), int(seg["geom"][k]), int(seg["tip_geom"][k])
        _, hi = env.offset_bounds
        env.reset(embodiment=np.stack([base, hi]))
        mass, ipos, gpos, size = (a.numpy() for a in (env.gpu_model.body_mass, env.gpu_model.body_ipos, env.gpu_model.geom_pos, env.gpu_model.geom_size))
        assert mass[0, b] == pytest.approx(env.m.body_mass[b], rel=1e-5)
        np.testing.assert_allclose(ipos[0, b], env.m.body_ipos[b], atol=1e-6)
        np.testing.assert_allclose(gpos[1, tg], 2.0 * gpos[1, g], atol=1e-6)        # sphere on the stretched tip
        r, rt, rho = float(seg["radius"][k]), float(seg["tip_r"][k]), float(seg["density"][k])
        length = 2.0 * float(size[1, g][1])
        assert mass[1, b] == pytest.approx(rho * (np.pi * r * r * length + 4.0 / 3.0 * np.pi * rt ** 3), rel=1e-5)
    finally:
        env.close()


def test_find_segments_refuses_an_unmodelled_extra_geom():
    """An extra geom the deform would neither move nor re-mass is refused."""
    import mujoco
    from envs.ball_catcher.offsettable_logic import find_segments
    xml = """<mujoco><worldbody><body name="arm_offsettable_z_pos"><joint type="hinge"/>
      <geom fromto="0 0 0 0 0 -0.4" size="0.05" type="capsule"/><geom pos="0 0 -0.2" size="0.08" type="sphere"/>
      </body></worldbody></mujoco>"""
    with pytest.raises(RuntimeError, match="extra geom"):
        find_segments(mujoco.MjModel.from_xml_string(xml))
    tip = xml.replace('pos="0 0 -0.2"', 'pos="0 0 -0.4"')          # centred on the far end: allowed
    assert find_segments(mujoco.MjModel.from_xml_string(tip))["tip_geom"][0] >= 0


def test_observations_stay_inside_the_declared_space():
    """The declared obs bounds are the true ones (finite only for quaternions and
    flags), so real observations -- including a stretched design's -- fit them."""
    env, base = _make_vec("BallCatcher-Pitch", 2)
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


def test_set_force_mag_reaches_the_kernels():
    """The live force setter changes the magnitude the device tunables carry."""
    env, _ = _make_vec("BallCatcher-Pitch", 1)
    try:
        env.set_force_mag(7.5)
        assert env.force_mag == 7.5
        assert (env._tunables == 7.5).sum() == 1
        env.set_force_mag(0.0)
        assert not (env._tunables == 7.5).any()
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
