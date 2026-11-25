"""Tests for the RotateInHand co-design environment (deformable humanoid hand).

Everything runs on the CPU warp device (``device="cpu", use_graph=False``): the
env is GPU-parallel in production, but the codesign contract -- per-world pads
written at reset without physics, masked partial resets, the frozen step-time
embodiment -- is device-independent.
"""

from __future__ import annotations

import warnings

import mujoco
import numpy as np
import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("warp")

import torch  # noqa: E402
import warp as wp  # noqa: E402

import codesign_gym as cg  # noqa: E402

SMALL = dict(device="cpu", use_graph=False)

# hand_no_xyz: 12 pads, 221 cells (a 10x8 palm, 3x3 to 3x6 finger pads), then 3
# link-offset slots (the links share them through _equ<n> groups).
N_CELLS = 221
N_SLOTS = 3
N_OFFSETS = N_CELLS + N_SLOTS
ACT_DIM = 19
# hand qpos (19) + qvel (19) + object/goal (18) + touching (1); the co-design
# env cuts the native can_deform flags and offsets in force (the design is not observed).
OBS_DIM = 19 + 19 + 18 + 1


def _make_vec(num_envs=2, **overrides):
    return cg.make_vec("RotateInHand-Cube", num_envs, **{**SMALL, **overrides})


def _random_designs(env, n, seed=0):
    lo, hi = env.embodiment_space.low, env.embodiment_space.high
    return lo + np.random.default_rng(seed).random((n, len(lo))).astype(np.float32) * (hi - lo)


_GOAL_FIELDS = ("quat", "t0", "dwell", "scored", "rebase", "dist", "reached", "paid", "elapsed", "gate", "pos_dist",
                "obj_pos", "obj_quat", "obj_vel", "obj_angvel", "obj_off_w")
_CONTACT_FIELDS = ("raw_hand", "raw_world", "hist_hand", "hist_world", "touch_hand", "touch_world", "mult")


def _task_state(env):
    state = {f: wp.to_torch(getattr(env._goal, f)).clone() for f in _GOAL_FIELDS}
    state.update({f: wp.to_torch(getattr(env._ct, f)).clone() for f in _CONTACT_FIELDS})
    return state


def test_registry_lists_rotate_in_hand():
    assert "RotateInHand-Cube" in cg.list_envs()


def test_preset_spaces_and_base():
    env, base = _make_vec()
    try:
        assert base.shape == (N_OFFSETS,) and not base.any()
        assert (env.n_cell, env.n_slots) == (N_CELLS, N_SLOTS)
        assert env.act_dim() == ACT_DIM and env.obs_dim() == OBS_DIM
        # True bounds: joints, velocities and the object position unbounded; quats in [-1, 1];
        # the goal angle and contact flags in [0, 1].
        box = env.single_observation_space
        free = np.r_[0:41, 45:51]                # hand qpos/qvel, object pos, object vel/angvel
        assert np.isinf(box.low[free]).all() and np.isinf(box.high[free]).all()
        assert (box.low[41:45] == -1).all() and (box.low[51:55] == -1).all() and (box.high[51:55] == 1).all()
        assert (box.low[55:] == 0).all() and (box.high[55:] == 1).all()
        lo, hi = env.embodiment_space.low, env.embodiment_space.high
        # A pad cell is one-sided: MuJoCo keeps elevation data in [0, 1].
        assert (lo[:N_CELLS] == 0).all() and (hi[:N_CELLS] == 1).all()
        assert env.embodiment_space.contains(base)
        # The agent config's task: a randomised goal stream, links deformable.
        assert not env.goal_fixed and env.goal_dist == 1.0
        assert env.max_link_offset == 0.015
    finally:
        env.close()


def test_rollout_deform_windows_are_refused():
    with pytest.raises(ValueError, match="must be 0"):
        _make_vec(deform_hfield_during_rollout_s=1.0)


def test_reset_writes_each_worlds_pads_without_physics():
    env, _ = _make_vec(3)
    try:
        designs = _random_designs(env, 3)
        obs, rejected, info = env.reset(embodiment=designs)
        assert obs.shape == (3, OBS_DIM)
        assert np.asarray(rejected).shape == (3,) and not np.asarray(rejected).any()
        np.testing.assert_allclose(info["embodiment"].cpu().numpy(), designs, atol=1e-6)
        rows = wp.to_torch(env._hfield_data).cpu().numpy()
        assert rows.shape[0] == 3
        assert len({row.tobytes() for row in rows}) == 3
        assert (env.world_steps == 0).all()
    finally:
        env.close()


def test_partial_reset_changes_only_masked_worlds():
    env, base = _make_vec(3)
    try:
        env.reset(embodiment=_random_designs(env, 3))
        for _ in range(3):
            env.step(torch.zeros((3, ACT_DIM)))
        before = wp.to_torch(env._hfield_data).clone()
        steps = env.world_steps.clone()
        mask = torch.tensor([False, True, False])
        env.reset(embodiment=np.stack([base] * 3), options={"reset_mask": mask})
        after = wp.to_torch(env._hfield_data)
        assert torch.equal(after[0], before[0]) and torch.equal(after[2], before[2])
        assert not after[1].any()                  # the base design is flat pads
        assert env.world_steps[1] == 0
        assert torch.equal(env.world_steps[[0, 2]], steps[[0, 2]])
        # The shaping record restarts from the respawn pose, not the old episode.
        assert torch.equal(env._prev_dist[1], env._dist[1])
        assert not env.reset_mask.any()            # no second respawn pending
    finally:
        env.close()


def test_partial_reset_leaves_other_worlds_task_state():
    # A partial reset must not tick the task of the worlds it does not reset: their dwell, goal and contact
    # history stay exactly as the last step left them, like every other env's side-effect-free reset.
    env, base = _make_vec(3)
    try:
        env.reset(embodiment=_random_designs(env, 3))
        for _ in range(3):
            obs, *_ = env.step(torch.zeros((3, ACT_DIM)))
        mask = torch.tensor([False, True, False])
        # A state a second task tick would change: one remembered contact shifts; then a goal at the pose dwells.
        for plant in ("hist_hand", "quat"):
            if plant == "hist_hand":
                wp.to_torch(env._ct.hist_hand)[:] = 1
            else:
                wp.to_torch(env._goal.quat)[:] = wp.to_torch(env._goal.obj_quat)
            before, diverged = _task_state(env), int(env._diverged_total)
            reset_obs, *_ = env.reset(embodiment=np.stack([base] * 3), options={"reset_mask": mask})
            after = _task_state(env)
            for name in before:
                assert torch.equal(after[name][[0, 2]], before[name][[0, 2]]), (plant, name)
            assert int(env._diverged_total) == diverged
            if plant == "hist_hand":
                assert torch.equal(reset_obs[[0, 2]], obs[[0, 2]])
    finally:
        env.close()


def test_step_needs_every_copy_reset_but_resets_may_be_partial():
    env, base = _make_vec(2)
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


def test_rollout_stays_finite():
    env, base = _make_vec(2)
    try:
        env.reset(embodiment=base)
        for t in range(20):
            action = 0.5 * torch.sin(0.1 * t + torch.arange(ACT_DIM, dtype=torch.float32))
            obs, reward, terminated, truncated, _ = env.step(action.expand(2, -1))
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
            assert all(env.single_observation_space.contains(row) for row in obs.numpy())
        assert reward.shape == terminated.shape == truncated.shape == (2,)
        assert env.last_success.dtype == torch.bool and env.last_success.shape == (2,)
        assert not env.last_diverged.any()
    finally:
        env.close()


def test_seeded_reset_draws_the_same_goal():
    env, base = _make_vec(2)
    try:
        env.reset(seed=5, embodiment=base)
        first = env._goal_quat_t.clone()
        env.reset(seed=5, embodiment=base)
        assert torch.equal(env._goal_quat_t, first)
        env.reset(seed=6, embodiment=base)
        assert not torch.equal(env._goal_quat_t, first)
    finally:
        env.close()


def test_step_embodiment_reshapes_the_hand_and_carries_into_autoreset():
    env, base = _make_vec(2, total_limit_s=0.1)
    try:
        assert env.metadata.get("supports_transition_embodiment") is True
        env.reset(embodiment=base)
        zeros = torch.zeros((2, ACT_DIM))
        designs = _random_designs(env, 2)
        obs, *_ = env.step(zeros, embodiment=designs)
        assert obs.shape == (2, OBS_DIM)
        np.testing.assert_allclose(env._offsets_t.cpu().numpy(), designs, atol=1e-6)
        rows = wp.to_torch(env._hfield_data)
        assert rows.any(dim=1).all() and not torch.equal(rows[0], rows[1])
        # One design is broadcast to every world.
        env.step(zeros, embodiment=designs[0])
        np.testing.assert_allclose(env._offsets_t.cpu().numpy(), designs[[0, 0]], atol=1e-6)
        # A world that autoresets respawns with its latest design, not the reset one.
        for _ in range(200):
            _, _, terminated, truncated, _ = env.step(zeros)
            done = terminated | truncated
            if bool(done.any()):
                break
        assert bool(done.any())
        _, reward, *_ = env.step(zeros)             # NEXT_STEP: the reset step, no physics for these worlds
        assert (env.world_steps[done] == 0).all() and (reward[done] == 0).all()
        np.testing.assert_allclose(env._offsets_t.cpu().numpy(), designs[[0, 0]], atol=1e-6)
    finally:
        env.close()


def test_eval_env_freezes_step_embodiment_with_warning():
    env, base = _make_vec(2, is_eval_env=True)
    try:
        env.reset(embodiment=base)
        pads = wp.to_torch(env._hfield_data).clone()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            env.step(torch.zeros((2, ACT_DIM)), embodiment=_random_designs(env, 2))
        assert any("Transition" in str(w.message) for w in caught)
        assert torch.equal(wp.to_torch(env._hfield_data), pads)
    finally:
        env.close()


def test_validate_embodiment_rejects_impossible():
    env, base = _make_vec(2)
    try:
        _, hi = env.offset_bounds
        assert env.validate_embodiment(base).all()
        assert not env.validate_embodiment(hi + 1.0).any()           # out of bounds
        with pytest.raises(ValueError):
            env.reset(embodiment=hi + 1.0)
        assert not env.validate_embodiment(np.zeros(N_OFFSETS + 1, np.float32)).any()   # wrong shape
        env.reset(embodiment=base)
        with pytest.raises(ValueError):          # a step-time design is checked too
            env.step(torch.zeros((2, ACT_DIM)), embodiment=hi + 1.0)
        nan = np.zeros(N_OFFSETS, np.float32)
        nan[0] = np.nan
        assert not env.validate_embodiment(nan).any()                # non-finite
    finally:
        env.close()


def _composite_truth(env, w):
    """Recompile the spec with world w's deformed geometry: MuJoCo's own masses."""
    import mujoco
    spec = env._build_world_spec()
    m0 = spec.compile()
    gs, gp, gq = (a.numpy()[w % a.shape[0]] for a in (env.gpu_model.geom_size, env.gpu_model.geom_pos, env.gpu_model.geom_quat))
    hs = env._hfield_size.numpy()[w]
    bp = env.gpu_model.body_pos.numpy()[w % env.gpu_model.body_pos.shape[0]]
    moving = (int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER))
    for g, sg in enumerate(spec.geoms):
        if int(m0.geom_type[g]) in moving:
            sg.fromto = [np.nan] * 6
            sg.size, sg.pos, sg.quat = gs[g].tolist(), gp[g].tolist(), gq[g].tolist()
    for h, sh in enumerate(spec.hfields):
        sh.size = hs[h].tolist()
    for b, sb in enumerate(spec.bodies):
        if b > 0:
            sb.pos = bp[b].tolist()
    return spec.compile()


def _tensor(inertia, iquat):
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(iquat, np.float64))
    R = R.reshape(3, 3)
    return R @ np.diag(inertia) @ R.T


def test_finger_links_are_massed_as_their_composite_geoms():
    """Each finger link's mass, centre of mass and inertia are its box, its (grown)
    heightfield pads and its fixed geoms together -- MuJoCo's values for that
    geometry -- at the base design and stretched."""
    env, _ = cg.make_vec("RotateInHand-Cube", 2)
    try:
        lo, hi = env.offset_bounds
        env.reset(embodiment=np.stack([np.zeros_like(hi), hi]))
        mass, ipos, inert, iquat = (a.numpy() for a in (env.gpu_model.body_mass, env.gpu_model.body_ipos, env.gpu_model.body_inertia, env.gpu_model.body_iquat))
        for w, truth in ((0, env.m), (1, _composite_truth(env, 1))):
            for b in (int(b) for b in env._segs["body"]):
                assert mass[w, b] == pytest.approx(truth.body_mass[b], rel=1e-4)
                np.testing.assert_allclose(ipos[w, b], truth.body_ipos[b], atol=1e-6)
                ref = _tensor(truth.body_inertia[b], truth.body_iquat[b])
                np.testing.assert_allclose(_tensor(inert[w, b], iquat[w, b]), ref, atol=1e-4 * np.abs(ref).max())
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Box links: quaternions in MuJoCo's w, x, y, z order
# --------------------------------------------------------------------------- #
from envs.rotate_in_hand import offsettable_logic as _ol  # noqa: E402


@wp.kernel
def _box_probe(q: wp.array(dtype=wp.quat), half: wp.array(dtype=wp.vec3),
               inertia: wp.array(dtype=wp.mat33), back: wp.array(dtype=wp.quat)):
    i = wp.tid()
    inertia[i] = _ol.box_inertia(2.0, half[i], q[i])
    back[i] = _ol.matrix_to_wxyz(_ol.wxyz_to_matrix(q[i]))


def test_box_inertia_reads_and_writes_wxyz_quaternions():
    """A box turned about any axis gets MuJoCo's inertia, and a principal frame
    converts back to the same w, x, y, z quaternion (up to sign)."""
    rng = np.random.default_rng(0)
    quats = rng.normal(size=(16, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    halves = rng.uniform(0.01, 0.05, (16, 3))
    inertia = wp.zeros(16, dtype=wp.mat33, device="cpu")
    back = wp.zeros(16, dtype=wp.quat, device="cpu")
    wp.launch(_box_probe, dim=16, device="cpu",
              inputs=[wp.array(quats, dtype=wp.quat, device="cpu"),
                      wp.array(halves, dtype=wp.vec3, device="cpu"), inertia, back])
    for q, h, got, q_back in zip(quats, halves, inertia.numpy(), back.numpy()):
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        R = R.reshape(3, 3)
        x, y, z = h ** 2
        diag = 2.0 / 3.0 * np.array([y + z, x + z, x + y])
        np.testing.assert_allclose(got, R @ np.diag(diag) @ R.T, atol=1e-6)
        assert min(np.abs(q_back - q).max(), np.abs(q_back + q).max()) < 1e-5


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


def test_relief_tapers_finger_link_pads_towards_the_joints():
    """With every cell raised, a finger-link pad's samples follow min(1, d / relief_length) along
    its columns, other pads stay flat at 1, and the reported design is not tapered."""
    env, base = _make_vec(1)
    try:
        design = np.ones(N_OFFSETS, dtype=np.float32)
        design[N_CELLS:] = 0.0
        _, _, info = env.reset(embodiment=design)
        assert torch.equal(info["embodiment"][0, :N_CELLS].cpu(), torch.ones(N_CELLS))
        data = wp.to_torch(env._hfield_data)[0].cpu().numpy()
        pads, k = env._pads, env.relief_length
        assert k == 2
        tapered_both, tapered_tip = 0, 0
        for i, h in enumerate(pads["hfield"]):
            nrow, ncol = int(pads["hrow"][i]), int(pads["hcol"][i])
            adr = int(env.m.hfield_adr[int(h)])
            samples = data[adr:adr + nrow * ncol].reshape(nrow, ncol)
            name = mujoco.mj_id2name(env.m, mujoco.mjtObj.mjOBJ_HFIELD, int(h))
            c = np.arange(ncol)
            if name.endswith(("proximal_x", "middle_x")):
                expected = np.minimum(1.0, np.minimum(c, ncol - 1 - c) / k)
                tapered_both += 1
            elif name.endswith("tip_x"):
                expected = np.minimum(1.0, (ncol - 1 - c) / k)
                tapered_tip += 1
            else:
                expected = np.ones(ncol)
            np.testing.assert_allclose(samples, np.broadcast_to(expected, samples.shape), atol=1e-6, err_msg=name)
        assert tapered_both == 7      # 3 fingers x proximal/middle + thumb proximal
        assert tapered_tip == 4       # 3 fingertips + thumb tip (untapered at free end)
    finally:
        env.close()


def test_relief_length_is_checked():
    with pytest.raises(ValueError, match="relief_length"):
        _make_vec(1, relief_length=-1)


@pytest.mark.parametrize("debug", [False, True])
def test_hfield_overflow_warning_only_in_debug(debug):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env, _ = _make_vec(1, debug=debug)
    try:
        overflow = [w for w in caught if "candidate contacts against a cap" in str(w.message)]
        assert bool(overflow) is debug
    finally:
        env.close()
