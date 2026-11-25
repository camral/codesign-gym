"""Tests for NeroGrasp-Tools-Carabiner and NeroGrasp-Tools-WireCutters
(parallel-gripper Nero arm with deformable fingertip pads and articulated tools)."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco_warp")

import codesign_gym as cg  # noqa: E402
from gymnasium.vector import AutoresetMode  # noqa: E402


def _make_large_gripper_deformation(base_design):
    from envs.nerograsp.config import TOOLS_MOUNTING_VERTEX_COUNT as mc
    out = {k: m.copy() for k, m in base_design.items()}
    for key, sign in (("left", 1.0), ("right", -1.0)):
        v = out[key].vertices.astype(np.float64)
        y_mount_max = v[:mc, 1].max()
        y_tip_max = v[mc:, 1].max()
        t = np.clip((v[mc:, 1] - y_mount_max) / max(y_tip_max - y_mount_max, 1e-6), 0.0, 1.0)
        w = t * t * (3.0 - 2.0 * t)
        v[mc:, 1] += w * 0.005
        v[mc:, 0] += w * 0.004 * sign
        out[key].vertices = v
    return out


@pytest.fixture(scope="module")
def carabiner():
    env, base = cg.make_vec("NeroGrasp-Tools-Carabiner", 2, use_cpu=True, shape_parallel_workers=2)
    yield env, base
    env.close()


@pytest.fixture(scope="module")
def wire_cutters():
    env, base = cg.make_vec("NeroGrasp-Tools-WireCutters", 2, use_cpu=True, shape_parallel_workers=2)
    yield env, base
    env.close()


@pytest.mark.parametrize("fixture_name,expected_obj", [
    ("carabiner", "carabiner"),
    ("wire_cutters", "wire_cutters"),
])
def test_spaces_contract_and_8d_rollout(request, fixture_name, expected_obj):
    env, base = request.getfixturevalue(fixture_name)
    assert env.metadata["autoreset_mode"] == AutoresetMode.DISABLED
    assert env.single_action_space.shape == (8,)
    assert env.action_space.shape == (2, 8)
    assert env.single_observation_space["actor"].shape == (52,)
    assert env.single_observation_space["critic"].shape == (98,)
    assert env.embodiment_space.mounting_vertex_counts == {"left": 12, "right": 12}
    assert set(base[0].keys()) == {"left", "right"}

    obs, rejected, info = env.reset(seed=0, embodiment=base)
    assert tuple(rejected.shape) == (2,) and not bool(rejected.any())
    assert info["task/object"] == (expected_obj, expected_obj)
    assert obs["actor"].shape == (2, 52) and obs["critic"].shape == (2, 98)
    assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()

    # Verify radian joint limits and realistic fingertip mass (~48 g, not multi-kg shell inertia).
    import mujoco
    tip_mass = float(env.m.body_mass[mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_BODY, "nero_gripper_link1_tip")])
    assert 0.02 < tip_mass < 0.10
    if expected_obj == "carabiner":
        assert np.isclose(env.task_joint_limit, 0.36651914291881926 - (-0.51), atol=1e-3)
    else:
        assert np.isclose(env.task_joint_limit, 0.7853981633974483, atol=1e-3)

    with pytest.raises(NotImplementedError):
        env.reset(options={"reset_mask": np.array([True, False])})

    rng = np.random.default_rng(12)
    for step in range(env.max_episode_steps):
        action = rng.uniform(-0.2, 0.2, size=(2, 8)).astype(np.float32) if step < 5 else np.zeros((2, 8), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert torch.isfinite(obs["actor"]).all() and torch.isfinite(obs["critic"]).all()
        assert torch.isfinite(reward).all()
        assert info["contacts/fingertip_object"].shape == (2, 2)
        assert info["task/articulation_ratio"].shape == (2,)
        if info["finished"].all():
            assert "episode/final_return" in info
            assert f"object_return/{expected_obj}" in info
            assert "all_reward/articulation_reward" in info
            break
    assert info["finished"].all()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros((2, 8), dtype=np.float32))
    obs_rec, rej_rec, _ = env.reset(seed=1)
    assert not bool(rej_rec.any()) and torch.isfinite(obs_rec["critic"]).all()


def test_seeding_isolation_and_noise_partition():
    env, base = cg.make_vec("NeroGrasp-Tools-Carabiner", 4, use_cpu=True, shape_parallel_workers=2)
    try:
        random.seed(123); np.random.seed(123); torch.manual_seed(123)
        expected = (random.random(), np.random.rand(), float(torch.rand(1)))
        random.seed(123); np.random.seed(123); torch.manual_seed(123)
        obs_a, _, _ = env.reset(seed=7)
        assert (random.random(), np.random.rand(), float(torch.rand(1))) == expected

        # Same seed -> bit-for-bit identical reset and noisy rollout
        act = np.zeros((4, 8), dtype=np.float32)
        obs_a1, rew_a1, _, _, _ = env.step(act)
        obs_b, _, _ = env.reset(seed=7)
        obs_b1, rew_b1, _, _, _ = env.step(act)
        np.testing.assert_array_equal(obs_a["critic"].cpu().numpy(), obs_b["critic"].cpu().numpy())
        np.testing.assert_array_equal(obs_a1["critic"].cpu().numpy(), obs_b1["critic"].cpu().numpy())
        np.testing.assert_array_equal(rew_a1.cpu().numpy(), rew_b1.cpu().numpy())

        # Mixed-noise layout for num_envs=4: env 0 clean (0.0), env 1 low (0.25), envs 2-3 full (1.0)
        obs_c, _, _ = env.reset(seed=99)
        pos_b = obs_b["critic"][:, 52:55].cpu().numpy()
        pos_c = obs_c["critic"][:, 52:55].cpu().numpy()
        np.testing.assert_array_equal(pos_b[0], pos_c[0])
        assert not np.allclose(pos_b[2], pos_c[2])
    finally:
        env.close()

    # With all noise disabled, different seeds produce identical reset poses and step outcomes.
    clean_env, _ = cg.make_vec(
        "NeroGrasp-Tools-WireCutters",
        2,
        use_cpu=True,
        shape_parallel_workers=2,
        object_position_randomization=0.0,
        object_rotation_randomization_deg=0.0,
        gripper_action_noise_std=0.0,
    )
    try:
        obs_1, _, _ = clean_env.reset(seed=1)
        obs_1s, rew_1s, _, _, _ = clean_env.step(np.zeros((2, 8), dtype=np.float32))
        obs_2, _, _ = clean_env.reset(seed=999)
        obs_2s, rew_2s, _, _, _ = clean_env.step(np.zeros((2, 8), dtype=np.float32))
        np.testing.assert_array_equal(obs_1["critic"].cpu().numpy(), obs_2["critic"].cpu().numpy())
        np.testing.assert_array_equal(obs_1s["critic"].cpu().numpy(), obs_2s["critic"].cpu().numpy())
        np.testing.assert_array_equal(rew_1s.cpu().numpy(), rew_2s.cpu().numpy())
    finally:
        clean_env.close()


def test_snap_plane_clamp_accepts_deformation_and_rejects_topology_change():
    env, base = cg.make_vec("NeroGrasp-Tools-Carabiner", 2, use_cpu=True, shape_parallel_workers=2, snap_plane_clamp=True)
    try:
        deformed = _make_large_gripper_deformation(base[1])
        _, rej, _ = env.reset(seed=0, embodiment=[base[0], deformed])
        assert not bool(rej.any())
        # Dropping a non-mounting face alters topology and must be rejected when snap_plane_clamp=True.
        topo_bad = {k: m.copy() for k, m in base[1].items()}
        topo_bad["left"].faces = topo_bad["left"].faces[:-1]
        _, rej_topo, info_topo = env.reset(seed=0, embodiment=[base[0], topo_bad])
        assert rej_topo.tolist() == [False, True]
        assert info_topo["embodiment/rejection_reasons"][1]
    finally:
        env.close()


def test_deformed_tips_mixed_slots_and_rejection(carabiner):
    env, base = carabiner
    deformed = _make_large_gripper_deformation(base[1])
    valid = np.asarray(env.validate_embodiment([base[0], deformed]))
    assert valid.tolist() == [True, True]

    _, rejected, info = env.reset(seed=0, embodiment=[base[0], deformed])
    assert not bool(rejected.any())
    assert "mesh_quality/mesh_triangle_shape" in info
    assert info["mesh_quality/mesh_triangle_shape"].shape == (2, 2)

    m = env.m
    names = [m.mesh(i).name for i in range(m.nmesh)]
    dataid = env.cp_model.geom_dataid.numpy()
    slots = [g for g in range(m.ngeom) if "_part_" in m.geom(g).name]
    assert len(slots) == 2 * env.max_parts
    for w in range(2):
        for g in slots:
            mesh = names[dataid[w, g]]
            assert m.geom_contype[g] and m.geom_conaffinity[g]
            assert mesh.startswith("convex_part_") or mesh.endswith("_unused_slot")
    obs, rew, _, _, _ = env.step(np.zeros((2, 8), dtype=np.float32))
    assert torch.isfinite(obs["critic"]).all() and torch.isfinite(rew).all()

    # Moving a mounting vertex (index 0) must reject only that env and block step()
    bad = {k: m.copy() for k, m in base[1].items()}
    bad["left"].vertices[0, 0] += 1e-3
    _, rej_bad, info_bad = env.reset(seed=0, embodiment=[base[0], bad])
    assert rej_bad.tolist() == [False, True]
    assert "Mounting vertices" in info_bad["embodiment/rejection_reasons"][1]
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros((2, 8), dtype=np.float32))

    # Recover cleanly using None -> base design
    _, rej_rec, _ = env.reset(seed=0, embodiment=[None, None])
    assert not bool(rej_rec.any())


@pytest.mark.parametrize("env_id", ["NeroGrasp-Tools-Carabiner", "NeroGrasp-Tools-WireCutters"])
def test_single_env_demo_and_render(env_id):
    env, base = cg.make(env_id, use_cpu=True, shape_parallel_workers=2, trajectory="demo", render_mode="rgb_array")
    try:
        obs, rejected, _ = env.reset(seed=0, embodiment=base)
        assert not bool(rejected)
        assert obs["actor"].shape == (52,) and obs["critic"].shape == (98,)
        frame = env.render()
        assert frame.ndim == 3 and frame.shape[-1] == 3 and frame.dtype == np.uint8
        _, rew, _, _, _ = env.step(np.zeros(8, dtype=np.float32))
        assert np.isfinite(float(rew))
    finally:
        env.close()
