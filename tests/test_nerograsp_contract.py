"""NeroGraspAlt (src/envs/nerograsp) under the shared contracts NeroGrasp is tested against:
per-env validation, rendering, device forwarding, a CPU rollout, and the mesh space."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco_warp")

import codesign_gym as cg  # noqa: E402
from codesign_gym import SingleEnv  # noqa: E402

DEVICE = os.environ.get("CODESIGN_RENDER_TEST_DEVICE", "cpu")
needs_viewer = pytest.mark.skipif(
    sys.platform == "darwin" or not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
    reason="interactive viewer needs a display")


def test_validates_per_env_and_flags_per_env():
    env, base = cg.make_vec("NeroGrasp-Thin", 2, use_cpu=True)
    try:
        valid = np.asarray(env.validate_embodiment(base))
        assert valid.shape == (2,) and valid.all()
        assert np.asarray(env.validate_embodiment([None] * len(base))).all()   # None entries are the base design
        _, rejected, _ = env.reset(embodiment=[None] * len(base))
        assert tuple(rejected.shape) == (2,) and not bool(rejected.any())
    finally:
        env.close()


def test_rejected_design_is_reported_per_env_and_blocks_step():
    env, base = cg.make_vec("NeroGrasp-Tiny", 8, use_cpu=True)
    try:
        bad = [base[0], {k: m.copy() for k, m in base[1].items()}]
        bad[1]["index"].vertices[0, 0] += 1e-3                      # moves a mounting vertex
        _, rejected, info = env.reset(embodiment=bad)
        assert rejected.tolist() == [False] * 4 + [True] * 4
        assert all("Mounting vertices" in r for r in info["embodiment/rejection_reasons"][4:])
        with pytest.raises(RuntimeError, match="reset"):
            env.step(torch.zeros(env.action_space.shape))
    finally:
        env.close()


def test_render_mode_reaches_its_native_renderer():
    env, _ = cg.make_vec("NeroGrasp-Thin", num_envs=2, device=DEVICE, render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frames = env.render()
        assert isinstance(frames, tuple) and len(frames) == 2
        assert all(frame.shape[-1] == 3 and frame.dtype == np.uint8 for frame in frames)
    finally:
        env.close()


@needs_viewer
def test_human_viewer_uses_retained_model():
    env, _ = cg.make("NeroGrasp-Thin", device=DEVICE, render_mode="human")
    try:
        env.reset(seed=0)
        viewer = env.vector_env._human_viewer
        assert viewer is not None and viewer.is_running()
        assert env.render() is None
    finally:
        env.close()


def _mock_devices(monkeypatch, torch_count, warp_indices, current=0):
    import warp as wp
    monkeypatch.setattr(torch.cuda, "is_available", lambda: torch_count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: torch_count)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)

    def get_device(name):
        if int(name.split(":")[1]) not in warp_indices:
            raise ValueError("Invalid device identifier")
        return SimpleNamespace(is_cuda=True)
    monkeypatch.setattr(wp, "get_device", get_device)


@pytest.mark.parametrize("kwargs,expected", [({}, "cuda:1"), ({"device": "cpu"}, "cpu"),
                                             ({"device": "cuda:0"}, "cuda:0"), ({"use_cpu": True}, "cpu")])
def test_device_forwarding(monkeypatch, kwargs, expected):
    from envs.nerograsp import registration
    _mock_devices(monkeypatch, 2, [0, 1], current=1)
    # Observe what the builder hands to the simulator, without mesh preparation.
    monkeypatch.setitem(sys.modules, "envs.nerograsp.codesign_env",
                        SimpleNamespace(NeroGraspCoDesignEnv=lambda **kw: SimpleNamespace(**kw)))
    env = registration.build_vec(2, **kwargs)
    assert env.config.device == expected
    assert env.config.use_cpu == (expected == "cpu")


def test_conflicting_device():
    from envs.nerograsp.registration import build_vec
    with pytest.raises(ValueError, match="conflicts"):
        build_vec(1, use_cpu=True, device="cuda")


@pytest.mark.slow
@pytest.mark.parametrize("single", [True, False])
def test_cpu_rollout(monkeypatch, tmp_path, single):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    kwargs = dict(runtime_dir=str(tmp_path), shape_parallel_workers=1)
    env, base = cg.make("NeroGrasp-SetScrew", **kwargs) if single else cg.make_vec("NeroGrasp-SetScrew", 2, **kwargs)
    try:
        assert env.device == "cpu"
        assert "MeshSpace" in str(env.embodiment_space)
        _, rejected, _ = env.reset(seed=0, embodiment=base)
        assert not np.asarray(rejected).any()
        for _ in range(2):
            action = env.action_space.sample() if single else np.stack([env.single_action_space.sample() for _ in range(env.num_envs)])
            _, reward, _, _, _ = env.step(action)
            assert np.isfinite(np.asarray(reward)).all()
    finally:
        env.close()


def test_mesh_space_matches_the_original_contract():
    from envs.nerograsp import Mesh, MeshSpace
    mesh = Mesh(np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
                np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]))
    space = MeshSpace({"tip": mesh}, {"tip": 3})
    assert space.contains({"tip": mesh})
    assert str(SingleEnv(SimpleNamespace(embodiment_space=space)).embodiment_space) == repr(space)


def test_every_used_convex_part_collides_in_every_world():
    """Collision filtering is shared by all worlds: a world whose design has more convex parts than
    world 0's must still collide with all of them, and its unused slots hold the placeholder."""
    env, base = cg.make_vec("NeroGrasp-Thin", 2, use_cpu=True)
    try:
        rng = np.random.default_rng(1234)
        designs = [base[0], [env.embodiment_space.sample(rng) for _ in range(2)][1]]   # passes the rejectors
        _, rejected, _ = env.reset(embodiment=designs)
        assert not bool(rejected.any())
        m = env.m
        names = [m.mesh(i).name for i in range(m.nmesh)]
        dataid = env.cp_model.geom_dataid.numpy()
        mass = env.cp_model.body_mass.numpy()
        slots = [g for g in range(m.ngeom) if "_part_" in m.geom(g).name]
        for world in range(env.num_envs):
            for g in slots:
                mesh = names[dataid[world, g]]
                assert m.geom_contype[g] and m.geom_conaffinity[g], f"slot {m.geom(g).name} cannot collide"
                assert mesh.startswith("convex_part_") or mesh.endswith("_unused_slot")
        tip_mass = mass[:, m.body("leap_if_tip").id]
        parts = [sum(names[dataid[w, g]].startswith("convex_part_") for g in slots if m.geom(g).name.startswith("leap_if_tip")) for w in range(2)]
        assert parts[0] != parts[1] and tip_mass[0] != tip_mass[1]     # the two designs really differ
    finally:
        env.close()
