"""Device policy and real factory smoke coverage for MuJoCo Warp environments."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

import codesign_gym as cg
from core.device import resolve_mujoco_device


def mock_devices(monkeypatch, torch_count, warp_indices, current=0):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: torch_count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: torch_count)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current)

    def get_device(name):
        if int(name.split(":")[1]) not in warp_indices:
            raise ValueError("Invalid device identifier")
        return SimpleNamespace(is_cuda=True)
    monkeypatch.setattr(wp, "get_device", get_device)


@pytest.mark.parametrize("count,indices,current,expected", [
    (0, [], 0, "cpu"), (0, [0], 0, "cpu"), (2, [], 0, "cpu"),
    (2, [0, 1], 1, "cuda:1"), (2, [1], 0, "cuda:1"),
])
def test_auto_device_requires_both_runtimes(monkeypatch, count, indices, current, expected):
    mock_devices(monkeypatch, count, indices, current)
    assert resolve_mujoco_device() == expected


def test_explicit_device(monkeypatch):
    mock_devices(monkeypatch, 2, [0, 1], current=1)
    assert resolve_mujoco_device("cpu") == "cpu"
    assert resolve_mujoco_device("cuda") == "cuda:1"
    assert resolve_mujoco_device("cuda:0") == "cuda:0"
    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_mujoco_device("cuda:2")
    mock_devices(monkeypatch, 0, [])
    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_mujoco_device("cuda")


@pytest.mark.parametrize("device", ["mps", "gpu", "cuda:-1", "cuda:no", "cpu:1"])
def test_invalid_device(device):
    with pytest.raises(ValueError, match="Unsupported MuJoCo device"):
        resolve_mujoco_device(device)


ENV_IDS = ["BallCatcher-Pitch", "HalfCheetah-v4", "Swimmer-v4", "Hopper-v4",
           "Walker2d-v4", "Ant-v4", "Humanoid-v4", "HumanoidStandup-v4",
           "SolarCleaner-Reach", "TruckUnload-Single", "RotateInHand-Cube", "HandPickUp-Cube"]


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("single", [True, False])
@pytest.mark.parametrize("explicit", [True, False])
def test_factory_cpu_reset_and_step(monkeypatch, env_id, single, explicit):
    # Force the auto branch on GPU CI too; explicit CPU must also work there.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    kwargs = {"device": "cpu"} if explicit else {}
    env, base = cg.make(env_id, **kwargs) if single else cg.make_vec(env_id, 2, **kwargs)
    try:
        assert env.device == "cpu"
        assert not env._use_graph
        assert repr(env.embodiment_space).startswith(("Box", "EmbodimentSpace("))  # hand envs print a compact summary
        obs, rejected, _ = env.reset(seed=0, embodiment=base)
        assert not np.asarray(rejected).any()
        for _ in range(2):
            obs, reward, _, _, _ = env.step(env.action_space.sample())
            assert np.isfinite(np.asarray(obs)).all()
            assert np.isfinite(np.asarray(reward)).all()
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("single", [True, False])
def test_factory_cuda_reset_and_step(env_id, single):
    device = resolve_mujoco_device()
    if device == "cpu":
        pytest.skip("CUDA is unavailable to Torch and Warp")
    env, base = cg.make(env_id, device=device) if single else cg.make_vec(env_id, 2)
    try:
        assert env.device == device
        env.reset(seed=0, embodiment=base)
        for _ in range(2):
            _, reward, _, _, _ = env.step(env.action_space.sample())
            assert torch.isfinite(torch.as_tensor(reward)).all()
    finally:
        env.close()


@pytest.mark.parametrize("kwargs,expected", [({}, "cuda:1"), ({"device": "cpu"}, "cpu"),
    ({"device": "cuda:0"}, "cuda:0"), ({"use_cpu": True}, "cpu")])
def test_nerograsp_device_forwarding(monkeypatch, kwargs, expected):
    import sys
    from envs.nerograsp import registration
    mock_devices(monkeypatch, 2, [0, 1], current=1)
    # Observe what the builder hands to the simulator, without mesh preparation.
    monkeypatch.setitem(sys.modules, "envs.nerograsp.codesign_env",
                        SimpleNamespace(NeroGraspCoDesignEnv=lambda **kw: SimpleNamespace(**kw)))
    env = registration.build_vec(2, **kwargs)
    assert env.config.device == expected
    assert env.config.use_cpu == (expected == "cpu")


def test_nerograsp_conflicting_device():
    from envs.nerograsp.registration import build_vec
    with pytest.raises(ValueError, match="conflicts"):
        build_vec(1, use_cpu=True, device="cuda")


@pytest.mark.slow
@pytest.mark.parametrize("single", [True, False])
@pytest.mark.parametrize("explicit", [True, False])
def test_nerograsp_cpu_rollout(monkeypatch, tmp_path, single, explicit):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    kwargs = dict(runtime_dir=str(tmp_path), shape_parallel_workers=1)
    if explicit:
        kwargs["device"] = "cpu"
    env, base = (cg.make("NeroGrasp-SetScrew", **kwargs) if single else
                 cg.make_vec("NeroGrasp-SetScrew", 2, **kwargs))
    try:
        assert env.device == "cpu"
        assert "MeshSpace" in str(env.embodiment_space)
        _, rejected, _ = env.reset(seed=0, embodiment=base)
        assert not np.asarray(rejected).any()
        for _ in range(2):
            action = env.action_space.sample() if single else np.stack(
                [env.single_action_space.sample() for _ in range(env.num_envs)])
            _, reward, _, _, _ = env.step(action)
            assert np.isfinite(np.asarray(reward)).all()
    finally:
        env.close()
