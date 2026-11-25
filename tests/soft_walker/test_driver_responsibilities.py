"""Driver/task owners and lazy device resolution retain their public contracts."""

from __future__ import annotations

import importlib.util
import pickle
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def test_configuration_and_device_modules_import_without_jax() -> None:
    """Reading host configuration does not initialize an accelerator or concrete driver."""
    script = """
import sys
from envs.soft_walker.engine import device
from envs.soft_walker.engine.physics.spring_config import SpringConfig
assert SpringConfig().dt == 0.002
assert not {'jax', 'jaxlib', 'warp', 'diffrax'}.intersection(sys.modules)
assert 'envs.soft_walker.engine.drivers.spring_environment' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_device_resolution_preserves_explicit_selection_and_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU/CUDA selection, backend failures and invalid indices never silently fall back."""
    from envs.soft_walker.engine import device

    cpu = SimpleNamespace(platform="cpu")
    gpu = (SimpleNamespace(platform="gpu"), SimpleNamespace(platform="gpu"))
    calls: list[str | None] = []

    def devices(platform: str | None = None) -> list[object]:
        calls.append(platform)
        return [cpu] if platform in (None, "cpu") else list(gpu)

    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(devices=devices))
    assert device.resolve_device() is cpu
    assert device.resolve_device("cpu:0") is cpu
    assert device.resolve_device("cuda:1") is gpu[1]
    assert device.resolve_device("gpu") is gpu[0]
    assert device.resolve_device(cpu) is cpu
    assert calls == [None, "cpu", "gpu", "gpu"]
    with pytest.raises(ValueError, match="device must"):
        device.resolve_device("tpu")
    with pytest.raises(ValueError, match="Device index"):
        device.resolve_device("cuda:2")

    def unavailable(platform: str | None = None) -> list[object]:
        raise RuntimeError("backend unavailable")

    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(devices=unavailable))
    with pytest.raises(RuntimeError, match="backend unavailable"):
        device.resolve_device("cuda")


def test_precision_resolution_retains_x64_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The relocated precision check accepts only float32/64 and respects process policy."""
    from envs.soft_walker.engine import device

    policy = SimpleNamespace(x64_enabled=False)
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(config=policy))
    assert device.resolve_dtype("float32") == np.dtype("float32")
    with pytest.raises(ValueError, match="JAX_ENABLE_X64"):
        device.resolve_dtype("float64")
    with pytest.raises(ValueError, match="dtype must"):
        device.resolve_dtype("float16")
    policy.x64_enabled = True
    assert device.resolve_dtype(np.float64) == np.dtype("float64")


def test_driver_configuration_and_state_records_identify_their_actual_owners() -> None:
    """Canonical classes retain frozen config hashing and ordinary Python pickle identity."""
    from envs.soft_walker.engine.drivers.task import BodyConfig, TaskState
    from envs.soft_walker.engine.physics.spring_config import SpringConfig

    for cls, module in (
        (SpringConfig, "envs.soft_walker.engine.physics.spring_config"),
        (BodyConfig, "envs.soft_walker.engine.drivers.task"),
        (TaskState, "envs.soft_walker.engine.drivers.task"),
    ):
        assert cls.__module__ == module
        assert pickle.loads(pickle.dumps(cls)) is cls
    for config in (SpringConfig(), BodyConfig()):
        restored = pickle.loads(pickle.dumps(config))
        assert restored == config
        assert hash(restored) == hash(config)
        assert restored.dt == config.dt
    assert TaskState._fields == ("body", "displacement", "effort", "initial_com", "status")


def test_scalar_and_vector_drivers_share_the_canonical_lifecycle() -> None:
    """Body reset still inherits the shared scalar lifecycle; vector coordination stays distinct."""
    from envs.soft_walker.engine import device
    from envs.soft_walker.engine.drivers import environment, spring_environment, vector_lifecycle

    assert issubclass(environment.BodyEnv, spring_environment.Spring2DEnv)
    assert environment.BodyEnv.reset is spring_environment.Spring2DEnv.reset
    assert issubclass(spring_environment.Spring2DVectorEnv, vector_lifecycle.VectorLifecycle)
    assert spring_environment.resolve_device is device.resolve_device
    assert spring_environment.resolve_dtype is device.resolve_dtype
    assert environment.resolve_device is device.resolve_device
    assert environment.resolve_dtype is device.resolve_dtype
    for retired in ("env", "vector_env", "tasks", "config"):
        assert importlib.util.find_spec(f"envs.soft_walker.engine.{retired}") is None


def test_root_factories_look_up_the_moved_driver_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory dispatch remains live after relocation and forwards caller options unchanged."""
    from envs.soft_walker import engine
    from envs.soft_walker.engine.drivers import spring_environment

    scalar, vector = object(), object()
    calls: list[tuple[str, dict[str, object]]] = []

    def make(env_id: str, **kwargs: object) -> tuple[object, dict[str, object]]:
        calls.append((env_id, kwargs))
        return scalar, {}

    def make_vec(env_id: str, **kwargs: object) -> tuple[object, dict[str, object]]:
        calls.append((env_id, kwargs))
        return vector, {}

    monkeypatch.setattr(spring_environment, "make", make)
    monkeypatch.setattr(spring_environment, "make_vec", make_vec)
    assert engine.make("SoftSpring2D-v0", dtype="float32") == (scalar, {})
    assert engine.make_vec("SoftSpring2D-v0", num_envs=3, dtype="float32") == (vector, {})
    assert calls == [
        ("SoftSpring2D-v0", {"dtype": "float32"}),
        ("SoftSpring2D-v0", {"num_envs": 3, "dtype": "float32"}),
    ]
