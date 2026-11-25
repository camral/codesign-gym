"""Explicit driver modules retain lightweight namespaces and factory/runtime routing."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest

from envs.soft_walker import engine
from envs.soft_walker.engine import drivers
from envs.soft_walker.engine.drivers import environment, model_access, reporting
from envs.soft_walker.engine.drivers.task import BodyConfig
from envs.soft_walker.engine.physics import beam, fem, mpm

if TYPE_CHECKING:
    from collections.abc import Callable


def test_driver_namespace_does_not_initialize_backends() -> None:
    """Importing the namespace alone does not select physics or load an accelerator."""
    script = """
import json, sys
from envs.soft_walker.engine import drivers
assert not hasattr(drivers, 'BodyEnv')
print(json.dumps({name: name in sys.modules for name in ('jax', 'warp', 'newton')}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    assert json.loads(result.stdout) == {"jax": False, "warp": False, "newton": False}


def test_driver_helpers_and_physics_dispatch_have_canonical_owners() -> None:
    """Environment adapters share accessor/reporting functions and concrete physics modules."""
    assert environment.BodyEnv.__module__ == "envs.soft_walker.engine.drivers.environment"
    assert environment.reference is model_access.reference
    assert environment.positions is model_access.positions
    assert environment.channel_mask is model_access.channel_mask
    assert reporting.positions is model_access.positions
    assert environment.observe is reporting.observe
    assert environment.mechanics_diagnostics is reporting.mechanics_diagnostics
    for kind, expected in (("beam", beam), ("fem", fem), ("mpm2", mpm), ("mpm3", mpm)):
        assert model_access.module(kind) is expected
    assert not hasattr(drivers, "BodyEnv")
    for old_name in ("driver_models", "driver_reporting"):
        assert importlib.util.find_spec(f"envs.soft_walker.engine.{old_name}") is None


@pytest.mark.parametrize("env_id", tuple(environment.IDS))
def test_factory_constructs_canonical_body_environment_without_an_episode(env_id: str) -> None:
    """Factory imports reach all four body drivers without creating or settling live state."""
    planar = env_id == "SoftMPM2D-v0"
    config = BodyConfig(
        grid_shape=(2, 1) if planar else (2, 1, 1),
        size=(0.1, 0.05) if planar else (0.1, 0.05, 0.05),
        settle_steps=0,
    )
    instance, baseline = engine.make(env_id, dtype="float32", device="cpu", config=config)
    try:
        assert type(instance) is environment.BodyEnv
        assert instance.kind == environment.IDS[env_id]
        assert instance.embodiment_space.contains(baseline)
        assert instance._model is None and instance._state is None
    finally:
        instance.close()


def test_runtime_renderer_and_rollout_imports_reach_engine_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lazy rendering and rollout binding resolve the parent engine after the module move."""
    from envs.soft_walker.engine import body_rollout, render

    instance, _ = engine.make("SoftHexFEM3D-v0", dtype="float32", device="cpu")
    frame = np.zeros((1, 1, 3), dtype=np.uint8)
    calls: list[object] = []

    def renderer(owner: object) -> np.ndarray:
        calls.append(owner)
        return frame

    def binder(owner: object, **options: object) -> Callable[..., object]:
        calls.append((owner, options))
        return renderer

    monkeypatch.setattr(render, "render_body_rgb", renderer)
    monkeypatch.setattr(body_rollout, "bind", binder)
    try:
        # Rendering delegates after checking only that an episode state exists.
        instance._state = object()
        instance.render_mode = "rgb_array"
        assert instance.render() is frame
        assert instance.bind_rollout(record_trajectory=True) is renderer
        assert calls == [instance, (instance, {"record_trajectory": True})]
    finally:
        instance.close()
