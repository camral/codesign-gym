"""Host rendering scenes preserve ownership and do not require accelerator imports."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest

from envs.soft_walker.engine.render import (
    BeamPreview,
    BodyPreviewScene,
    ParticlePreview,
    SurfacePreview,
    TerrainPreview,
    body_preview_scene,
    render_body_scene_rgb,
)

if TYPE_CHECKING:
    from envs.soft_walker.engine.drivers.environment import BodyEnv


def _scene(geometry: BeamPreview | ParticlePreview | SurfacePreview) -> BodyPreviewScene:
    """Construct one small host-only scene with active and absent geometry."""
    return BodyPreviewScene(
        positions=np.array(
            [[0.0, 0.04, 0.0], [0.08, 0.04, 0.0], [0.0, 0.12, 0.0], [0.08, 0.12, 0.0]]
        ),
        active=np.array([True, True, True, False]),
        terrain=TerrainPreview(
            normal=np.array([0.0, 1.0, 0.0]),
            offset=0.0,
            centers=np.array([[0.25, 0.0, 0.0]]),
            half_sizes=np.array([[0.02, 0.01, 0.02]]),
            active=np.array([True]),
        ),
        geometry=geometry,
        length_ref=0.2,
        label="Host geometry fixture",
    )


@pytest.mark.parametrize(
    "geometry",
    [
        BeamPreview(np.array([[0, 1], [0, 2], [1, 3]]), 0.08),
        ParticlePreview(0.04, 2),
        SurfacePreview(
            faces=np.array([[[0, 1, 2, 3]]]),
            neighbors=np.zeros((1, 1), dtype=int),
            has_neighbor=np.zeros((1, 1), dtype=bool),
            alpha=np.array([1.0]),
            young=np.array([20000.0]),
        ),
    ],
)
def test_host_scene_raster_is_deterministic_and_does_not_mutate_inputs(
    geometry: BeamPreview | ParticlePreview | SurfacePreview,
) -> None:
    """All families render read-only NumPy inputs without a live simulator."""
    scene = _scene(geometry)
    for owner in (scene, scene.terrain, scene.geometry):
        for value in vars(owner).values():
            if isinstance(value, np.ndarray):
                value.flags.writeable = False
    first = render_body_scene_rgb(scene, 320, 240)
    second = render_body_scene_rgb(scene, 320, 240)
    assert first.shape == (240, 320, 3)
    assert first.dtype == np.uint8
    np.testing.assert_array_equal(first, second)
    # A frame responds to actual host geometry, not only its label or background.
    moved = replace(scene, positions=scene.positions + np.array([0.0, 0.1, 0.0]))
    assert not np.array_equal(first, render_body_scene_rgb(moved, 320, 240))


def test_body_adapter_copies_state_topology_and_terrain(monkeypatch: pytest.MonkeyPatch) -> None:
    """A captured frame remains usable after the environment's arrays change."""
    from envs.soft_walker.engine.drivers import environment

    geometry = BeamPreview(np.array([[0, 1], [0, 2]]), 0.08)
    scene = _scene(geometry)
    model = SimpleNamespace(
        active=scene.active.copy(),
        topology=SimpleNamespace(edges=geometry.edges.copy(), h=0.08),
    )
    state = SimpleNamespace(body=SimpleNamespace(x=scene.positions.copy()), time=0.125)
    ground = SimpleNamespace(
        **{
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in vars(scene.terrain).items()
        }
    )
    env = SimpleNamespace(
        _model=model,
        _state=state,
        _terrain=ground,
        kind="beam",
        env_id="SoftBeam3D-v0",
        dtype="float64",
        config=SimpleNamespace(length_ref=0.2),
    )
    monkeypatch.setattr(environment, "positions", lambda model, state, kind: state.x)
    captured = body_preview_scene(cast("BodyEnv", env))
    expected = render_body_scene_rgb(captured, 320, 240)
    state.body.x[:] = 100.0
    model.active[:] = False
    model.topology.edges[:] = 0
    ground.normal[:] = 0.0
    ground.centers[:] = 100.0
    np.testing.assert_array_equal(expected, render_body_scene_rgb(captured, 320, 240))
    assert captured.label == "SoftBeam3D-v0 | t=0.125s | float64"


def test_host_rendering_does_not_import_accelerator_or_driver_modules() -> None:
    """Importing and rasterizing explicit host data stays behind the lazy boundary."""
    script = """
import importlib.abc
import sys

class RejectBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'jaxlib', 'warp', 'newton'}:
            raise AssertionError(f'Unexpected backend import: {fullname}')

sys.meta_path.insert(0, RejectBackends())
import numpy as np
from envs.soft_walker.engine.render import (
    BodyPreviewScene, ParticlePreview, TerrainPreview, render_body_scene_rgb,
)
scene = BodyPreviewScene(
    positions=np.array([[0., .04], [.04, .08]]), active=np.ones(2, dtype=bool),
    terrain=TerrainPreview(np.array([0., 1.]), 0., np.empty((0, 2)), np.empty((0, 2)), np.empty(0, dtype=bool)),
    geometry=ParticlePreview(.04, 2), length_ref=.2, label='Host frame',
)
assert render_body_scene_rgb(scene, 320, 240).shape == (240, 320, 3)
assert 'envs.soft_walker.engine.drivers.environment' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
