"""Opt-in checks of real Newton graphics; a Pillow fallback cannot pass these.

Run with CODESIGN_TEST_NEWTON_RENDER=1 and an OpenGL/EGL context. The standalone
terrain and morphology tools additionally require CUDA, as their CLI does.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("CODESIGN_TEST_NEWTON_RENDER") != "1",
    reason="set CODESIGN_TEST_NEWTON_RENDER=1 to exercise real OpenGL rendering",
)


@pytest.mark.parametrize("driver", ["spring", "mpm2", "mpm3", "beam", "fem"])
def test_newton_frame_tracks_geometry(driver):
    from envs.soft_walker import config as C, embodiment as E
    from envs.soft_walker.render3d import NewtonRenderer

    props = C.base_props(driver)
    bundle = C.build_bundle(driver, props, "float64")
    design = E.from_cells(props, C.dim_of(driver), "float64")
    renderer = NewtonRenderer(driver, bundle.fields, design, None, width=400, height=400)
    try:
        positions = bundle.fields["reference"].copy()
        first = renderer.draw(positions)
        positions[:, 1] += 0.03
        second = renderer.draw(positions)
        assert first.shape == second.shape == (400, 400, 3)
        assert first.dtype == second.dtype == np.uint8
        assert first.std() > 1
        assert not np.array_equal(first, second)
    finally:
        renderer.close()


def test_newton_terrain_editor():
    import warp as wp

    from envs.soft_walker.viewer.terrain_editor import TerrainEditor

    if not wp.is_cuda_available():
        pytest.skip("standalone terrain editor requires CUDA")
    args = SimpleNamespace(preset="stairs", dim=2, seed=7, width=400, height=400,
                           gallery=None, headless=True)
    editor = TerrainEditor(args)
    try:
        editor.render()
        frame = editor.viewer.get_frame().numpy()
        assert frame.shape == (400, 400, 3)
        assert frame.std() > 1
    finally:
        editor.viewer.close()


def test_newton_morphology_material_and_actuator_frames():
    import warp as wp

    from envs.soft_walker.viewer.render_morphologies import MorphologyStudio

    if not wp.is_cuda_available():
        pytest.skip("standalone morphology renderer requires CUDA")
    candidate = dict(
        id="newton-smoke", dim=2, shape=(2, 2), size=np.array([0.08, 0.08]),
        coordinates=np.array(list(np.ndindex(2, 2))), active=np.ones(4, dtype=bool),
        fields=dict(young=np.full(4, 20000.0), actuator_type=np.ones(4, dtype=int),
                    strength=np.ones(4), channel=np.zeros(4, dtype=int)),
    )
    studio = MorphologyStudio(SimpleNamespace(width=600, height=600, extrusion=0.006),
                             [candidate])
    try:
        material, count = studio.render(candidate, "material", (10000.0, 30000.0), 0)
        actuator, other_count = studio.render(candidate, "actuator", (10000.0, 30000.0), 1)
        assert count == other_count == 16
        assert material.size == actuator.size == (600, 600)
        assert np.asarray(material).std() > 1
        assert not np.array_equal(material, actuator)
    finally:
        studio.viewer.close()
