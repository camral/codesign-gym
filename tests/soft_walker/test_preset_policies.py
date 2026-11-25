"""Frozen chamber seeds and independent terrain generators survive ownership changes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from soft_paths import SOFT_ROOT

from envs.soft_walker.presets.embodiments.chambers import ChamberDecoder
from envs.soft_walker.presets.terrain.editor_courses import PRESETS, make_recipe
from envs.soft_walker.presets.terrain.scenario_courses import terrain_config
from envs.soft_walker.terrain.recipes import recipe_terrain, validate_recipe


@pytest.fixture(scope="module")
def reference() -> dict[str, Any]:
    """Read baseline data captured before the policy ownership refactor."""
    path = Path(__file__).parent / "fixtures" / "preset_policy_v1.json"
    return json.loads(path.read_text())


@pytest.mark.parametrize("preset", PRESETS)
def test_editor_and_scenario_courses_keep_distinct_frozen_samples(
    preset: str, reference: dict[str, Any]
) -> None:
    """Both RNG families retain their sample ordering and original dimensions."""
    editor = make_recipe(preset, dim=3, seed=7)
    scenario = json.loads(json.dumps(asdict(terrain_config(preset, dim=3, seed=7))))
    assert editor == reference["editor_courses"][preset]
    assert scenario == reference["scenario_courses"][preset]
    if preset in ("low_blocks", "rough_blocks"):
        assert editor["config"]["fixed_boxes"] != scenario["fixed_boxes"]


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("representation", ["primitives", "cppn"])
def test_chamber_seed_arrays_preserve_saved_parameter_layout(
    dim: int, representation: str, dtype: np.dtype[np.floating], reference: dict[str, Any]
) -> None:
    """Named graph roles preserve seed coefficients, array order and material priors."""
    decoder = ChamberDecoder(dim, representation, dtype)
    for key, actual in decoder.base().items():
        expected = np.asarray(reference["chambers"][str(dim)][key], dtype=actual.dtype)
        np.testing.assert_array_equal(actual, expected)
    assert decoder.descriptor()["name"] == "scenario_chamber_shape_v1"


def test_chamber_routes_keep_side_and_torso_boundary_conventions(
    dtype: np.dtype[np.floating],
) -> None:
    """The centre plane belongs to the low-side leg; chamber midlines route high."""
    decoder = ChamberDecoder(2, dtype=dtype)
    points = jnp.asarray([[-0.75, 0.0], [0.75, 0.0], [0.0, 0.0], [0.0, 0.4]], dtype)
    torso, leg, channel = decoder.regions(points)
    np.testing.assert_array_equal(torso, [False, False, False, True])
    np.testing.assert_array_equal(leg, [0, 1, 0, 0])
    np.testing.assert_array_equal(channel, [1, 3, 1, 1])
    fields = decoder.decode(decoder.base(), points)
    np.testing.assert_array_equal(fields.actuator_type, [1, 1, 1, 0])
    np.testing.assert_array_equal(fields.young, [80000.0, 80000.0, 80000.0, 90000.0])


@pytest.mark.parametrize("dim", [2, 3])
def test_box_record_projection_uses_xyz_schema_in_both_dimensions(dim: int) -> None:
    """A 2D recipe still stores three centre and half-size coordinates before radius."""
    recipe = make_recipe("flat", dim=dim)
    recipe["config"].update(
        kind="blocks", blocks=1, fixed_boxes=[[0.31, 0.02, -0.07, 0.04, 0.01, 0.25, 0.003]]
    )
    terrain = recipe_terrain(recipe)
    assert terrain["centers"] == [[0.31, 0.02, -0.07][:dim]]
    assert terrain["half_sizes"] == [[0.04, 0.01, 0.25][:dim]]
    assert terrain["rounding"] == [0.003]
    recipe["config"]["fixed_boxes"][0][-1] = 0.01
    with pytest.raises(ValueError, match="rounding must fit"):
        validate_recipe(recipe)


def test_chamber_profile_and_terrain_authoring_are_host_only() -> None:
    """Authoring modules import without loading accelerator or viewer packages."""
    code = """
import importlib.abc
import sys

class NoAccelerators(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'jaxlib', 'warp', 'newton'}:
            raise AssertionError('Unexpected import: ' + fullname)

sys.meta_path.insert(0, NoAccelerators())
from envs.soft_walker.presets.embodiments.chamber_profile import GEOMETRY
from envs.soft_walker.presets.terrain.editor_courses import make_recipe
from envs.soft_walker.presets.scenarios.config import Scenario
assert make_recipe('rough_blocks')['dim'] == Scenario().dim
assert GEOMETRY.leg_half_x > 0
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        env=dict(os.environ, PYTHONPATH=str(SOFT_ROOT.parents[1])),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
