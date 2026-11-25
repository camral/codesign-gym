"""Custom terrain replay plumbing; these checks do not integrate dynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType


import importlib.util
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from soft_paths import SOFT_ROOT, needs_scripts

from envs.soft_walker.engine.physics.terrain import generate, serialize
from envs.soft_walker.presets.scenarios.build import build
from envs.soft_walker.presets.scenarios.config import Scenario
from envs.soft_walker.presets.terrain.editor_courses import PRESETS, make_recipe
from envs.soft_walker.presets.terrain.scenario_courses import case_terrain, case_terrain_config
from envs.soft_walker.terrain.recipes import recipe_terrain, validate_recipe

ROOT = SOFT_ROOT


def recipe(kind: str = "blocks", dim: int = 2) -> dict[str, object]:
    config = (
        {"kind": "slopes", "blocks": 0, "fixed_slopes": [0.1, 0.04 if dim == 3 else 0.0]}
        if kind == "slopes"
        else {
            "kind": "blocks",
            "blocks": 2,
            "rounding": 0.001,
            "fixed_boxes": [
                [0.27, 0.0, 0.0, 0.04, 0.008, 0.25, 0.001],
                [0.35, 0.0, 0.0, 0.04, 0.016, 0.25, 0.001],
            ],
        }
    )
    return {
        "schema": "codesign_terrain_recipe_v1",
        "name": "Editable terrain",
        "dim": dim,
        "seed": 7,
        "config": config,
    }


def script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "terrain_test_" + name, ROOT / "scripts" / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("kind,dim", [("slopes", 2), ("blocks", 2), ("slopes", 3), ("blocks", 3)])
def test_custom_geometry_matches_export_and_is_frozen_across_reset_seeds(
    kind: str, dim: int, dtype: np.dtype[np.floating]
) -> None:
    case = Scenario(
        driver="mpm2" if dim == 2 else "mpm3",
        tier="smoke",
        dtype=dtype.name,
        terrain_recipe=recipe(kind, dim),
    )
    restored = Scenario.from_dict(json.loads(json.dumps(case.to_dict())))
    assert restored == case
    config = case_terrain_config(case)
    assert isinstance(hash(config), int)
    preview = case_terrain(case)
    for reset_seed in (0, 999):
        realized = generate(config, reset_seed, dim, dtype)
        for expected, actual in zip(preview, realized):
            np.testing.assert_array_equal(expected, actual)
    plain = recipe_terrain(case.terrain_recipe)
    for field, actual in serialize(preview).items():
        np.testing.assert_allclose(actual, plain[field], rtol=2e-7, atol=2e-8)


@pytest.mark.parametrize("driver", ["spring", "mpm2", "mpm3"])
def test_factory_uses_the_same_custom_config_without_reset(
    driver: str, dtype: np.dtype[np.floating]
) -> None:
    import jax

    dim = 3 if driver == "mpm3" else 2
    case = Scenario(
        driver=driver, tier="smoke", dtype=dtype.name, terrain_recipe=recipe("blocks", dim)
    )
    env, _ = build(case, device=jax.devices()[0])
    try:
        assert env.config.terrain == case_terrain_config(case)
        for expected, actual in zip(case_terrain(case), env._terrain):
            np.testing.assert_array_equal(expected, actual)
    finally:
        env.close()


def test_scenario_rejects_dimension_and_preset_ambiguity() -> None:
    with pytest.raises(ValueError):
        Scenario(driver="mpm2", terrain_recipe=recipe(dim=3))
    with pytest.raises(ValueError, match="named terrain"):
        Scenario(terrain="stairs", terrain_recipe=recipe())
    for invalid in (
        {**recipe(), "schema": "unknown"},
        {**recipe(), "config": {"kind": "blocks", "blocks": 1}},
        {**recipe(), "config": {"kind": "flat", "unknown_parameter": 1}},
    ):
        with pytest.raises(ValueError):
            validate_recipe(invalid)


@needs_scripts
def test_cli_recipe_is_embedded_and_terrain_switch_is_unambiguous(tmp_path: Path) -> None:
    cli = script("scenarios")
    path = tmp_path / "terrain.json"
    path.write_text(json.dumps(recipe()), encoding="utf-8")
    args = ["preview", "--terrain-file", str(path), "--out", str(tmp_path / "output")]
    edits = cli.overrides(cli.parser().parse_args(args))
    case = replace(Scenario(), **edits)
    assert case.terrain_recipe == validate_recipe(recipe())
    assert "custom_Editable_terrain" in case.key
    for extra in (["--terrain", "stairs"], ["--set", "terrain=flat"]):
        with pytest.raises(ValueError, match="cannot be combined"):
            cli.overrides(cli.parser().parse_args(args + extra))


@needs_scripts
@pytest.mark.parametrize("execution", [None, "resident"])
def test_suite_forwards_execution_choice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, execution: None | str
) -> None:
    runner = script("run_scenario_suite")
    case = tmp_path / "case.json"
    case.write_text(json.dumps(Scenario().to_dict()), encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"cases": [case.name]}), encoding="utf-8")
    commands = []

    def fake_run(command: Sequence[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    argv = [
        "run_scenario_suite.py",
        "--plan",
        str(plan),
        "--mode",
        "smoke",
        "--out",
        str(tmp_path / "suite"),
    ]
    if execution:
        argv += ["--execution", execution]
    monkeypatch.setattr(sys, "argv", argv)
    assert runner.main() == 0
    assert len(commands) == 1
    index = commands[0].index("--execution")
    assert commands[0][index + 1] == (execution or "legacy")
    assert ("--api-check" in commands[0]) == (execution != "resident")


def test_recipe_planning_import_does_not_initialize_jax() -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT.parents[1]))
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from envs.soft_walker.presets.scenarios.config import Scenario; "
            "from envs.soft_walker.terrain.recipes import validate_recipe; "
            "assert 'jax' not in sys.modules",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "preset,dim",
    [(name, dim) for dim in (2, 3) for name in PRESETS if name != "cross_slope" or dim == 3],
)
def test_all_editor_presets_match_physics_and_repeat_deterministically(
    preset: str, dim: int, dtype: np.dtype[np.floating]
) -> None:
    made = make_recipe(preset, dim=dim, seed=42)
    assert made == make_recipe(preset, dim=dim, seed=42)
    case = Scenario(driver="mpm2" if dim == 2 else "mpm3", dtype=dtype.name, terrain_recipe=made)
    physical = serialize(case_terrain(case))
    for field, expected in recipe_terrain(made).items():
        np.testing.assert_allclose(physical[field], expected, rtol=2e-7, atol=2e-8)
    changed = make_recipe(preset, dim=dim, seed=43)
    assert (made["config"] != changed["config"]) == (preset in ("low_blocks", "rough_blocks"))


@pytest.mark.parametrize(
    "options",
    [
        {"count": 0},
        {"count": 32},
        {"rounding": 0.0},
        {"rounding": 2.0, "height": 0.0001, "width": 0.001},
    ],
)
def test_generated_box_boundary_options_remain_valid(
    options: dict[str, object], dtype: np.dtype[np.floating]
) -> None:
    made = make_recipe("stairs", **options)
    case = Scenario(dtype=dtype.name, terrain_recipe=made)
    config = case_terrain_config(case)
    assert len(config.fixed_boxes) == made["config"]["blocks"]
    for box in config.fixed_boxes:
        assert 0 <= box[6] < min(box[3:6])
    assert all(np.isfinite(np.asarray(field)).all() for field in case_terrain(case))


def test_maximum_supported_slope_matches_physics(dtype: np.dtype[np.floating]) -> None:
    made = make_recipe("uphill", slope_degrees=math.degrees(math.atan(0.5)))
    case = Scenario(dtype=dtype.name, terrain_recipe=made)
    assert made["config"]["fixed_slopes"][0] == pytest.approx(0.5)
    for field, expected in recipe_terrain(made).items():
        np.testing.assert_allclose(
            serialize(case_terrain(case))[field], expected, rtol=2e-7, atol=2e-8
        )


@pytest.mark.parametrize(
    "options",
    [
        {"dim": 4},
        {"dim": True},
        {"seed": -1},
        {"seed": True},
        {"count": 33},
        {"count": 1.5},
        {"height": 0.0},
        {"rounding": -0.001},
        {"slope_degrees": 30.0},
        {"height": float("nan")},
        {"start": float("inf")},
    ],
)
def test_generator_rejects_invalid_controls(options: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_recipe("stairs", **options)


@pytest.mark.parametrize(
    "box",
    [
        [0.27, 0.0, 0.0, 0.04, 0.008, 0.25, 0.008],
        [0.27, 0.0, 0.0, 0.0, 0.008, 0.25, 0.001],
        [0.27, 0.0, 0.0, 0.04, 0.008, 0.25, float("nan")],
        [0.27, 0.0, 0.0, 0.04, 0.008, 0.25],
    ],
)
def test_custom_invalid_box_rounding_dimensions_and_finiteness(box: list[float]) -> None:
    invalid = recipe()
    invalid["config"]["fixed_boxes"][0] = box
    with pytest.raises(ValueError):
        validate_recipe(invalid)
