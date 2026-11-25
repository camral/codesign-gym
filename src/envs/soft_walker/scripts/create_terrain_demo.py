#!/usr/bin/env python3
"""Create editable terrain recipes and explicit legacy scenario plans.

No dynamics or graphics are run by this command. The Newton editor can change
the same geometry and the scenario CLI can replay the resulting recipes.
"""

from __future__ import annotations

import argparse

# Resolve integrated imports independently of the caller's working directory.
import sys as _script_sys
from pathlib import Path, Path as _ScriptPath
from typing import Any

_source_root = str(_ScriptPath(__file__).resolve().parents[3])
if _source_root not in _script_sys.path:
    _script_sys.path.insert(0, _source_root)
_script_dir = str(_ScriptPath(__file__).resolve().parent)
if _script_dir not in _script_sys.path:
    _script_sys.path.insert(0, _script_dir)
import json
import sys

from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
)

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    """Command entry point. Create editable terrain recipes and explicit legacy scenario plans."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=SOFT_ROOT / "examples/terrain_demo")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be fresh")
    from envs.soft_walker.presets.scenarios.config import Scenario
    from envs.soft_walker.presets.terrain.editor_courses import PRESETS, make_recipe
    from envs.soft_walker.terrain.recipes import recipe_terrain

    output.mkdir(parents=True, exist_ok=True)

    def write(name: str, data: Any) -> None:
        (output / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    cases = []
    for preset in PRESETS:
        dim = 3 if preset == "cross_slope" else 2
        recipe = make_recipe(preset, dim=dim, seed=args.seed)
        write(f"{preset}.recipe.json", recipe)
        write(f"{preset}.terrain.json", recipe_terrain(recipe))
        case = Scenario(
            driver="mpm3" if dim == 3 else "mpm2",
            terrain_recipe=recipe,
            tier="smoke",
            dtype="float64",
            seed=args.seed,
        )
        filename = f"{preset}.case.json"
        write(filename, case.to_dict())
        cases.append(filename)
    plan = dict(
        schema="codesign_scenario_plan_v1",
        suite="editable_terrains",
        cases=cases,
        execution="legacy",
        status="geometry recipes; no dynamics executed",
        design_mode="hard",
        material_mode="continuous",
    )
    write("plan.json", plan)
    write("smoke_plan.json", dict(plan, cases=["uphill.case.json", "stairs.case.json"]))
    print(f"Saved eight terrain recipes and plans to {output}")


if __name__ == "__main__":
    main()
