"""Keep the engine independent of experiment protocols and host authoring backend-free."""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def test_engine_does_not_import_workflows_or_presentation() -> None:
    """Catch both eager and function-local backedges, including type-only imports."""
    root = Path(__file__).resolve().parents[2] / "src/envs/soft_walker"
    forbidden = tuple(
        "envs.soft_walker." + name for name in ("experiments", "presets", "diagnostics", "viewer")
    )
    failures = []
    for path in (root / "engine").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        name = "envs.soft_walker." + path.relative_to(root).with_suffix("").as_posix().replace(
            "/", "."
        )
        package = (
            name.removesuffix(".__init__")
            if path.name == "__init__.py"
            else name.rpartition(".")[0]
        )
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = (
                    importlib.util.resolve_name("." * node.level + (node.module or ""), package)
                    if node.level
                    else node.module or ""
                )
                names = [base, *(base + "." + alias.name for alias in node.names)]
            for imported in names:
                if any(
                    imported == prefix or imported.startswith(prefix + ".") for prefix in forbidden
                ):
                    failures.append(f"{path.relative_to(root)}:{node.lineno}: {imported}")
    assert failures == []


def test_recipe_authoring_and_scenario_planning_do_not_load_backends() -> None:
    """Validate portable courses/configuration in a process rejecting accelerator imports."""
    source = """
import importlib.abc
import sys
class RejectBackends(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'jaxlib', 'warp', 'newton'}:
            raise AssertionError('Unexpected backend import: ' + fullname)
sys.meta_path.insert(0, RejectBackends())
from envs.soft_walker.terrain.recipes import recipe_terrain, validate_recipe
from envs.soft_walker.presets.terrain.editor_courses import make_recipe
from envs.soft_walker.presets.scenarios.config import Scenario
from envs.soft_walker.presets.scenarios.suites import suite
recipe = make_recipe('stairs', dim=3, seed=7)
assert validate_recipe(recipe) == recipe
assert len(recipe_terrain(recipe)['centers']) == recipe['config']['blocks']
case = Scenario(driver='mpm3', terrain_recipe=recipe)
assert Scenario.from_dict(case.to_dict()).to_dict() == case.to_dict()
assert len(suite('drivers')) == 5
"""
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(repo / "src"), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-c", source], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_specialized_episode_adapters_are_owned_by_workflows() -> None:
    """The engine exposes the generic callback mechanism, not named demonstration protocols."""
    from envs.soft_walker.engine import execution
    from envs.soft_walker.experiments.locomotion.resident import bind_demo_episode
    from envs.soft_walker.experiments.scenarios.resident import bind_scenario_episode

    assert "bind_episode" in execution.__all__
    assert "bind_demo_episode" not in execution.__all__
    assert "bind_scenario_episode" not in execution.__all__
    assert bind_demo_episode.__module__ == "envs.soft_walker.experiments.locomotion.resident"
    assert bind_scenario_episode.__module__ == "envs.soft_walker.experiments.scenarios.resident"
