"""Keep current guide links and explicit Python imports tied to live source owners.

Historical reports deliberately retain revision-specific paths and are excluded.
The separate Sphinx build verifies generated references without importing devices.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest
from packaging.requirements import Requirement

REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE = REPOSITORY / "src/envs/soft_walker"
CURRENT_GUIDES = (
    PACKAGE / "README.md",
    PACKAGE / "viewer/README.md",
    *(
        PACKAGE / "docs" / name
        for name in (
            "COMPONENTS_AND_GYMNASIUM.md",
            "CAPABILITIES.md",
            "MODULARITY_MATRIX.md",
            "INTEGRATOR_SELECTION.md",
            "NATIVE_CALLBACKS.md",
            "EMBODIMENT_NATIVE_GRADIENTS.md",
            "EMBODIMENT_DEFAULTS.md",
            "SPATIAL_MATERIALS_AND_TERRAIN.md",
            "ACTION_HOLD.md",
            "PARALLEL_ENVIRONMENTS.md",
            "RANDOM_CREATURES.md",
            "TERRAIN_DEMO.md",
            "DOCUMENTATION.md",
            "HISTORY.md",
            "RELEASE_ORGANIZATION.md",
            "SHARED_HELPER_MIGRATION.md",
        )
    ),
)


def test_preview_pillow_minimum_supports_sized_default_fonts() -> None:
    """The declared render dependency must cover load_default(size=...), added in 10.1."""
    metadata = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = [Requirement(value) for value in metadata["project"]["dependencies"]]
    pillow = next(
        requirement for requirement in dependencies if requirement.name.lower() == "pillow"
    )
    assert "10.0.1" not in pillow.specifier
    assert "10.1.0" in pillow.specifier


@pytest.mark.parametrize("guide", CURRENT_GUIDES, ids=lambda path: path.name)
def test_current_guide_local_link_targets_exist(guide: Path) -> None:
    """Detect stale source/test owners without rewriting historical linked content."""
    text = guide.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^\]\n]+\]\(([^\s)]+)\)", text):
        target = target.strip("<>")
        parts = urlsplit(target)
        if parts.scheme or not parts.path:
            continue
        resolved = guide.parent / unquote(parts.path)
        assert resolved.exists(), f"{guide.relative_to(REPOSITORY)} links to missing {target}"


@pytest.mark.parametrize("guide", CURRENT_GUIDES, ids=lambda path: path.name)
def test_current_guide_python_blocks_parse_and_import_live_modules(guide: Path) -> None:
    """Catch removed package imports without executing simulations from examples."""
    text = guide.read_text(encoding="utf-8")
    for block in re.findall(r"```(?:python|py)\s*\n(.*?)```", text, flags=re.DOTALL):
        tree = ast.parse(block, filename=str(guide))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
        for module in modules:
            if not module.startswith("envs.soft_walker"):
                continue
            owner = REPOSITORY / "src" / Path(*module.split("."))
            assert owner.with_suffix(".py").is_file() or owner.is_dir(), (
                f"{guide.relative_to(REPOSITORY)} imports removed module {module}"
            )
