"""Inspect built release artifacts without installing unrelated simulator dependencies.

Set CODESIGN_DISTRIBUTION_DIR to a directory containing a freshly built wheel and
sdist, then run this file. Ordinary source tests skip these artifact-only checks.
"""

from __future__ import annotations

import json
import os
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE_PREFIX = "envs/soft_walker/"
EXCLUDED_SOURCE_PARTS = {"runs", "handoff", "__pycache__"}


@pytest.fixture(scope="module")
def artifacts() -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Read one wheel/sdist pair; archive extraction is unnecessary for these checks."""
    directory = os.environ.get("CODESIGN_DISTRIBUTION_DIR")
    if directory is None:
        pytest.skip("Set CODESIGN_DISTRIBUTION_DIR to validate built release artifacts")
    root = Path(directory)
    wheels, sdists = sorted(root.glob("*.whl")), sorted(root.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1, "Provide one matching wheel and sdist"
    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel_files = {
            name: wheel.read(name) for name in wheel.namelist() if not name.endswith("/")
        }
    with tarfile.open(sdists[0]) as sdist:
        source_files = {}
        for member in sdist.getmembers():
            if not member.isfile():
                continue
            stream = sdist.extractfile(member)
            assert stream is not None
            source_files[member.name.split("/", 1)[1]] = stream.read()
    return wheel_files, source_files


def test_wheel_contains_current_soft_walker_modules(
    artifacts: tuple[dict[str, bytes], dict[str, bytes]],
) -> None:
    """All current runtime/CLI Python files must survive packaging byte for byte."""
    wheel_files, _ = artifacts
    source = REPOSITORY / "src"
    for path in (source / PACKAGE_PREFIX).rglob("*.py"):
        relative = path.relative_to(source)
        if EXCLUDED_SOURCE_PARTS.intersection(relative.parts):
            continue
        assert wheel_files[relative.as_posix()] == path.read_bytes(), relative


def test_artifacts_exclude_historical_results_and_generated_content(
    artifacts: tuple[dict[str, bytes], dict[str, bytes]],
) -> None:
    """Release installation archives retain small references, not historical run trees."""
    forbidden = {".git", ".codex", ".agents", "__pycache__", ".pytest_cache", ".mypy_cache"}
    for archive in artifacts:
        for name in archive:
            assert not forbidden.intersection(Path(name).parts), name
            assert f"{PACKAGE_PREFIX}runs/" not in name, name
            assert f"{PACKAGE_PREFIX}handoff/" not in name, name


def test_runtime_reference_inputs_and_viewer_guides_are_packaged(
    artifacts: tuple[dict[str, bytes], dict[str, bytes]],
) -> None:
    """Installed tools retain their example inputs and local viewer documentation."""
    wheel_files, _ = artifacts
    for relative in (
        "examples/scenarios/drivers/plan.json",
        "examples/terrain_demo/flat.recipe.json",
        "baselines/direct_2d_float32.json",
    ):
        assert json.loads(wheel_files[PACKAGE_PREFIX + relative])
    for relative in (
        "baselines/warp_followup/spring_biped/model.npz",
        "baselines/warp_followup/spring_biped/commands.npy",
        "viewer/README.md",
        "viewer/terrain_demo.md",
    ):
        assert wheel_files[PACKAGE_PREFIX + relative]


def test_sdist_contains_tests_fixtures_and_typing(
    artifacts: tuple[dict[str, bytes], dict[str, bytes]],
) -> None:
    """An sdist must preserve nested numerical regressions and source-only tooling."""
    _, source_files = artifacts
    paths = [REPOSITORY / "mypy-soft.ini"]
    paths.extend(
        path
        for path in (REPOSITORY / "tests").rglob("*")
        if path.is_file() and path.suffix in {".py", ".json"} and "__pycache__" not in path.parts
    )
    for path in paths:
        relative = path.relative_to(REPOSITORY).as_posix()
        assert source_files[relative] == path.read_bytes(), relative


def test_wheel_dependency_metadata_is_parseable_and_preserves_required_backends(
    artifacts: tuple[dict[str, bytes], dict[str, bytes]],
) -> None:
    """Parse every dependency marker and preserve declared CPU/native requirements."""
    wheel_files, _ = artifacts
    metadata_name = next(name for name in wheel_files if name.endswith(".dist-info/METADATA"))
    metadata = BytesParser().parsebytes(wheel_files[metadata_name])
    requirements = [Requirement(value) for value in metadata.get_all("Requires-Dist", [])]
    core = {
        requirement.name: requirement for requirement in requirements if requirement.marker is None
    }
    assert {"jax", "warp-lang", "numpy", "gymnasium", "pillow"} <= core.keys()
    assert "10.0.1" not in core["pillow"].specifier
    assert metadata["Requires-Python"] == ">=3.11"
