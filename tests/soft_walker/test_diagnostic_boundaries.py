"""Canonical diagnostic entry points preserve offline imports and source provenance."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from soft_paths import REPO_ROOT, SOFT_ROOT

from envs.soft_walker import shared as _paths
from envs.soft_walker.diagnostics import run_resident_tests


def test_canonical_helpers_and_entrypoint_share_the_module_namespace() -> None:
    """Runner hooks affect the imported implementation without executing copied source."""
    module = importlib.import_module("envs.soft_walker.diagnostics.run_resident_tests")
    assert module is run_resident_tests
    assert module.main.__globals__ is vars(module)
    assert module.planned_stages.__globals__ is vars(module)
    assert module.ROOT == REPO_ROOT
    assert Path(module.main.__code__.co_filename).parent == _paths.DIAGNOSTICS_ROOT


def test_diagnostic_manifest_hashes_canonical_implementations_and_live_contracts() -> None:
    """Every source entry exists and includes the runner's actual host dependencies."""
    hashes = run_resident_tests.source_hashes()
    normalized = {name.replace("\\", "/") for name in hashes}
    assert "src/envs/soft_walker/diagnostics/run_resident_tests.py" in normalized
    assert "src/envs/soft_walker/shared.py" in normalized
    for module in ("config", "embodiment", "render", "render3d", "vector_env", "registration"):
        assert f"src/envs/soft_walker/{module}.py" in normalized
    for package, modules in (
        (
            "contracts",
            (
                "__init__",
                "fields",
                "observations",
                "metrics",
                "batching",
                "api_types",
                "status_codes",
            ),
        ),
        ("diagnostics/residency", ("__init__", "probes", "fixtures")),
        ("diagnostics/warp", ("__init__", "checks", "audit", "solid_fixtures")),
    ):
        for module in modules:
            assert f"src/envs/soft_walker/{package}/{module}.py" in normalized
    assert all((REPO_ROOT / name).is_file() for name in normalized)


def test_diagnostic_source_inventory_uses_only_explicit_package_roots(tmp_path: Path) -> None:
    """Relocated host dependencies are included without collecting nested evidence."""
    package = tmp_path / "src/envs/soft_walker"
    included = (
        "diagnostics/runner.py",
        "config.py",
        "env.py",
        "contracts/fields.py",
        "diagnostics/residency/probes.py",
        "diagnostics/warp/checks.py",
    )
    excluded = (
        "runs/old_source.py",
        "handoff/old_source.py",
        "registered/runs/old_source.py",
        "contracts/handoff/old_source.py",
        "diagnostics/runs/old_source.py",
        "diagnostics/residency/runs/old_source.py",
        "diagnostics/warp/handoff/old_source.py",
    )
    for name in (*included, *excluded):
        source = package / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("", encoding="utf-8")
    actual = _paths.diagnostic_sources(tmp_path)
    assert [path.relative_to(package).as_posix() for path in actual] == list(included)


@pytest.mark.parametrize("invocation", ("module", "file"))
@pytest.mark.parametrize(
    "name",
    (
        "run_validation",
        "run_resident_tests",
        "run_locomotion_tests",
        "run_scenario_tests",
        "review_resident_evidence",
    ),
)
def test_canonical_help_from_an_unrelated_directory(
    tmp_path: Path, name: str, invocation: str
) -> None:
    """Module and direct-file commands parse help without a working-directory dependency."""
    environment = dict(os.environ)
    if invocation == "module":
        environment["PYTHONPATH"] = str(REPO_ROOT / "src")
        command = [sys.executable, "-m", f"envs.soft_walker.diagnostics.{name}", "--help"]
    else:
        # Direct-file execution must establish its own checkout import root.
        environment.pop("PYTHONPATH", None)
        command = [sys.executable, str(SOFT_ROOT / "diagnostics" / f"{name}.py"), "--help"]
    completed = subprocess.run(
        command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "usage:" in completed.stdout.lower()
    assert not list(tmp_path.iterdir()), "Help must not create run artifacts"


def test_path_helpers_do_not_initialize_backends_or_change_import_paths(tmp_path: Path) -> None:
    """Reading canonical paths has no device imports or interpreter search-path side effects."""
    code = (
        "import sys,json; "
        f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r}); "
        "before=list(sys.path); "
        "from envs.soft_walker import shared as _paths; "
        "assert sys.path == before; "
        "print(json.dumps({key:key in sys.modules for key in ('jax','warp','newton')}))"
    )
    output = subprocess.check_output([sys.executable, "-c", code], cwd=tmp_path, text=True)
    assert json.loads(output) == {"jax": False, "warp": False, "newton": False}
