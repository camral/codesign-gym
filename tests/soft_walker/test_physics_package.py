"""Canonical physics imports, type identities, and lazy package boundaries."""

from __future__ import annotations

import importlib
import pickle
import subprocess
import sys

import pytest

_ENGINE = "envs.soft_walker.engine"
_MODULES = (
    "beam",
    "beam_tangent",
    "common",
    "contact",
    "fem",
    "local_tangent",
    "material",
    "mpm",
    "mpm_support",
    "mpm_tangent",
    "physics_constants",
    "spring",
    "spring_tangent",
    "spring_config",
    "terrain",
)
_MOVED_NUMERICS = ("integrators", "krylov", "solvers")
_REMOVED_MODULES = (
    "spring",
    "beam",
    "spring_tangent",
    "beam_tangent",
    "physics_constants",
    "continuum",
    "structural",
    "config",
    "terrain",
)
_RECORDS = (
    ("spring", "Topology"),
    ("spring", "Model"),
    ("spring", "State"),
    ("spring", "Diagnostics"),
    ("beam", "Topology"),
    ("beam", "Model"),
    ("beam", "State"),
    ("common", "StepInfo"),
    ("fem", "Mesh"),
    ("fem", "Model"),
    ("fem", "State"),
    ("mpm", "Cloud"),
    ("mpm", "Model"),
    ("mpm", "State"),
    ("mpm", "Stencil"),
    ("mpm_support", "CompactSupport"),
    ("terrain", "Terrain"),
)


def test_physics_package_and_constants_do_not_initialize_accelerators() -> None:
    """The package boundary remains usable before selecting an execution backend."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import sys; from {_ENGINE}.physics import physics_constants; "
                "assert not {'jax', 'warp', 'newton'}.intersection(sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("module_name", _MODULES)
def test_physics_modules_use_their_canonical_names(module_name: str) -> None:
    """The physics package owns each implementation module directly."""
    path = f"{_ENGINE}.physics.{module_name}"
    module = importlib.import_module(path)
    assert module.__name__ == path
    assert module.__package__ == f"{_ENGINE}.physics"
    assert sys.modules[path] is module
    assert getattr(importlib.import_module(f"{_ENGINE}.physics"), module_name) is module


@pytest.mark.parametrize("canonical_first", (True, False))
def test_removed_physics_imports_remain_absent(canonical_first: bool) -> None:
    """Loading canonical modules does not recreate removed engine module paths."""
    script = f"""
import importlib, importlib.util
engine = importlib.import_module({_ENGINE!r})
if {canonical_first!r}:
    for name in {_MODULES!r}:
        importlib.import_module({_ENGINE!r} + '.physics.' + name)
    for name in {_MOVED_NUMERICS!r}:
        importlib.import_module({_ENGINE!r} + '.numerics.' + name)
for name in {_REMOVED_MODULES!r}:
    legacy_path = {_ENGINE!r} + '.' + name
    assert importlib.util.find_spec(legacy_path) is None
    try:
        importlib.import_module(legacy_path)
    except ModuleNotFoundError as error:
        assert error.name == legacy_path
    else:
        raise AssertionError(legacy_path)
    assert not hasattr(engine, name)
physics = importlib.import_module({_ENGINE!r} + '.physics')
for name in {_MOVED_NUMERICS!r}:
    old_path = physics.__name__ + '.' + name
    assert importlib.util.find_spec(old_path) is None
    try:
        importlib.import_module(old_path)
    except ModuleNotFoundError as error:
        assert error.name == old_path
    else:
        raise AssertionError(old_path)
    assert not hasattr(physics, name)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("module_name", "record_name"), _RECORDS)
def test_physics_records_round_trip_through_canonical_pickle_paths(
    module_name: str, record_name: str
) -> None:
    """New Python records identify their defining physics module."""
    canonical_path = f"{_ENGINE}.physics.{module_name}"
    canonical = importlib.import_module(canonical_path)
    record = getattr(canonical, record_name)
    assert record.__module__ == canonical_path
    assert pickle.loads(pickle.dumps(record)) is record
    value = record(*range(len(record._fields)))
    restored = pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    assert type(restored) is record
    assert restored == value
    assert restored._fields == value._fields


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    (
        ("common", "DynamicsConfig"),
        ("contact", "PlaneContact"),
        ("mpm", "Grid"),
    ),
)
def test_physics_configs_round_trip_through_canonical_pickle_paths(
    module_name: str, class_name: str
) -> None:
    """New immutable numerical profiles use their actual defining module."""
    module_path = f"{_ENGINE}.physics.{module_name}"
    cls = getattr(importlib.import_module(module_path), class_name)
    assert cls.__module__ == module_path
    assert pickle.loads(pickle.dumps(cls)) is cls
    value = cls()
    restored = pickle.loads(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    assert type(restored) is cls
    assert restored == value


@pytest.mark.parametrize(
    ("module_name", "attribute", "consumer"),
    (
        ("spring", "advance", "kernels"),
        ("beam", "strains", "link_energy"),
        ("spring_tangent", "local_force", "edge_forces"),
        ("beam_tangent", "local_load", "elastic_force_torque"),
    ),
)
def test_canonical_monkeypatch_updates_function_globals(
    module_name: str, attribute: str, consumer: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A helper override reaches consumers in the same implementation module."""
    canonical = importlib.import_module(f"{_ENGINE}.physics.{module_name}")
    replacement = object()
    monkeypatch.setattr(canonical, attribute, replacement)
    function = getattr(canonical, consumer)
    function = getattr(function, "__wrapped__", function)
    assert getattr(canonical, attribute) is replacement
    assert function.__globals__[attribute] is replacement


def test_spring_kernel_factory_reuses_its_cached_result() -> None:
    """Repeated canonical calls reuse the existing executable factory cache."""
    from envs.soft_walker.engine.physics.spring_config import SpringConfig

    canonical = importlib.import_module(f"{_ENGINE}.physics.spring")
    config = SpringConfig(grid_shape=(1, 1), substeps=1, settle_steps=0)
    first = canonical.kernels(config)
    hits = canonical.kernels.cache_info().hits
    assert canonical.kernels(config) is first
    assert canonical.kernels.cache_info().hits == hits + 1
