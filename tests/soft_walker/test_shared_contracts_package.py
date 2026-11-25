"""Shared contracts stay canonical, host-only, and independent of adapters."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = "envs.soft_walker"
_MODULES = ("fields", "observations", "metrics", "batching", "api_types", "status_codes")
_REMOVED = (
    "field_contracts",
    "observation_layouts",
    "metric_layouts",
    "batch_contracts",
    "api_types",
)
_SOURCE = Path(__file__).resolve().parents[2] / "src"


@pytest.mark.parametrize("name", _MODULES)
def test_contract_modules_use_canonical_paths(name: str) -> None:
    """Each implementation lives directly beneath the shared contracts package."""
    path = f"{_ROOT}.contracts.{name}"
    module = importlib.import_module(path)
    assert module.__name__ == path
    assert module.__package__ == f"{_ROOT}.contracts"
    assert getattr(importlib.import_module(f"{_ROOT}.contracts"), name) is module
    assert sys.modules[path] is module


@pytest.mark.parametrize("canonical_first", (False, True))
def test_old_contract_modules_remain_absent(canonical_first: bool) -> None:
    """Loading the new package never restores removed top-level import paths."""
    script = f"""
import importlib, importlib.util
package = importlib.import_module({_ROOT!r})
if {canonical_first!r}:
    for name in {_MODULES!r}:
        importlib.import_module({_ROOT!r} + '.contracts.' + name)
for name in {_REMOVED!r}:
    path = {_ROOT!r} + '.' + name
    assert importlib.util.find_spec(path) is None
    try:
        importlib.import_module(path)
    except ModuleNotFoundError as error:
        assert error.name == path
    else:
        raise AssertionError(path)
    assert not hasattr(package, name)
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_contract_package_and_modules_keep_host_only_imports() -> None:
    """The namespace is empty and concrete contracts need at most NumPy."""
    script = f"""
import importlib, sys
importlib.import_module({_ROOT!r} + '.contracts')
assert 'numpy' not in sys.modules
for name in {_MODULES!r}:
    importlib.import_module({_ROOT!r} + '.contracts.' + name)
assert not {{'jax', 'jaxlib', 'warp', 'newton', 'diffrax'}}.intersection(sys.modules)
assert not any(name == {_ROOT!r} + '.engine' or name.startswith({_ROOT!r} + '.engine.') for name in sys.modules)
assert not {{{_ROOT!r} + '.' + m for m in ('config', 'embodiment', 'vector_env', 'registration')}}.intersection(sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


def test_host_aliases_retain_shared_type_objects() -> None:
    """Moving owners keeps the established wrapper and batching type aliases."""
    from envs.soft_walker.contracts import api_types, batching, fields

    assert api_types.HostArray is fields.HostArray
    assert api_types.Fields is fields.HostFields
    assert batching.HostArray is fields.HostArray
    assert batching.EmbodimentInput is api_types.EmbodimentInput


@pytest.mark.parametrize(
    ("module_name", "class_name", "args"),
    (
        ("fields", "RawFieldLayout", (3, 4)),
        ("observations", "ObservationLayout", ((("position", 6), ("ledger", 4)),)),
        ("metrics", "ScenarioMetricLayout", (3,)),
    ),
)
def test_layout_records_pickle_through_canonical_modules(
    module_name: str, class_name: str, args: tuple[object, ...]
) -> None:
    """New host records retain values while identifying their defining module."""
    path = f"{_ROOT}.contracts.{module_name}"
    cls = getattr(importlib.import_module(path), class_name)
    assert cls.__module__ == path
    assert pickle.loads(pickle.dumps(cls)) is cls
    value = cls(*args)
    restored = pickle.loads(pickle.dumps(value))
    assert type(restored) is cls
    assert restored == value


def test_api_timing_schema_keeps_its_fields_and_canonical_identity() -> None:
    """Registered timing remains the same host TypedDict schema."""
    from envs.soft_walker.contracts.api_types import EnvironmentTiming

    assert EnvironmentTiming.__module__ == f"{_ROOT}.contracts.api_types"
    assert pickle.loads(pickle.dumps(EnvironmentTiming)) is EnvironmentTiming
    assert EnvironmentTiming.__required_keys__ == {
        "substeps",
        "settling_steps",
        "horizon",
        "length_ref",
        "work_ref",
        "energy_weight",
    }
    assert not EnvironmentTiming.__optional_keys__


def _imports(path: Path) -> list[tuple[int, str]]:
    """Resolve explicit imports without importing an optional runtime."""
    parts = path.relative_to(_SOURCE).with_suffix("").parts
    package = ".".join(parts[:-1])
    result = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            result.extend((node.lineno, name.name) for name in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = "." * node.level + (node.module or "")
            resolved = importlib.util.resolve_name(base, package) if node.level else base
            result.append((node.lineno, resolved))
            result.extend((node.lineno, resolved + "." + name.name) for name in node.names)
    return result


def _under(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def test_shared_contracts_do_not_depend_on_engines_or_registered_tasks() -> None:
    """Contracts can serve every backend and host tool without importing a consumer."""
    violations = []
    for path in (_SOURCE / "envs/soft_walker/contracts").glob("*.py"):
        for line, name in _imports(path):
            if (
                _under(name, _ROOT)
                and not _under(name, _ROOT + ".contracts")
                or name.split(".")[0] in {"jax", "jaxlib", "warp", "newton", "diffrax"}
            ):
                violations.append((path.name, line, name))
    assert not violations


def test_reusable_engine_does_not_depend_on_registered_adapters() -> None:
    """The engine may consume shared contracts but never task registration or wrappers."""
    forbidden = tuple(
        f"{_ROOT}.{name}"
        for name in ("registration", "vector_env", "config", "embodiment", "render", "render3d")
    )
    violations = []
    for path in (_SOURCE / "envs/soft_walker/engine").rglob("*.py"):
        for line, name in _imports(path):
            if any(_under(name, prefix) for prefix in forbidden):
                violations.append((str(path.relative_to(_SOURCE)), line, name))
    assert not violations
