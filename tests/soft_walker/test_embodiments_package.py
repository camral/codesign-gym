"""Canonical embodiment imports, Python records, and lazy package boundaries."""

from __future__ import annotations

import importlib
import json
import pickle
import subprocess
import sys

import numpy as np
import pytest

from envs.soft_walker.engine.embodiments import morphology, priors

ENGINE = "envs.soft_walker.engine"
MODULE_NAMES = ("decoders", "fields", "constants", "morphology", "priors", "precedents")
REMOVED_MODULES = (
    "design",
    "embodiment_decoders",
    "design_fields",
    "decoder_constants",
    "morphology",
    "morphology_priors",
    "precedents",
)
CLASS_PATHS = (
    *(
        ("decoders", name)
        for name in (
            "Fields",
            "FieldSpec",
            "Decoder",
            "GridDecoder",
            "BSplineDecoder",
            "CPPNDecoder",
            "PrimitiveDecoder",
        )
    ),
    *(
        ("priors", name)
        for name in (
            "PropertyPrior",
            "PrimitivePrior",
            "DirectPrior",
            "BSplinePrior",
            "CPPNPrior",
        )
    ),
    ("morphology", "GenerationError"),
)


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_embodiment_modules_use_their_canonical_names(module_name: str) -> None:
    """The embodiments package owns each implementation module directly."""
    path = f"{ENGINE}.embodiments.{module_name}"
    canonical = importlib.import_module(path)
    assert canonical.__name__ == path
    assert canonical.__package__ == f"{ENGINE}.embodiments"
    assert sys.modules[path] is canonical
    assert getattr(importlib.import_module(f"{ENGINE}.embodiments"), module_name) is canonical


@pytest.mark.parametrize("canonical_first", (True, False))
def test_removed_embodiment_imports_remain_absent(canonical_first: bool) -> None:
    """Removed paths fail regardless of whether canonical modules are loaded."""
    script = f"""
import importlib, importlib.util
engine = {ENGINE!r}
if {canonical_first!r}:
    for name in {MODULE_NAMES!r}:
        importlib.import_module(f'{{engine}}.embodiments.{{name}}')
for name in {REMOVED_MODULES!r}:
    path = f'{{engine}}.{{name}}'
    assert importlib.util.find_spec(path) is None
    try:
        importlib.import_module(path)
    except ModuleNotFoundError as error:
        assert error.name == path
    else:
        raise AssertionError(path)
    assert not hasattr(importlib.import_module(engine), name)
"""
    subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)


@pytest.mark.parametrize(("module_name", "class_name"), CLASS_PATHS)
def test_class_pickle_uses_its_canonical_module(module_name: str, class_name: str) -> None:
    """New class references round-trip through their defining embodiment modules."""
    module_path = f"{ENGINE}.embodiments.{module_name}"
    cls = getattr(importlib.import_module(module_path), class_name)
    assert cls.__module__ == module_path
    assert pickle.loads(pickle.dumps(cls)) is cls


@pytest.mark.parametrize(
    "name", ("PROPERTY_PRIOR", "PRIMITIVE_PRIOR", "DIRECT_PRIOR", "BSPLINE_PRIOR", "CPPN_PRIOR")
)
def test_prior_values_round_trip_through_the_canonical_module(name: str) -> None:
    """A new dataclass record retains values and its canonical class identity."""
    prior = getattr(priors, name)
    assert type(prior).__module__ == f"{ENGINE}.embodiments.priors"
    restored = pickle.loads(pickle.dumps(prior))
    assert type(restored) is type(prior)
    assert restored == prior


def test_canonical_monkeypatch_reaches_function_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sampler setting override affects its canonical implementation."""
    monkeypatch.setattr(morphology, "DIRECTION_NORM_FLOOR", np.inf)
    np.testing.assert_array_equal(morphology._unit(np.random.default_rng(3), 2), np.zeros(2))


def test_embodiments_package_and_host_settings_do_not_load_accelerators() -> None:
    """The package, numeric defaults and random priors remain safe for host-only tooling."""
    script = """
import json, sys
from envs.soft_walker.engine import embodiments
from envs.soft_walker.engine.embodiments import constants, priors
assert embodiments.__name__ == 'envs.soft_walker.engine.embodiments'
assert constants.__package__ == priors.__package__ == embodiments.__name__
print(json.dumps({name: name in sys.modules for name in ('jax', 'warp', 'newton')}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert json.loads(completed.stdout) == {"jax": False, "warp": False, "newton": False}
