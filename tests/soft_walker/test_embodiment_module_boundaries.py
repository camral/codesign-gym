"""Canonical decoder identities, schema ownership and lazy host imports."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys

import numpy as np
import pytest

from envs.soft_walker.contracts import api_types
from envs.soft_walker.contracts.fields import (
    DENSITY_BOUNDS,
    POISSON_BOUNDS,
    YOUNG_BOUNDS,
    HostArray,
    HostFields,
)
from envs.soft_walker.engine.embodiments import decoders
from envs.soft_walker import config as task_presets, embodiment
from envs.soft_walker import config as bundle_builders


@pytest.mark.parametrize("name", ["direct", "bspline", "cppn", "primitives"])
def test_decoder_pickles_use_canonical_module(name: str) -> None:
    """New decoder pickles resolve directly to their implementation module."""
    decoder = decoders.make_decoder(name)
    assert type(decoder).__module__ == "envs.soft_walker.engine.embodiments.decoders"
    restored = pickle.loads(pickle.dumps(decoder))
    assert type(restored) is type(decoder)
    assert restored.descriptor() == decoder.descriptor()
    for key, values in decoder.base().items():
        np.testing.assert_array_equal(restored.base()[key], values)


def test_fields_and_specs_round_trip_with_canonical_identity() -> None:
    """Dataclass and NamedTuple serialization retains fields, defaults and class identity."""
    spec = decoders.FieldSpec(dim=3, channels=4)
    assert pickle.loads(pickle.dumps(spec)) == spec
    scalar = np.ones(2)
    fields = decoders.Fields(scalar, scalar, scalar, scalar, np.ones((2, 3)), scalar, scalar)
    restored = pickle.loads(pickle.dumps(fields))
    assert type(restored) is decoders.Fields and restored.poisson is None
    assert restored._fields == fields._fields


def test_shared_host_schema_keeps_custom_decoder_bounds() -> None:
    """Shared defaults do not force a custom decoder to use the entire envelope."""
    assert api_types.HostArray is HostArray
    assert api_types.Fields is HostFields
    spec = decoders.FieldSpec()
    assert spec.young_bounds == embodiment._BOUNDS["young"] == YOUNG_BOUNDS
    assert spec.density_bounds == embodiment._BOUNDS["density"] == DENSITY_BOUNDS
    assert spec.poisson_bounds == embodiment._BOUNDS["poisson"] == POISSON_BOUNDS
    narrowed = decoders.FieldSpec(young_bounds=(10000.0, 90000.0), poisson_bounds=(0.3, 0.4))
    assert narrowed.poisson_bounds == (0.3, 0.4)
    assert embodiment._BOUNDS["poisson"] == (0.25, 0.45)


def test_task_overrides_reach_timing_and_bundle_assembly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Canonical task overrides affect dimensions, timing and native grid assembly."""
    original_grid, _ = bundle_builders.mpm_grid("mpm2")
    original_body_grid = task_presets.grid("mpm2")
    monkeypatch.setattr(task_presets, "BASE_GRID", {2: (2, 1), 3: (2, 1, 1)})
    monkeypatch.setattr(task_presets, "CONTROL_DT", 0.0004)
    monkeypatch.setattr(task_presets, "SUBSTEPS", 4)
    monkeypatch.setattr(task_presets, "SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(task_presets, "HORIZON", 6)
    assert task_presets.grid("spring") == (2, 1)
    assert len(task_presets.base_props("spring")["occupancy"]) == 2
    timing = task_presets.env_kwargs("spring", physics_steps_per_action=10)
    assert timing["substeps"] == 10 and timing["horizon"] == 3
    assert timing["settling_steps"] == 0
    changed_grid, _ = bundle_builders.mpm_grid("mpm2")
    np.testing.assert_array_equal(
        np.subtract(original_grid, changed_grid),
        np.subtract(original_body_grid, task_presets.grid("mpm2")),
    )


def test_task_timing_reads_current_preset_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preset functions consult their owning module instead of copied constants."""
    monkeypatch.setattr(task_presets, "HORIZON", 17)
    assert task_presets.env_kwargs("spring")["horizon"] == 17


def test_host_schema_and_presets_do_not_initialize_accelerators() -> None:
    """A fresh host process can read bounds, timing and terrain without JAX or Warp."""
    script = """
import json, sys
from envs.soft_walker.contracts import fields as field_contracts
from envs.soft_walker import config as task_presets
terrain_presets = task_presets
assert task_presets.grid('mpm2') == (8, 6)
assert terrain_presets.terrain('mpm2', 'stairs', 'float64')['centers'].shape == (4, 2)
print(json.dumps({name: name in sys.modules for name in ('jax', 'warp', 'newton')}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert json.loads(completed.stdout.splitlines()[-1]) == {"jax": False, "warp": False, "newton": False}
