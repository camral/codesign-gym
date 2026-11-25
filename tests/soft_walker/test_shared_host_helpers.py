"""Shared host logic preserves backend and standalone viewer boundaries."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from envs.soft_walker import host_topology
from envs.soft_walker.contracts import batching as batch_contracts
from envs.soft_walker.engine import topology_checks
from envs.soft_walker.engine.drivers import model_access as driver_models
from envs.soft_walker.engine.physics.terrain import Terrain
from envs.soft_walker import config as registered_task, embodiment
from envs.soft_walker import config as bundle_builders
from envs.soft_walker import config as task_presets
from envs.soft_walker.viewer.recording import terrain_height
from envs.soft_walker.viewer.terrain_geometry import height_at


def test_shared_host_imports_and_standalone_recording_need_no_accelerator() -> None:
    """Both package and historical standalone imports remain usable with only NumPy."""
    viewer = Path(__file__).resolve().parents[2] / "src/envs/soft_walker/viewer"
    script = f"""
import json, sys
from envs.soft_walker import host_terrain
from envs.soft_walker import embodiment
from envs.soft_walker.contracts import batching as batch_contracts
from envs.soft_walker import config as registered_task
sys.path.insert(0, {str(viewer)!r})
import recording
print(json.dumps({{name: name in sys.modules for name in ('jax', 'warp', 'newton')}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    assert json.loads(completed.stdout.splitlines()[-1]) == {"jax": False, "warp": False, "newton": False}


@pytest.mark.parametrize(
    ("occupied", "expected"),
    [
        (np.zeros((2, 2), bool), ["EMPTY_GEOMETRY"]),
        (np.ones((2, 2), bool), []),
        (np.array([[1, 0, 0, 1]], bool), ["DISCONNECTED_GEOMETRY"]),
        (np.eye(2, dtype=bool), ["DISCONNECTED_GEOMETRY", "NONMANIFOLD_JUNCTION"]),
        (np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], bool), ["NONMANIFOLD_JUNCTION"]),
    ],
)
def test_topology_errors_retain_connectivity_and_junction_vocabulary(
    occupied: np.ndarray, expected: list[str]
) -> None:
    """A globally connected design can still contain an illegal local corner junction."""
    assert host_topology.errors(occupied, occupied.shape) == expected
    assert embodiment.topology_errors(occupied) == expected
    assert topology_checks.errors(occupied, occupied.shape) == expected
    legacy = [
        "VERTEX_ONLY_JUNCTION" if item == "NONMANIFOLD_JUNCTION" else item for item in expected
    ]
    assert (
        host_topology.errors(occupied, occupied.shape, junction_error="VERTEX_ONLY_JUNCTION")
        == legacy
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_compiled_topology_matches_every_local_host_pattern(dim: int) -> None:
    """JAX retains the shared bit ordering for all 16 planar or 256 spatial patterns."""
    count = 2**dim
    occupied = ((np.arange(2**count)[:, None] >> np.arange(count)) & 1).reshape((-1,) + (2,) * dim)
    expected = np.asarray([not host_topology.errors(row, row.shape) for row in occupied])
    actual = jax.jit(jax.vmap(lambda row: topology_checks.valid(row, (2,) * dim)))(
        jnp.asarray(occupied)
    )
    np.testing.assert_array_equal(actual, expected)
    assert topology_checks.local_table is host_topology.local_table
    assert embodiment._local_table is host_topology.local_table
    assert embodiment._local_pattern is host_topology.local_pattern
    assert embodiment._connected is topology_checks._connected is host_topology.connected


def test_timing_metadata_preserves_default_target_and_overridden_hold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-dividing holds extend the episode while the default target stays fixed."""
    monkeypatch.setattr(task_presets, "HORIZON", 5)
    monkeypatch.setattr(task_presets, "SUBSTEPS", 4)
    monkeypatch.setattr(task_presets, "CONTROL_DT", 0.004)
    monkeypatch.setattr(task_presets, "SETTLE_SECONDS", 0.008)
    default, timing = registered_task.resolve_timing("spring", 6)
    metadata = registered_task.timing_metadata(timing, default, 0.001)
    assert metadata.physics_steps_per_action == 6
    assert metadata.horizon == 4
    assert metadata.control_dt == 0.006
    assert metadata.target_episode_duration_s == 0.020
    assert metadata.episode_duration_s == 0.024
    assert metadata.settling_duration_s == 0.008
    default, timing = registered_task.resolve_timing("spring", None)
    assert default is timing


def test_candidate_structural_gate_and_builder_rejection_are_shared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejected structure never builds; native validation failures retain BUILD messages."""
    calls = []

    def build(*args: object) -> object:
        calls.append(args)
        raise ValueError("invalid resolved mass")

    monkeypatch.setattr(bundle_builders, "build_bundle", build)
    options = dict(driver="spring", dim=2, dtype=np.dtype("float64"), terrain="flat")
    candidate = np.zeros(9)
    assert registered_task.candidate_bundle(candidate, errors=["EMPTY_GEOMETRY"], **options) == (
        None,
        ["EMPTY_GEOMETRY"],
    )
    assert not calls
    assert registered_task.candidate_bundle(candidate, errors=[], **options) == (
        None,
        ["BUILD: invalid resolved mass"],
    )
    assert len(calls) == 1
    assert calls[0][0] == "spring"


def test_render_selection_and_info_stacking_keep_registered_contract() -> None:
    """Missing info values stay lists; rendering selects only requested lanes."""
    calls = []

    def frame(index: int) -> np.ndarray:
        calls.append(index)
        return np.full((2, 2, 3), index, np.uint8)

    np.testing.assert_array_equal(
        batch_contracts.render_lanes(3, 1, frame), np.ones((2, 2, 3), np.uint8)
    )
    assert calls == [1]
    with pytest.raises(IndexError, match="env_idx"):
        batch_contracts.render_lanes(3, 3, frame)
    assert calls == [1]
    result = batch_contracts.stack_info([{"nested": {"status": 1}}, {"nested": {"status": 2}}])
    np.testing.assert_array_equal(result["nested"]["status"], [1, 2])
    assert batch_contracts.stack_info([{"value": 1}, {}]) == {"value": [1, None]}


def terrain_record(dim: int) -> dict[str, object]:
    """A half-metre plane, a rounded box and an inactive higher box."""
    return dict(
        normal=[0.0, 1.0] + ([0.0] if dim == 3 else []),
        offset=0.5,
        centers=[
            [0.0, 0.5] + ([0.0] if dim == 3 else []),
            [0.0, 5.0] + ([0.0] if dim == 3 else []),
        ],
        half_sizes=[[1.0, 0.5] + ([1.0] if dim == 3 else [])] * 2,
        rounding=[0.25, 0.0],
        active=[True, False],
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_viewer_and_scenario_height_adapters_preserve_rounded_geometry(dim: int) -> None:
    """Caps, outside samples, inactive obstacles and spatial corners have known heights."""
    saved = terrain_record(dim)
    terrain = Terrain(*(jnp.asarray(saved[key]) for key in Terrain._fields))
    x, z = np.array([0.0, 1.0, 1.01, 1.0]), np.array([0.0, 0.0, 0.0, 1.0])
    expected = [1.0, 0.75, 0.5, 0.5 if dim == 3 else 0.75]
    np.testing.assert_allclose(terrain_height(x, z, saved), expected)
    np.testing.assert_allclose(height_at(x, z, terrain), expected)


def test_height_adapters_keep_none_vertical_and_broadcast_policies() -> None:
    """Viewer defaults keep x's shape; scenarios broadcast and retain vertical division."""
    x, z = np.zeros((2, 1), np.float32), np.zeros(3, np.float32)
    absent = terrain_height(x, z, None)
    assert absent.shape == x.shape and absent.dtype == x.dtype
    saved = terrain_record(2)
    terrain = Terrain(*(jnp.asarray(saved[key]) for key in Terrain._fields))
    assert terrain_height(x, z, saved).shape == (2, 1)
    assert height_at(x, z, terrain).shape == (2, 3)
    saved = dict(normal=[1.0, 0.0], offset=1.0)
    with pytest.raises(ValueError, match="not a height field"):
        terrain_height(x, z, saved)
    vertical = terrain._replace(normal=jnp.array([1.0, 0.0]), offset=jnp.asarray(1.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        assert np.isinf(height_at(0.0, 0.0, vertical))


def test_resolved_and_driver_geometry_accessors_have_one_owner() -> None:
    """Historical imports share helpers without merging their observation schemas."""
    from envs.soft_walker.engine.drivers import environment as drivers

    assert drivers.reference is driver_models.reference
    assert drivers.positions is driver_models.positions
    model = SimpleNamespace(cloud=SimpleNamespace(reference=jnp.array([[1.0, 2.0]])))
    state = SimpleNamespace(x=jnp.array([[3.0, 4.0]]))
    np.testing.assert_array_equal(driver_models.reference(model, "mpm2"), [[1.0, 2.0]])
    np.testing.assert_array_equal(driver_models.positions(model, state, "mpm2"), [[3.0, 4.0]])
