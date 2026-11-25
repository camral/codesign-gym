"""Ownership and persisted-schema contracts shared by native host code."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("warp")

from envs.soft_walker.engine.warp_native.episode import Episode
from envs.soft_walker.engine.warp_native.host_types import FieldArrays
from envs.soft_walker.engine.warp_native.model import Bundle, DeviceModel, uniform_body
from envs.soft_walker.engine.warp_native.solid_builders import regular_solid
from envs.soft_walker.engine.warp_native.solid_episode import SolidEpisode
from envs.soft_walker.engine.warp_native.solid_model import SolidBundle, SolidDeviceModel

PRECISION = os.environ.get("WARP_TEST_DTYPE", "float64")
DEVICE = os.environ.get("WARP_TEST_DEVICE", "cpu")


def _bundle(kind: str) -> Bundle | SolidBundle:
    if kind == "mpm":
        return uniform_body(2, precision=PRECISION)
    return regular_solid("spring", shape=(1, 1), precision=PRECISION)


def _terrain() -> FieldArrays:
    return {
        "normal": np.array([0, 1], dtype=PRECISION),
        "offset": np.array(0.015, dtype=PRECISION),
        "centers": np.array([[0.02, -0.01]], dtype=PRECISION),
        "half_sizes": np.array([[0.04, 0.01]], dtype=PRECISION),
        "rounding": np.array([0.002], dtype=PRECISION),
        "active": np.array([1], dtype=np.int32),
    }


@pytest.mark.parametrize("kind", ["mpm", "spring"])
def test_archives_keep_schema_state_terrain_and_independent_arrays(
    tmp_path: Path, kind: str
) -> None:
    """Archive transport must retain every group and each family's schema tag."""
    bundle = _bundle(kind)
    bundle.terrain = _terrain()
    if isinstance(bundle, Bundle):
        owner = DeviceModel(bundle, DEVICE)
        bundle.initial = owner.download(owner.initial)
        schema: int | str = 1
    else:
        solid_owner = SolidDeviceModel(bundle, DEVICE)
        bundle.initial = solid_owner.numpy_state(solid_owner.initial)
        schema = "warp-solid-v1"
    bundle.initial["actuator"].fill(0.125)
    bundle.initial["ledger"][:] = [0.01, 0.02, 0.03, 0.04]
    path = tmp_path / kind / "model.npz"
    bundle.save(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        assert metadata["schema"] == schema
        assert set(archive.files) == {
            "metadata",
            *(f"model_{key}" for key in bundle.fields),
            *(f"terrain_{key}" for key in bundle.terrain),
            *(f"state_{key}" for key in bundle.initial),
        }
    loaded = type(bundle).load(path)
    assert loaded.config == bundle.config
    assert isinstance(loaded.config.grid_shape, tuple)
    assert isinstance(loaded.config.origin, tuple)
    if isinstance(bundle, SolidBundle):
        assert isinstance(loaded, SolidBundle)
        assert (loaded.h, loaded.area_ref) == (bundle.h, bundle.area_ref)
    for original, restored in (
        (bundle.fields, loaded.fields),
        (bundle.terrain, loaded.terrain),
        (bundle.initial, loaded.initial),
    ):
        assert original is not None and restored is not None
        for key, value in original.items():
            np.testing.assert_array_equal(restored[key], value)
            assert restored[key].dtype == value.dtype
            assert not np.shares_memory(restored[key], value)
    wrong_loader = SolidBundle.load if kind == "mpm" else Bundle.load
    message = (
        "Not a native Lagrangian bundle" if kind == "mpm" else "Unsupported model-bundle version"
    )
    with pytest.raises(ValueError, match=message):
        wrong_loader(path)


@pytest.mark.parametrize(
    "kind,message",
    [
        ("mpm", "Terrain plane normal must have unit length"),
        ("spring", "Invalid terrain box centers"),
    ],
)
def test_terrain_validation_preserves_family_error_precedence(kind: str, message: str) -> None:
    bundle = _bundle(kind)
    bundle.terrain = _terrain()
    bundle.terrain["normal"] = np.ones(3, dtype=PRECISION)
    bundle.terrain["centers"] = np.ones((1, 3), dtype=PRECISION)
    with pytest.raises(ValueError, match=message):
        bundle.validate()


@pytest.mark.parametrize("kind", ["mpm", "spring"])
def test_terrain_uploads_keep_independent_device_storage(kind: str) -> None:
    bundle = _bundle(kind)
    bundle.terrain = _terrain()
    if isinstance(bundle, Bundle):
        first = DeviceModel(bundle, DEVICE).model
        second = DeviceModel(bundle, DEVICE).model
    else:
        first = SolidDeviceModel(bundle, DEVICE).model.terrain
        second = SolidDeviceModel(bundle, DEVICE).model.terrain
    assert first.box_centers.ptr != second.box_centers.ptr
    np.testing.assert_array_equal(first.box_centers.numpy(), bundle.terrain["centers"])
    np.testing.assert_array_equal(first.box_active.numpy(), bundle.terrain["active"])
    assert np.asarray(first.plane_normal).dtype == np.dtype(PRECISION)
    np.testing.assert_allclose(first.plane_offset, bundle.terrain["offset"])
    second.box_centers.zero_()
    bundle.terrain["centers"].fill(2)
    np.testing.assert_array_equal(first.box_centers.numpy(), _terrain()["centers"])


@pytest.mark.parametrize("kind", ["mpm", "spring"])
def test_episode_reuse_preserves_ownership_and_family_gradient_schema(kind: str) -> None:
    """Repeated adjoints must retain private buffers and the correct field exclusions."""
    bundle = _bundle(kind)
    bundle.config = replace(bundle.config, gravity=0, contact_enabled=False)
    commands = np.zeros((2, bundle.channels), dtype=PRECISION)
    if isinstance(bundle, Bundle):
        episode = Episode(DeviceModel(bundle, DEVICE), commands, checkpoint_stride=1)
        other = Episode(DeviceModel(bundle, DEVICE), commands, checkpoint_stride=1)
        constant_fields = {"kind", "channel"}
        assert episode.reuse_replay_roots is False
    else:
        episode = SolidEpisode(SolidDeviceModel(bundle, DEVICE), commands, checkpoint_stride=1)
        other = SolidEpisode(SolidDeviceModel(bundle, DEVICE), commands, checkpoint_stride=1)
        constant_fields = {"kind", "channel", "entities", "cells", "free"}
        assert episode.reuse_replay_roots is True
    assert episode.commands.ptr != other.commands.ptr
    assert episode.graphs is not other.graphs
    assert episode.compile_seconds is not other.compile_seconds
    pointers = episode.commands.ptr, episode.final.v.ptr, episode.owner.initial.v.ptr
    first = episode.run(gradient=True)
    assert first["status"] == first["adjoint_status"] == 0
    gradients = episode.gradients()
    assert set(gradients["model"]) == set(bundle.fields) - constant_fields
    saved_graphs = dict(episode.graphs)
    episode.set_commands(commands.copy())
    with pytest.raises(RuntimeError, match="Run a whole value-and-gradient evaluation first"):
        episode.gradients()
    second = episode.run(gradient=True)
    assert second["status"] == second["adjoint_status"] == 0
    assert second["loss"] == first["loss"]
    assert episode.graphs == saved_graphs
    assert pointers == (episode.commands.ptr, episode.final.v.ptr, episode.owner.initial.v.ptr)
    for field, values in gradients["model"].items():
        np.testing.assert_array_equal(episode.gradients()["model"][field], values)
    assert other.completed is False and other.graphs == {}
