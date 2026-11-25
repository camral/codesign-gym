"""Recording/display extraction preserves saved samples and optional imports."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from envs.soft_walker.viewer import recording, view_newton


def saved_recording(directory: Path, *, driver: str = "mpm2") -> Path:
    """Write a tiny planar recording with physical timestamps and no solver failures."""
    reference = np.array([[0.0, 0.0], [0.0, 0.02], [0.02, 0.0], [0.02, 0.02]])
    positions = np.stack([reference + [0.01 * frame, 0.0] for frame in range(3)])
    metrics = np.zeros((3, 8))
    metrics[:, 0] = [0.0, 0.02, 0.04]
    metrics[:, 1] = [0.0, 0.01, 0.02]
    directory.mkdir()
    np.savez_compressed(
        directory / "trajectory.npz",
        positions=positions,
        metrics=metrics,
        reference=reference,
        active=np.ones(4, dtype=bool),
    )
    (directory / "protocol.json").write_text(json.dumps({"driver": driver}), encoding="utf-8")
    return directory


def test_compatibility_imports_and_coordinate_handedness() -> None:
    for name in ("Recording", "Skin", "DetailGeometry", "world_coordinates", "terrain_height"):
        assert getattr(view_newton, name) is getattr(recording, name)
    np.testing.assert_array_equal(recording.world_coordinates([[2.0, 3.0]]), [[2.0, 0.0, 3.0]])
    basis = recording.world_coordinates(np.eye(3))
    np.testing.assert_array_equal(basis[2], np.cross(basis[0], basis[1]))


@pytest.mark.parametrize("failure", ["solver", "time", "positions"])
def test_only_valid_saved_prefix_is_rendered(tmp_path: Path, failure: str) -> None:
    directory = saved_recording(tmp_path / "run")
    path = directory / "trajectory.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if failure == "solver":
        arrays["metrics"][2, 6] = 1
    elif failure == "time":
        arrays["metrics"][2, 0] = arrays["metrics"][1, 0]
    else:
        arrays["positions"][2, 0, 0] = np.nan
    np.savez_compressed(path, **arrays)
    before = path.read_bytes()
    rec = recording.Recording(directory)
    assert rec.source_count == 3
    assert len(rec.times) == 2
    assert rec.excluded_reason
    assert rec.verdict() == "Failure preview: valid prefix"
    np.testing.assert_array_equal(rec.positions, arrays["positions"][:2])
    assert path.read_bytes() == before


def test_mpm_particles_and_display_skin_remain_distinct(tmp_path: Path) -> None:
    rec = recording.Recording(saved_recording(tmp_path / "run"))
    before = rec.positions.copy()
    detail = recording.DetailGeometry(rec)
    skin = recording.Skin(rec, 0.004)
    provenance = detail.provenance()
    assert detail.particles and detail.available
    assert provenance["particle_count"] == 4
    assert provenance["unique_edge_count"] == 0
    assert provenance["marker_radius_is_cosmetic"] is True
    assert "not solver elements" in skin.description
    assert skin.geometry(1)
    np.testing.assert_array_equal(rec.positions, before)
    np.testing.assert_array_equal(rec.frame_indices(50), [0, 1, 2])
    np.testing.assert_array_equal(rec.frame_indices(50, maximum=2), [0, 2])


def test_element_details_use_saved_edges_only(tmp_path: Path) -> None:
    directory = saved_recording(tmp_path / "run", driver="spring")
    np.savez_compressed(directory / "topology.npz", edges=np.array([[0, 1], [1, 0], [2, 3]]))
    rec = recording.Recording(directory)
    detail = recording.DetailGeometry(rec)
    assert not detail.particles
    np.testing.assert_array_equal(detail.edges, [[0, 1], [2, 3]])
    # The cosmetic skin can close the lattice; detail never infers the missing links.
    assert len(recording.lattice_cells(rec.reference, rec.active)) == 1
    assert detail.provenance()["unique_edge_count"] == 2


def test_declared_nonflat_terrain_requires_saved_geometry(tmp_path: Path) -> None:
    directory = saved_recording(tmp_path / "run")
    (directory / "protocol.json").write_text('{"terrain": "stairs"}', encoding="utf-8")
    with pytest.raises(ValueError, match="terrain.json is missing"):
        recording.Recording(directory)


def test_viewer_imports_do_not_load_simulators_or_graphics() -> None:
    viewer = Path(recording.__file__).parent
    command = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "import view_newton, terrain_editor, render_morphologies; "
        "assert not {'jax', 'warp', 'newton'} & sys.modules.keys()"
    )
    result = subprocess.run(
        [sys.executable, "-c", command, str(viewer)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
