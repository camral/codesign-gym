"""Viewer reads declared metric names and retains metadata-free recording support."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from envs.soft_walker.viewer.recording import Recording


def write_recording(directory: Path, metrics: np.ndarray) -> None:
    """Write a tiny fixed-geometry trajectory with supplied metric rows."""
    reference = np.array([[0.0, 0.0], [1.0, 0.0]])
    np.savez_compressed(
        directory / "trajectory.npz",
        reference=reference,
        positions=np.repeat(reference[None], len(metrics), axis=0),
        active=np.ones(2, dtype=bool),
        metrics=metrics,
    )


@pytest.mark.parametrize("metadata", ("scenario", "locomotion"))
def test_saved_column_names_control_time_displacement_and_status(
    tmp_path: Path, metadata: str
) -> None:
    """Reordering declared columns cannot swap displacement for time or hide a failed row."""
    metrics = np.array([[0.0, 0.0, 0.0], [0.2, 0.0, 0.1], [0.5, 1.0, 0.2]])
    write_recording(tmp_path, metrics)
    columns = ["dx_m" if metadata == "scenario" else "distance_x_m", "status", "time_s"]
    path = tmp_path / ("metric_columns.json" if metadata == "scenario" else "replay.json")
    path.write_text(json.dumps(columns if metadata == "scenario" else {"columns": columns}))
    record = Recording(tmp_path)
    np.testing.assert_array_equal(record.times, [0.0, 0.1])
    np.testing.assert_array_equal(record.displacements_x, [0.0, 0.2])
    assert record.excluded_reason == "nonzero saved solver status"
    np.testing.assert_array_equal(record.metrics, metrics[:2])


def test_legacy_recording_without_metadata_retains_positional_fallback(tmp_path: Path) -> None:
    metrics = np.zeros((3, 8))
    metrics[:, 0] = [0.0, 0.1, 0.2]
    metrics[:, 1] = [0.0, 0.2, 0.4]
    metrics[2, 6] = 1
    write_recording(tmp_path, metrics)
    record = Recording(tmp_path)
    np.testing.assert_array_equal(record.times, [0.0, 0.1])
    np.testing.assert_array_equal(record.displacements_x, [0.0, 0.2])
    assert record.excluded_reason == "nonzero saved solver status"


@pytest.mark.parametrize(
    "columns",
    (
        None,
        {"time_s": 0},
        ["time_s", "time_s", "dx_m"],
        ["time_s", "dx_m"],
        ["time_s", "dx_m", "distance_x_m"],
        ["unknown", "dx_m", "status"],
    ),
)
def test_declared_malformed_metric_columns_fail_clearly(tmp_path: Path, columns: object) -> None:
    write_recording(tmp_path, np.zeros((2, 3)))
    (tmp_path / "metric_columns.json").write_text(json.dumps(columns))
    with pytest.raises(ValueError, match="Metric"):
        Recording(tmp_path)


def test_scenario_sidecar_has_precedence_over_replay_metadata(tmp_path: Path) -> None:
    write_recording(tmp_path, np.array([[0.0, 0.0], [0.2, 0.1]]))
    (tmp_path / "metric_columns.json").write_text('["dx_m", "time_s"]')
    (tmp_path / "replay.json").write_text('{"columns": ["time_s", "distance_x_m"]}')
    record = Recording(tmp_path)
    np.testing.assert_array_equal(record.times, [0.0, 0.1])
    np.testing.assert_array_equal(record.displacements_x, [0.0, 0.2])


def test_legacy_time_only_recording_still_loads_without_a_status_column(tmp_path: Path) -> None:
    write_recording(tmp_path, np.array([[0.0], [0.1]]))
    record = Recording(tmp_path)
    np.testing.assert_array_equal(record.times, [0.0, 0.1])
    assert record.excluded_reason is None
    with pytest.raises(ValueError, match="horizontal displacement"):
        _ = record.displacements_x
