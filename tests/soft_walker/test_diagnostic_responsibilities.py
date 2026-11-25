"""Relocated diagnostics retain event schemas and independent runtime imports."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from soft_paths import REPO_ROOT

from envs.soft_walker import shared as evidence, shared as gradient_checks


def test_explicit_gradient_ladder_retains_raw_direction_and_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The extracted ladder keeps all samples and emits its established event."""
    monkeypatch.setenv("CODESIGN_EVIDENCE_DIR", str(tmp_path))
    theta, direction = np.array([2.0]), np.array([3.0])
    report = gradient_checks.finite_difference_ladder(
        lambda value: value[0] ** 3,
        theta,
        direction,
        steps=(0.1, 0.05, 0.025),
        autodiff=36.0,
        name="cubic_direction",
    )
    np.testing.assert_array_equal(report["direction"], direction)
    assert [row["h"] for row in report["rows"]] == [0.1, 0.05, 0.025]
    for row in report["rows"]:
        assert row["central"] == pytest.approx(36.0 + 27.0 * row["h"] ** 2)
    assert report["rows"][0]["observed_order"] is None
    assert report["rows"][1]["observed_order"] == pytest.approx(2.0)
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert len(events) == 1
    event = events[0]
    assert event.pop("event") == "central_difference_ladder"
    assert event.pop("utc")
    assert event.pop("monotonic_s") > 0
    assert event == evidence.evidence_jsonable(report)


def test_evidence_writer_preserves_nonfinite_markers_and_parent_creation(tmp_path: Path) -> None:
    """Moving the evidence owner does not adopt checkpoint-writer serialization."""
    path = tmp_path / "nested" / "result.json"
    evidence.evidence_write_json(path, {"value": np.float64(np.inf), "missing": float("nan")})
    assert json.loads(path.read_text()) == {
        "value": {"nonfinite": "inf"},
        "missing": {"nonfinite": "nan"},
    }
    assert not path.with_suffix(".json.tmp").exists()


@pytest.mark.parametrize("name", ("mpm_conversion", "solid_conversion"))
def test_production_converters_do_not_import_jax_or_diagnostic_fixtures(
    tmp_path: Path, name: str
) -> None:
    """Resolved host-data conversion remains independent of diagnostic references."""
    code = (
        "import importlib,json,sys; "
        f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r}); "
        f"importlib.import_module('envs.soft_walker.engine.warp_native.{name}'); "
        "print(json.dumps({'jax': 'jax' in sys.modules, "
        "'diagnostics': [n for n in sys.modules if n.startswith('envs.soft_walker.diagnostics')]}))"
    )
    output = subprocess.check_output([sys.executable, "-c", code], cwd=tmp_path, text=True)
    assert json.loads(output) == {"jax": False, "diagnostics": []}
