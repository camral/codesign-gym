"""Evidence gates and installed-command errors use data and explicit checkout contracts."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from envs.soft_walker import shared as _paths
from envs.soft_walker.diagnostics import compare_performance_audits, compare_warp_followup


def _followup(
    root: Path,
    fixtures: dict[str, dict[str, Any]],
    **arguments: Any,
) -> None:
    """Write small complete, checksummed collections without executing physics."""
    root.mkdir()
    selection = {
        "dtypes": ["float64"],
        "physics": ["spring"],
        "skip_solids": False,
        "skip_mpm": False,
        "skip_spring": False,
        "offline_spaces": False,
    }
    selection.update(arguments)
    manifest = {"source_sha256": {"fixture": "same-source"}, "arguments": selection}
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "summary.json").write_text(
        json.dumps({"success": True, "complete": True, "source_unchanged": True})
    )
    for name, arrays in fixtures.items():
        folder = root / name / "numerics"
        folder.mkdir(parents=True)
        np.savez(folder / "parity.npz", **arrays)
    hashes = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    (root / "file_hashes.json").write_text(json.dumps(hashes))


def _followup_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Exercise the public comparator and its unchanged return/report schema."""
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_warp_followup",
            str(tmp_path / "cpu"),
            str(tmp_path / "cuda"),
            "--out",
            str(output),
        ],
    )
    code = compare_warp_followup.main()
    result: dict[str, Any] = json.loads(output.read_text())
    assert code == (0 if result["passed"] else 1)
    assert set(result) <= {"passed", "comparisons", "error"}
    return result


@pytest.mark.parametrize(
    "cpu,cuda,passed",
    [
        (["A", "B"], ["A"], False),
        (["A"], ["A", "B"], False),
        ([], [], False),
        (["A"], [], False),
        (["A", "B"], ["A", "B"], True),
    ],
)
def test_followup_requires_symmetric_nonempty_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu: list[str], cuda: list[str], passed: bool
) -> None:
    """Either side omitting a fixture fails before any numerical comparisons."""
    for side, names in (("cpu", cpu), ("cuda", cuda)):
        _followup(tmp_path / side, {name: {"values": np.array([1.0])} for name in names})
    result = _followup_report(tmp_path, monkeypatch)
    assert result["passed"] is passed
    if not passed:
        assert result["comparisons"] == {}
        assert "fixture selections" in result["error"]


@pytest.mark.parametrize(
    "selection",
    [
        {"dtypes": ["float32"]},
        {"physics": ["beam"]},
        {"skip_solids": True},
        {"skip_mpm": True},
        {"skip_spring": True},
        {"offline_spaces": True},
    ],
)
def test_followup_rejects_different_numerical_protocols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: dict[str, Any]
) -> None:
    """Equal saved values cannot hide a differently selected numerical collection."""
    fixtures = {"A": {"values": np.array([1.0])}}
    _followup(tmp_path / "cpu", fixtures)
    _followup(tmp_path / "cuda", fixtures, **selection)
    report = _followup_report(tmp_path, monkeypatch)
    assert not report["passed"] and "protocol selections" in report["error"]


def test_followup_allows_collection_only_differences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Devices, watchdogs, repetitions, unit tests and performance audits may differ."""
    fixtures = {"A": {"values": np.array([1.0])}}
    _followup(
        tmp_path / "cpu",
        fixtures,
        device="cpu",
        out="left",
        timeout=1,
        repeats=1,
        skip_unit=False,
        audit=False,
        dtypes=["float64", "float32"],
        physics=["beam", "spring"],
    )
    _followup(
        tmp_path / "cuda",
        fixtures,
        device="cuda",
        out="right",
        timeout=2,
        repeats=3,
        skip_unit=True,
        audit=True,
        dtypes=["float32", "float64"],
        physics=["spring", "beam"],
    )
    assert _followup_report(tmp_path, monkeypatch)["passed"]


def test_followup_ignores_an_unselected_solid_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The family option does not select fixtures when solid collection is disabled."""
    fixtures = {"mpm": {"values": np.array([1.0])}}
    _followup(tmp_path / "cpu", fixtures, skip_solids=True, physics=["spring"])
    _followup(tmp_path / "cuda", fixtures, skip_solids=True, physics=["beam"])
    assert _followup_report(tmp_path, monkeypatch)["passed"]


@pytest.mark.parametrize(
    "dtype,directory,base,delta,passed",
    [
        ("float32", "float64", 1.0, 0.001, True),
        ("float64", "float32", 1.0, 0.001, False),
        ("float32", "renamed", 0.0, 2e-7, True),
        ("float32", "renamed", 0.0, 4e-7, False),
        ("float64", "float32", 0.0, 1e-11, True),
        ("float64", "float32", 0.0, 3e-11, False),
    ],
)
def test_followup_precision_follows_arrays_and_preserves_absolute_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dtype: str,
    directory: str,
    base: float,
    delta: float,
    passed: bool,
) -> None:
    """Renaming directories cannot select looser precision acceptance."""
    _followup(tmp_path / "cpu", {directory: {"values": np.array([base], dtype=dtype)}})
    _followup(tmp_path / "cuda", {directory: {"values": np.array([base + delta], dtype=dtype)}})
    report = _followup_report(tmp_path, monkeypatch)
    assert report["passed"] is passed
    assert set(next(iter(report["comparisons"].values()))) == {"passed", "relative_l2"}


def test_followup_rejects_paired_dtype_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal numerical values in different precisions are different protocols."""
    _followup(tmp_path / "cpu", {"A": {"values": np.array([1.0], dtype="float32")}})
    _followup(tmp_path / "cuda", {"A": {"values": np.array([1.0], dtype="float64")}})
    result = _followup_report(tmp_path, monkeypatch)
    assert not result["passed"] and "Shape/dtype mismatch" in result["error"]


def test_followup_accepts_mixed_reference_and_status_dtypes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Float32 stages may contain float64 reference/loss values and integer statuses."""
    arrays = {
        "gradient": np.array([1.0], dtype="float32"),
        "loss": np.array([2.0]),
        "state_status": np.array([0], dtype="int32"),
    }
    for side in ("cpu", "cuda"):
        _followup(tmp_path / side, {"float32": arrays}, dtypes=["float32"])
    assert _followup_report(tmp_path, monkeypatch)["passed"]


def _performance(root: Path, array: Any, directory: str) -> None:
    """Write the existing focused-audit schema and checksums for one reference array."""
    folder = root / directory
    folder.mkdir(parents=True)
    np.savez(folder / "references.npz", values=array)
    manifest = {
        "schema": "codesign_performance_audit_v1",
        "passed": True,
        "completed": True,
        "source_hashes": {"fixture": "same-source"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    checksums = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}"
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    (root / "SHA256SUMS").write_text("\n".join(checksums) + "\n")


@pytest.mark.parametrize(
    "dtype,directory,delta,passed,limits",
    [
        ("float32", "float64", 0.001, True, (2e-3, 3e-6)),
        ("float64", "float32", 0.001, False, (1e-6, 1e-10)),
        ("float32", "renamed", 0.003, False, (2e-3, 3e-6)),
        ("float64", "renamed", 5e-7, True, (1e-6, 1e-10)),
    ],
)
def test_performance_precision_follows_stored_dtype(
    tmp_path: Path,
    dtype: str,
    directory: str,
    delta: float,
    passed: bool,
    limits: tuple[float, float],
) -> None:
    """Existing elementwise limits are retained and reported for the actual precision."""
    _performance(tmp_path / "left", np.array([1.0], dtype=dtype), directory)
    _performance(tmp_path / "right", np.array([1.0 + delta], dtype=dtype), directory)
    report = compare_performance_audits.compare(tmp_path / "left", tmp_path / "right")
    assert report["passed"] is passed
    assert set(report) == {"passed", "failures", "comparisons", "scope"}
    row = report["comparisons"][0]
    assert (row["rtol"], row["atol"]) == limits
    assert set(row) == {"stage", "array", "passed", "relative_l2", "max_absolute", "rtol", "atol"}


@pytest.mark.parametrize("right_dtype", ["float64", "int32"])
def test_performance_rejects_equal_values_with_different_dtypes(
    tmp_path: Path, right_dtype: str
) -> None:
    """A same-valued array does not excuse a changed storage/precision contract."""
    _performance(tmp_path / "left", np.array([1.0], dtype="float32"), "float32")
    _performance(tmp_path / "right", np.array([1.0], dtype=right_dtype), "float32")
    report = compare_performance_audits.compare(tmp_path / "left", tmp_path / "right")
    assert not report["passed"]
    assert report["comparisons"][0]["relative_l2"] is None
    assert report["comparisons"][0]["max_absolute"] is None


@pytest.mark.parametrize("missing", [None, "pyproject.toml", "package", "tests"])
def test_checkout_guard_checks_source_and_test_layout(tmp_path: Path, missing: str | None) -> None:
    """A package import alone does not establish the source/test tree needed by runners."""
    if missing != "pyproject.toml":
        (tmp_path / "pyproject.toml").touch()
    if missing != "package":
        package = tmp_path / "src/envs/soft_walker"
        package.mkdir(parents=True)
        (package / "__init__.py").touch()
    if missing != "tests":
        (tmp_path / "tests/soft_walker").mkdir(parents=True)
    if missing is None:
        _paths.require_source_checkout(tmp_path)
    else:
        with pytest.raises(ValueError, match="source checkout"):
            _paths.require_source_checkout(tmp_path)


CHECKOUT_COMMANDS = [
    ("run_locomotion_tests", []),
    ("run_scenario_tests", []),
    ("run_resident_tests", ["--device", "cpu"]),
    ("run_krylov_audit", ["--device", "cpu"]),
    ("run_validation", ["--device", "cpu"]),
    ("run_performance_audit", ["--device", "cpu"]),
    ("run_warp_followup", []),
    ("run_warp_native_tests", []),
    ("run_warp_ports_tests", []),
    ("run_warp_speed_audit", []),
    ("resident_worker", ["--driver", "spring"]),
    ("worker", ["--stage", "unit", "--device", "cpu", "--dtype", "float64"]),
    ("performance_worker", ["--unit", "--device", "cpu", "--dtype", "float64"]),
]


@pytest.mark.parametrize("name,arguments", CHECKOUT_COMMANDS)
def test_checkout_commands_fail_before_creating_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    arguments: list[str],
) -> None:
    """Wheel-only layouts fail clearly before work or partial output is launched."""
    module = importlib.import_module(f"envs.soft_walker.diagnostics.{name}")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path / "installed")
    if hasattr(module, "ROOT"):
        monkeypatch.setattr(module, "ROOT", tmp_path / "installed")
    output = tmp_path / "evidence"
    monkeypatch.setattr(sys, "argv", [name, *arguments, "--out", str(output)])
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 2
    assert "requires a codesign-gym source checkout" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("name,arguments", CHECKOUT_COMMANDS)
def test_checkout_command_help_remains_available_without_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
    arguments: list[str],
) -> None:
    """Help is informational and exits before checkout or optional-backend work."""
    module = importlib.import_module(f"envs.soft_walker.diagnostics.{name}")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path / "installed")
    if hasattr(module, "ROOT"):
        monkeypatch.setattr(module, "ROOT", tmp_path / "installed")
    monkeypatch.setattr(sys, "argv", [name, "--help"])
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert caught.value.code == 0
    assert "usage:" in capsys.readouterr().out
