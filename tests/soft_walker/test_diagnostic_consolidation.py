"""Diagnostic extraction preserves acceptance gates, failure evidence, and process policies."""

from __future__ import annotations

import hashlib
import importlib
import json
import signal
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from envs.soft_walker.diagnostics import (
    _evidence,
    _focused_tests,
    _processes,
    compare_m2_precision,
    compare_m3_precision,
    run_resident_tests,
)


@pytest.mark.parametrize("phase", ["m2", "m3"])
@pytest.mark.parametrize("failure", [None, "ladder", "summary", "keys", "shape", "nonfinite"])
def test_precision_acceptance_and_report_names(
    tmp_path: Path, phase: str, failure: str | None
) -> None:
    """Forward agreement cannot replace successful numerics and independent order evidence."""
    for dtype in ("float64", "float32"):
        folder = tmp_path / dtype / f"{phase}_numerics"
        folder.mkdir(parents=True)
        passed = not (failure == "summary" and dtype == "float32")
        (folder / "summary.json").write_text(json.dumps({"passed": passed}))
        ladder = {
            "name": "control",
            "passed": True,
            "second_order_window": failure != "ladder",
            "cross_precision_required": dtype == "float32",
        }
        (folder / "gradient_ladders.json").write_text(json.dumps({"reports": [ladder]}))
        array = np.array([0.2, 0.4], dtype=dtype)
        if dtype == "float32":
            if failure == "shape":
                array = array[:1]
            elif failure == "nonfinite":
                array[0] = np.nan
        key = "other" if failure == "keys" and dtype == "float32" else "control_gradient"
        np.savez(folder / "reference_arrays.npz", **{key: array})
    compare = compare_m2_precision.compare if phase == "m2" else compare_m3_precision.compare
    report = compare(tmp_path)
    assert report["passed"] is (failure is None)
    assert (tmp_path / f"{phase}_precision_comparison.json").is_file()
    assert set(report) == {"passed", "failures", "results", "scope"}
    if failure == "ladder":
        assert report["failures"] == ["control:missing_float64_order_evidence"]
    elif failure == "summary":
        assert report["failures"] == ["float32:numerics_not_passed"]
    elif failure == "keys":
        assert report["failures"] == ["array_key_mismatch"]
    if failure != "keys":
        row = report["results"][0]
        assert row["rtol"] == 2e-3 and row["atol"] == 2e-6
        assert row["passed"] is (failure not in {"shape", "nonfinite"})


@pytest.mark.parametrize("name", ["locomotion", "scenario"])
@pytest.mark.parametrize("outcome", ["success", "timeout", "launch_error", "interrupt"])
def test_canonical_focused_runners_keep_hooks_and_failure_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, outcome: str
) -> None:
    """Both public commands retain their writer injection, stage environments, and ZIP-on-failure."""
    root = Path(__file__).resolve().parents[2]
    script = root / "src/envs/soft_walker/diagnostics" / f"run_{name}_tests.py"
    module = importlib.import_module(f"envs.soft_walker.diagnostics.run_{name}_tests")
    assert module.main.__globals__ is vars(module)
    output = tmp_path / "evidence"
    (tmp_path / "pyproject.toml").touch()
    package = tmp_path / "src/envs/soft_walker"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (tmp_path / "tests/soft_walker").mkdir(parents=True)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "SRC_ROOT", tmp_path / "src")
    monkeypatch.setattr(sys, "argv", [str(script), "--out", str(output), "--dtypes", "float32"])
    monkeypatch.delenv("CODESIGN_SCENARIO_EVIDENCE", raising=False)
    monkeypatch.setenv("CODESIGN_OFFLINE_SPACES", "1")
    written: list[str] = []
    original_write = module.write_json

    def write(path: Path, value: Any) -> None:
        written.append(path.name)
        original_write(path, value)

    monkeypatch.setattr(module, "write_json", write)

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        env = kwargs["env"]
        assert env["JAX_PLATFORMS"] == "cpu"
        assert env["JAX_ENABLE_X64"] == "0" and env["CODESIGN_TEST_DTYPE"] == "float32"
        assert "CODESIGN_OFFLINE_SPACES" not in env
        if name == "scenario":
            assert env["CODESIGN_SCENARIO_EVIDENCE"] == str(output / "float32/ladders")
            assert len([arg for arg in command if arg.startswith("tests/")]) == 4
        else:
            assert "CODESIGN_SCENARIO_EVIDENCE" not in env
            assert "tests/soft_walker/test_locomotion.py" in command
        kwargs["stdout"].write("stage output\n")
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if outcome == "launch_error":
            raise OSError("cannot launch")
        if outcome == "interrupt":
            raise KeyboardInterrupt
        junit = Path(next(arg.split("=", 1)[1] for arg in command if arg.startswith("--junit")))
        junit.write_text('<testsuites><testsuite tests="1" failures="0" /></testsuites>')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(_focused_tests.subprocess, "run", run)
    if outcome == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            module.main()
    else:
        assert module.main() == (0 if outcome == "success" else 1)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema"] == f"{name}_focused_tests_v1"
    assert manifest["passed"] is (outcome == "success")
    assert manifest["completed"] is (outcome != "interrupt")
    assert written.count("manifest.json") >= 2 and "command.json" in written
    if outcome != "interrupt":
        stage = manifest["stages"][0]
        assert stage["returncode"] == {"success": 0, "timeout": 124, "launch_error": 1}[outcome]
        assert stage["timed_out"] is (outcome == "timeout")
        if outcome == "success":
            assert stage["junit"] == [{"tests": "1", "failures": "0"}]
    with zipfile.ZipFile(output.with_name("evidence.zip")) as archive:
        assert archive.testzip() is None
        assert "evidence/SHA256SUMS" in archive.namelist()
        assert archive.read("evidence/float32/stdout.log") == b"stage output\n"
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split("  ", 1)
        assert digest == hashlib.sha256((output / path).read_bytes()).hexdigest()


class FakeProcess:
    """An injected child tracks process-level versus group-level termination."""

    pid = 123

    def __init__(self, waits: list[int | BaseException]) -> None:
        self.waits = iter(waits)
        self.timeouts: list[float | None] = []
        self.returncode: int | None = None
        self.actions: list[str] = []

    def wait(self, timeout: float | None = None) -> int:
        self.timeouts.append(timeout)
        result = next(self.waits)
        if isinstance(result, BaseException):
            raise result
        self.returncode = result
        return result

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.actions.append("terminate")

    def kill(self) -> None:
        self.actions.append("kill")


@pytest.mark.parametrize("process_group", [False, True])
@pytest.mark.parametrize("force", [False, True])
def test_termination_retains_grace_period_and_platform_policy(
    monkeypatch: pytest.MonkeyPatch, process_group: bool, force: bool
) -> None:
    waits: list[int | BaseException] = (
        [subprocess.TimeoutExpired("child", 5), -9] if force else [-15]
    )
    process = FakeProcess(waits)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        _processes.os, "killpg", lambda pid, sig: signals.append((pid, sig)), raising=False
    )
    assert _processes.terminate_process(
        cast(subprocess.Popen[bytes], process), process_group=process_group
    ) == (-9 if force else -15)
    assert process.timeouts == ([5.0, None] if force else [5.0])
    if process_group:
        assert signals == [(123, signal.SIGTERM)] + ([(123, signal.SIGKILL)] if force else [])
        assert process.actions == []
    else:
        assert signals == []
        assert process.actions == ["terminate"] + (["kill"] if force else [])


@pytest.mark.parametrize("outcome", ["success", "timeout", "interrupt"])
def test_resident_child_preserves_timeout_and_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    waits: list[int | BaseException] = [0]
    if outcome != "success":
        waits = [
            subprocess.TimeoutExpired("child", 2) if outcome == "timeout" else KeyboardInterrupt(),
            -15,
        ]
    process = FakeProcess(waits)
    options: dict[str, Any] = {}

    def popen(command: list[str], **kwargs: Any) -> FakeProcess:
        options.update(kwargs)
        kwargs["stdout"].write("child log\n")
        return process

    signals: list[int] = []
    monkeypatch.setattr(_processes.subprocess, "Popen", popen)
    monkeypatch.setattr(
        _processes.os, "killpg", lambda pid, sig: signals.append(sig), raising=False
    )
    kwargs: dict[str, Any] = dict(
        cwd=tmp_path, env={}, log_path=tmp_path / "child.log", timeout=2.0
    )
    if outcome == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            run_resident_tests.run_child(["child"], **kwargs)
    else:
        report = run_resident_tests.run_child(["child"], **kwargs)
        assert report["completed"] is (outcome == "success")
        assert report["timed_out"] is (outcome == "timeout")
        assert report["returncode"] == (0 if outcome == "success" else -15)
    assert options["start_new_session"] is True
    assert options["stderr"] == subprocess.STDOUT
    assert signals == ([] if outcome == "success" else [signal.SIGTERM])
    assert (tmp_path / "child.log").read_text() == "child log\n"


def test_evidence_inventory_exclusions_and_checksum_order(tmp_path: Path) -> None:
    folder = tmp_path / "run"
    (folder / "nested").mkdir(parents=True)
    second, first = folder / "second.txt", folder / "nested/first.txt"
    second.write_bytes(b"second")
    first.write_bytes(b"first")
    hashes = _evidence.hash_files(folder, [second, first], posix_names=True)
    assert list(hashes) == ["second.txt", "nested/first.txt"]
    _evidence.write_checksums(folder / "SHA256SUMS", hashes)
    archive = tmp_path / "run.zip"
    _evidence.write_archive(folder, archive, posix_names=True)
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == {
            "run/second.txt",
            "run/nested/first.txt",
            "run/SHA256SUMS",
        }
        assert zipped.read("run/nested/first.txt") == b"first"
