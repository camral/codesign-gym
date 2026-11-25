#!/usr/bin/env python3
"""Bounded, fail-closed resident gate with cheap capture checks before physics.

Independent control/backsolve/adjoint canaries run in isolated subprocesses,
even when a sibling reports a diagnostic failure. All must pass before physics.
Timeouts still abort remaining GPU work to avoid compounding a stalled device. Every subprocess, timeout, selection, and source
hash is recorded. No historical full suite or optimizer search is launched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if __package__ in (None, ""):
    # Direct-file execution must select this checkout before package imports.
    _source_root = str(Path(__file__).resolve().parents[3])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)

from envs.soft_walker.diagnostics._evidence import hash_files, write_archive
from envs.soft_walker.diagnostics._processes import (
    start_logged_process,
    terminate_process,
)
from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
    require_source_checkout,
    source_group_files,
)

if TYPE_CHECKING:
    from pathlib import Path

    from envs.soft_walker.engine.types import (
        Array,
    )

ROOT = REPO_ROOT

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def write(path: Path, value: Any) -> None:
    """Serialize evidence to the requested path using this command's existing JSON encoder."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def source_hashes() -> dict[str, str]:
    """Hash the source files used to reproduce this command's evidence."""
    return hash_files(
        ROOT,
        (
            p
            for group in (
                "src/envs/soft_walker/engine",
                "tests/soft_walker",
                "src/envs/soft_walker/scripts",
                "src/envs/soft_walker/diagnostics",
            )
            for p in source_group_files(ROOT, group)
        ),
        posix_names=False,
    )


def run_child(
    command: Array, *, cwd: Path, env: dict[str, str], log_path: Path, timeout: float
) -> Report:
    """Run an isolated stage; retain explicit timeout versus ordinary failure."""
    entry: Report = {"command": list(command), "completed": False, "timed_out": False}
    start = time.perf_counter()
    process = None
    with log_path.open("w") as log:
        try:
            process = start_logged_process(
                command,
                cwd=cwd,
                env=env,
                log=log,
                process_group=True,
            )
            try:
                entry["returncode"] = process.wait(timeout=timeout)
                entry["completed"] = True
            except subprocess.TimeoutExpired:
                entry["timed_out"] = True
        finally:
            if process is not None and process.poll() is None:
                terminate_process(process)
                entry["returncode"] = process.returncode
            entry["seconds"] = time.perf_counter() - start
    return entry


def planned_stages(args: argparse.Namespace, base: Path) -> list[tuple[str, list[str], float]]:
    """List the explicit preflight/unit/driver subprocess stages and their time limits."""
    trace = ["--trace"] if args.trace else []
    stages = [
        (
            "preflight_" + name,
            [
                str(ROOT / "src/envs/soft_walker/diagnostics/resident_preflight.py"),
                "--probe",
                name,
                "--out",
                str(base / ("preflight_" + name)),
                *trace,
            ],
            args.preflight_timeout,
        )
        for name in ("control", "backsolve", "implicit_gradient")
    ]
    if args.preflight_only:
        return stages
    if not args.skip_unit:
        stages.append(
            (
                "unit",
                [
                    "-m",
                    "pytest",
                    "-q",
                    "tests/soft_walker/test_resident_trace_contract.py",
                    "tests/soft_walker/test_resident_completion.py",
                    "tests/soft_walker/test_resident_repair.py",
                    "tests/soft_walker/test_krylov_fused.py",
                    "tests/soft_walker/test_krylov_norm_failures.py",
                    "tests/soft_walker/test_resident_execution.py",
                    "-k",
                    "not whole_episode and not population",
                    "--junitxml=" + str(base / "unit.xml"),
                ],
                args.timeout,
            )
        )
    for driver in args.drivers:
        stages.append(
            (
                driver,
                [
                    str(ROOT / "src/envs/soft_walker/diagnostics/resident_worker.py"),
                    "--driver",
                    driver,
                    "--out",
                    str(base / driver),
                    *trace,
                ],
                args.timeout,
            )
        )
    return stages


def main() -> int:
    """Command entry point. Bounded, fail-closed resident gate with cheap capture checks before physics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--dtypes", nargs="+", choices=("float64", "float32"), default=["float64"])
    parser.add_argument(
        "--drivers", nargs="+", choices=("spring", "mpm2", "mpm3", "fem", "beam"), default=["mpm2"]
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per unit/physical stage, including compilation",
    )
    parser.add_argument("--preflight-timeout", type=float, default=90.0)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Only the three independent CUDA graph canaries",
    )
    parser.add_argument(
        "--skip-unit",
        action="store_true",
        help="Do not rerun unit tests; never skips capture checks",
    )
    parser.add_argument("--offline-spaces", action="store_true")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Additional detailed trace of SHORT canaries/physical workers",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be fresh")
    if not all(math.isfinite(x) and x > 0 for x in (args.timeout, args.preflight_timeout)):
        parser.error("Timeouts must be finite and positive")
    try:
        require_source_checkout(ROOT)
    except ValueError as exc:
        parser.error(str(exc))
    output.mkdir(parents=True, exist_ok=True)
    status: Report = {
        "completed": False,
        "passed": False,
        "stages": [],
        "request": vars(args),
        "source_sha256": source_hashes(),
        "full_historical_suite": False,
        "preflight_only_is_not_physics_acceptance": True,
        "independent_probe_policy": "collect all three; any failure blocks physics; timeout blocks remaining device work",
    }
    write(output / "manifest.json", status)
    start = time.perf_counter()
    try:
        for dtype in args.dtypes:
            base = output / dtype
            base.mkdir()
            blockers: list[str] = []
            device_stalled = False
            for name, command, timeout in planned_stages(args, base):
                independent_probe = name.startswith("preflight_")
                if device_stalled or (blockers and not independent_probe):
                    status["stages"].append(
                        {
                            "name": name,
                            "dtype": dtype,
                            "skipped": True,
                            "reason": "blocked by " + ", ".join(blockers),
                            "passed": False,
                        }
                    )
                    continue
                env = dict(
                    os.environ,
                    PYTHONPATH=str(ROOT / "src")
                    + os.pathsep
                    + str(ROOT / "src/envs/soft_walker/scripts"),
                    JAX_PLATFORMS=args.device,
                    JAX_ENABLE_X64="1" if dtype == "float64" else "0",
                    CODESIGN_TEST_DTYPE=dtype,
                    CODESIGN_RESIDENT_DEVICE=args.device,
                    OPENBLAS_NUM_THREADS="1",
                    OMP_NUM_THREADS="1",
                )
                env.setdefault(
                    "JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/codesign-jax")
                )
                if args.offline_spaces:
                    env["CODESIGN_OFFLINE_SPACES"] = "1"
                else:
                    env.pop("CODESIGN_OFFLINE_SPACES", None)
                entry: Report = {
                    "name": name,
                    "dtype": dtype,
                    "command": [sys.executable] + command,
                    "completed": False,
                    "timeout_seconds": timeout,
                }
                status["stages"].append(entry)
                write(output / "manifest.json", status)
                entry.update(
                    run_child(
                        entry["command"],
                        cwd=ROOT,
                        env=env,
                        log_path=base / (name + ".log"),
                        timeout=timeout,
                    )
                )
                entry["passed"] = entry["returncode"] == 0 and not entry["timed_out"]
                if not entry["passed"]:
                    blockers.append(name)
                    device_stalled = entry["timed_out"]
                write(output / "manifest.json", status)
                print(
                    f"{dtype}/{name}: returncode={entry['returncode']} {entry['seconds']:.2f}s",
                    flush=True,
                )
        status["completed"] = True
        status["passed"] = bool(status["stages"]) and all(
            s.get("passed", False) for s in status["stages"]
        )
    except BaseException as error:
        status["exception"] = repr(error)
    finally:
        status["wall_seconds"] = time.perf_counter() - start
        status["final_source_sha256"] = source_hashes()
        if status["final_source_sha256"] != status["source_sha256"]:
            status["passed"] = False
            status["source_changed_during_run"] = True
        write(output / "manifest.json", status)
        sums = hash_files(
            output,
            (p for p in sorted(output.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.json"),
            posix_names=False,
        )
        write(output / "SHA256SUMS.json", sums)
        archive = output.with_suffix(".zip")
        write_archive(output, archive, posix_names=False)
        print(f"Evidence: {archive}", flush=True)
    return 0 if status["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
