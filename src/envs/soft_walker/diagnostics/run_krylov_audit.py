#!/usr/bin/env python3
"""Run ONLY the new Krylov checks and a paired physical interval benchmark.

A fresh process is used per precision/stage. Every stage, including timeout or
CUDA-initialization failure, leaves logs; the complete run is archived. Default
benchmark precision is float64, matching the returned settled biped fixture.
No optimizer search, historical broad suite, or renderer is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if __package__ in (None, ""):
    # Direct-file execution must select this checkout before package imports.
    _source_root = str(Path(__file__).resolve().parents[3])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)

from envs.soft_walker.diagnostics._evidence import hash_files, write_archive, write_checksums
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

    from numpy.typing import DTypeLike

    from envs.soft_walker.engine.types import (
        Array,
    )

ROOT = REPO_ROOT

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def write(path: Path, data: Any) -> None:
    """Serialize evidence to the requested path using this command's existing JSON encoder."""
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def archive(out: Path) -> Path:
    """Package the completed or partial run directory into its sibling ZIP artifact."""
    entries = hash_files(
        out,
        (p for p in sorted(out.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"),
        posix_names=True,
    )
    write_checksums(out / "SHA256SUMS", entries)
    path = out.with_name(out.name + ".zip")
    write_archive(out, path, posix_names=True)
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise RuntimeError("Output ZIP failed CRC check")
    return path


def main() -> int:
    """Command entry point. Run ONLY the new Krylov checks and a paired physical interval benchmark."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument(
        "--dtypes", nargs="+", choices=("float32", "float64"), default=["float64", "float32"]
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--trace", action="store_true")
    p.add_argument(
        "--gradient", action="store_true", help="Also compare a full-interval physical adjoint"
    )
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--skip-benchmark", action="store_true")
    p.add_argument("--timeout", type=float, default=900.0)
    p.add_argument(
        "--offline-spaces",
        action="store_true",
        help="Explicit sandbox-only adapter; not Gymnasium validation",
    )
    args = p.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        p.error("Invalid repetition/timeout value")
    if args.skip_tests and args.skip_benchmark:
        p.error("No stages selected")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        p.error("Use a fresh output directory")
    try:
        require_source_checkout(ROOT)
    except ValueError as exc:
        p.error(str(exc))
    out.mkdir(parents=True, exist_ok=True)
    status: Report = {"completed": False, "passed": False, "stages": []}
    write(out / "command.json", vars(args))
    write(
        out / "manifest.json",
        {
            "source_sha256": {
                f.relative_to(ROOT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
                for folder in (
                    "src/envs/soft_walker/engine",
                    "src/envs/soft_walker/scripts",
                    "src/envs/soft_walker/diagnostics",
                    "tests/soft_walker",
                )
                for f in source_group_files(ROOT, folder)
            },
            "python": sys.version,
            "platform": platform.platform(),
            "device": args.device,
            "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
            "selection": "new Krylov unit/physical/numerical cases plus 5 affected legacy cases; no historical broad suite",
            "offline_spaces_requested": args.offline_spaces,
        },
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["JAX_PLATFORMS"] = args.device
    env.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path.home() / ".cache/codesign-jax"))
    if args.offline_spaces:
        env["CODESIGN_OFFLINE_SPACES"] = "1"
    else:
        env.pop("CODESIGN_OFFLINE_SPACES", None)
    failures = 0

    def stage(name: str, dtype: DTypeLike, command: Array) -> None:
        nonlocal failures
        path = out / name
        path.mkdir(parents=True)
        local = env.copy()
        local["JAX_ENABLE_X64"] = "1" if dtype == "float64" else "0"
        local["CODESIGN_TEST_DTYPE"] = dtype
        command = [
            str(path) if s == "{OUT}" else str(path / "junit.xml") if s == "{JUNIT}" else s
            for s in command
        ]
        write(path / "command.json", {"argv": command, "cwd": str(ROOT), "dtype": dtype})
        started = time.perf_counter()
        timed_out = False
        with (path / "stdout.log").open("w") as log:
            process = start_logged_process(
                command,
                cwd=ROOT,
                env=local,
                log=log,
                process_group=os.name == "posix",
            )
            try:
                code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_process(process, process_group=os.name == "posix")
                code = 124
        record: Report = {
            "stage": name,
            "dtype": dtype,
            "returncode": code,
            "timed_out": timed_out,
            "wall_s": time.perf_counter() - started,
        }
        status["stages"].append(record)
        if code != 0:
            failures += 1
        write(out / "status.json", status)
        print(f"{name}: returncode={code}; {record['wall_s']:.3f}s", flush=True)

    try:
        flags = env.get("XLA_FLAGS", "")
        if "xla_gpu_enable_command_buffer" in flags and "WHILE" in flags:
            raise RuntimeError(
                "WHILE command-buffer capture failed in returned evidence. Remove that experimental override for this audit."
            )
        if not args.skip_tests:
            for dtype in dict.fromkeys(args.dtypes):
                # JUnit names are known before the child imports JAX.
                stage(
                    dtype + "/unit",
                    dtype,
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "tests/soft_walker/test_krylov_fused.py",
                        "tests/soft_walker/test_krylov_norm_failures.py",
                        "tests/soft_walker/test_krylov_audit_guards.py",
                        "tests/soft_walker/test_performance.py::test_performance_gmres_value_gradient_and_failure",
                        "tests/soft_walker/test_performance.py::test_performance_gmres_zero_rhs_and_nonlinear_root",
                        "tests/soft_walker/test_performance.py::test_performance_invalid_gmres_method",
                        "--junitxml",
                        "{JUNIT}",
                    ],
                )
                stage(
                    dtype + "/physics",
                    dtype,
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "tests/soft_walker/test_krylov_physics.py",
                        "--junitxml",
                        "{JUNIT}",
                    ],
                )
                stage(
                    dtype + "/numerics",
                    dtype,
                    [
                        sys.executable,
                        "src/envs/soft_walker/diagnostics/audit_krylov_physics.py",
                        "numerics",
                        "--device",
                        args.device,
                        "--dtype",
                        dtype,
                        "--out",
                        "{OUT}",
                    ],
                )
        if not args.skip_benchmark:
            command = [
                sys.executable,
                "src/envs/soft_walker/diagnostics/audit_krylov_physics.py",
                "benchmark",
                "--device",
                args.device,
                "--dtype",
                "float64",
                "--repeats",
                str(args.repeats),
                "--out",
                "{OUT}",
            ]
            if args.trace:
                command.append("--trace")
            if args.gradient:
                command.append("--gradient")
            stage("benchmark_float64", "float64", command)
        status.update(completed=True, passed=failures == 0)
    except BaseException as exc:
        failures += 1
        status.update(error=repr(exc), traceback=traceback.format_exc())
        print(status["traceback"], file=sys.stderr)
    finally:
        write(out / "status.json", status)
        path = archive(out)
        print("Evidence:", path, flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
