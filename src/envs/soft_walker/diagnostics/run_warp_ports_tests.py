#!/usr/bin/env python3
"""Focused native-port validation, JAX parity, gradient ladders and evidence ZIP.

Each precision/model runs in its own process. The historical suite and optimizer
searches are not executed. Requested CUDA never falls back to CPU. The JAX oracle
runs on CPU intentionally; this is numerical parity, not a speed comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
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

from envs.soft_walker.diagnostics._evidence import hash_files
from envs.soft_walker.diagnostics._processes import (
    kill_process,
    start_logged_process,
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

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def write(path: Path, data: Any) -> None:
    """Serialize evidence to the requested path using this command's existing JSON encoder."""
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def hashes(root: Path) -> dict[str, str]:
    """Collect source file digests for the evidence manifest."""
    return hash_files(
        root,
        (
            p
            for name in (
                "src/envs/soft_walker/engine",
                "src/envs/soft_walker/scripts",
                "src/envs/soft_walker/diagnostics",
                "tests/soft_walker",
            )
            for p in source_group_files(root, name)
        ),
        posix_names=False,
    )


def main() -> int:
    """Command entry point. Focused native-port validation, JAX parity, gradient ladders and evidence ZIP."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument(
        "--dtypes", nargs="+", choices=["float64", "float32"], default=["float64", "float32"]
    )
    p.add_argument(
        "--physics", nargs="+", choices=["spring", "beam", "fem"], default=["spring", "beam", "fem"]
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--skip-parity", action="store_true")
    p.add_argument(
        "--audit",
        action="store_true",
        help="Require one native CUDA graph for whole forward and gradient",
    )
    p.add_argument(
        "--offline-spaces",
        action="store_true",
        help="Explicit sandbox-only adapter; not Gymnasium validation",
    )
    args = p.parse_args()
    if args.skip_tests and args.skip_parity:
        p.error("Empty test selection")
    if args.audit and args.device != "cuda":
        p.error("--audit requires CUDA")
    if args.timeout <= 0:
        p.error("Positive timeout required")
    root = REPO_ROOT
    args.out = args.out.resolve()
    try:
        require_source_checkout(root)
    except ValueError as exc:
        p.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=False)
    before = hashes(root)
    selected_tests = [
        "tests/soft_walker/test_warp_additions.py",
        "tests/soft_walker/test_warp_all_contracts.py",
        "tests/soft_walker/test_warp_all_mpm_edges.py",
    ]
    manifest: Report = dict(
        arguments=vars(args),
        python=sys.version,
        source_sha256=before,
        numerical_oracle="JAX CPU, not used by native kernels",
        offline_spaces=args.offline_spaces,
        selected_tests=selected_tests,
    )
    write(args.out / "manifest.json", manifest)
    stages = []

    def run(name: str, command: Array, env: dict[str, str]) -> tuple[bool, bool]:
        directory = args.out / name
        directory.mkdir(parents=True, exist_ok=False)
        write(directory / "command.json", command)
        start = time.perf_counter()
        timed_out = False
        with (directory / "stdout.log").open("w") as stream:
            proc = start_logged_process(
                command,
                cwd=root,
                env=env,
                log=stream,
                process_group=True,
            )
            try:
                code = proc.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                code = kill_process(proc)
        record: Report = dict(
            name=name, returncode=code, seconds=time.perf_counter() - start, timed_out=timed_out
        )
        write(directory / "stage.json", record)
        stages.append(record)
        write(args.out / "progress.json", dict(stages=stages, complete=False))
        print(json.dumps(record), flush=True)
        return code == 0, timed_out

    success = True
    try:
        stop = False
        for precision in args.dtypes:
            env = dict(
                os.environ,
                PYTHONPATH=str(root / "src"),
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
                JAX_ENABLE_X64="1" if precision == "float64" else "0",
                JAX_PLATFORMS="cpu",
                CODESIGN_TEST_DTYPE=precision,
                WARP_TEST_DTYPE=precision,
                WARP_TEST_DEVICE=args.device,
            )
            if args.offline_spaces:
                env["CODESIGN_OFFLINE_SPACES"] = "1"
            else:
                env.pop("CODESIGN_OFFLINE_SPACES", None)
            if not args.skip_tests:
                name = f"{precision}/unit"
                command = [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    *selected_tests,
                    "--junitxml=" + str(args.out / name / "junit.xml"),
                ]
                okay, stop = run(name, command, env)
                success &= okay
            if stop:
                break
            if not args.skip_parity:
                for physics in args.physics:
                    name = f"{precision}/{physics}"
                    command = [
                        sys.executable,
                        "src/envs/soft_walker/diagnostics/warp_solid_worker.py",
                        "--physics",
                        physics,
                        "--dtype",
                        precision,
                        "--device",
                        args.device,
                        "--out",
                        str(args.out / name / "numerics"),
                    ]
                    if args.audit:
                        command.append("--audit")
                    okay, stop = run(name, command, env)
                    success &= okay
                    if stop:
                        break
            if stop:
                break
    finally:
        same = before == hashes(root)
        summary: Report = dict(
            success=bool(success and same and stages and all(s["returncode"] == 0 for s in stages)),
            complete=len(stages)
            == len(args.dtypes)
            * ((not args.skip_tests) + len(args.physics) * (not args.skip_parity)),
            source_unchanged=same,
            stages=stages,
        )
        write(args.out / "summary.json", summary)
        sums = {
            str(path.relative_to(args.out)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.out.rglob("*"))
            if path.is_file()
        }
        write(args.out / "file_hashes.json", sums)
        shutil.make_archive(str(args.out), "zip", args.out.parent, args.out.name)
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
