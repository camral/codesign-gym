#!/usr/bin/env python3
"""Change-focused parity, gradient, and performance evidence with failure archives.

Every stage has a watchdog and its own process. No optimizer search or historical
full-suite run. JAX oracles run on CPU; requested native CUDA never falls back.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def write(path: Path, value: Any) -> None:
    """Serialize evidence to the requested path using this command's existing JSON encoder."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def sources(root: Path) -> dict[str, str]:
    """Collect source file digests rooted at the supplied repository."""
    return hash_files(
        root,
        (
            p
            for folder in (
                "src/envs/soft_walker/engine",
                "tests/soft_walker",
                "src/envs/soft_walker/scripts",
                "src/envs/soft_walker/diagnostics",
            )
            for p in source_group_files(root, folder)
        ),
        posix_names=True,
    )


def main() -> int:
    """Command entry point. Change-focused parity, gradient, and performance evidence with failure archives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--dtypes", nargs="+", choices=["float64", "float32"], default=["float64", "float32"]
    )
    parser.add_argument(
        "--physics", nargs="+", choices=["spring", "fem", "beam"], default=["spring", "fem", "beam"]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--skip-unit", action="store_true")
    parser.add_argument("--skip-solids", action="store_true")
    parser.add_argument("--skip-mpm", action="store_true")
    parser.add_argument(
        "--skip-spring",
        action="store_true",
        help="Skip only the separate 1,400-step JAX CPU reference replay",
    )
    parser.add_argument(
        "--offline-spaces",
        action="store_true",
        help="Explicit sandbox-only adapter; never counts as Gymnasium validation",
    )
    args = parser.parse_args()
    if args.audit and args.device != "cuda":
        parser.error("--audit requires CUDA")
    if args.timeout <= 0 or args.repeats < 1:
        parser.error("Positive timeout and repeats required")
    if len(set(args.dtypes)) != len(args.dtypes) or len(set(args.physics)) != len(args.physics):
        parser.error("Duplicate selections are not allowed")
    root = REPO_ROOT
    args.out = args.out.resolve()
    try:
        require_source_checkout(root)
    except ValueError as exc:
        parser.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=False)
    selected = [
        "tests/soft_walker/test_warp_followup_v2.py",
        "tests/soft_walker/test_warp_all_mpm_edges.py::test_production_adaptive_all_options_and_changed_commands",
    ]
    stages: list[Report] = []
    before = sources(root)
    versions: Report = {}
    for name in ("jax", "jaxlib", "warp-lang", "numpy", "scipy", "gymnasium", "pytest"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    manifest: Report = {
        "arguments": vars(args),
        "source_sha256": before,
        "versions": versions,
        "python": sys.version,
        "platform": platform.platform(),
        "selected_unit_tests": selected,
        "offline_spaces": args.offline_spaces,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write(args.out / "manifest.json", manifest)
    planned = []
    for dtype in args.dtypes:
        base = dict(
            os.environ,
            PYTHONPATH=str(root / "src"),
            JAX_PLATFORMS="cpu",
            JAX_ENABLE_X64="1" if dtype == "float64" else "0",
            CODESIGN_TEST_DTYPE=dtype,
            WARP_TEST_DTYPE=dtype,
            WARP_TEST_DEVICE=args.device,
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        if args.offline_spaces:
            base["CODESIGN_OFFLINE_SPACES"] = "1"
        else:
            base.pop("CODESIGN_OFFLINE_SPACES", None)
        if not args.skip_unit:
            name = f"{dtype}/unit"
            planned.append(
                (
                    name,
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        *selected,
                        "--junitxml=" + str(args.out / name / "junit.xml"),
                    ],
                    base,
                )
            )
        if not args.skip_solids:
            for physics in args.physics:
                name = f"{dtype}/{physics}"
                command = [
                    sys.executable,
                    "src/envs/soft_walker/diagnostics/warp_solid_worker.py",
                    "--physics",
                    physics,
                    "--dtype",
                    dtype,
                    "--device",
                    args.device,
                    "--out",
                    str(args.out / name / "numerics"),
                ]
                if args.audit:
                    command.append("--audit")
                planned.append((name, command, base))
        if not args.skip_mpm:
            name = f"{dtype}/mpm"
            command = [
                sys.executable,
                "src/envs/soft_walker/diagnostics/warp_followup_mpm.py",
                "--dtype",
                dtype,
                "--device",
                args.device,
                "--repeats",
                str(args.repeats),
                "--out",
                str(args.out / name / "numerics"),
            ]
            if args.audit:
                command.append("--audit")
            planned.append((name, command, base))
    if not args.skip_spring:
        name = "spring_reference"
        base = dict(
            os.environ,
            PYTHONPATH=str(root / "src"),
            JAX_PLATFORMS="cpu",
            JAX_ENABLE_X64="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        if args.offline_spaces:
            base["CODESIGN_OFFLINE_SPACES"] = "1"
        else:
            base.pop("CODESIGN_OFFLINE_SPACES", None)
        planned.append(
            (
                name,
                [
                    sys.executable,
                    "src/envs/soft_walker/diagnostics/warp_followup_spring.py",
                    "--gradient",
                    "--out",
                    str(args.out / name / "numerics"),
                ],
                base,
            )
        )
    summary: Report = {
        "success": False,
        "complete": False,
        "stages": stages,
        "planned": [p[0] for p in planned],
    }
    write(args.out / "summary.json", summary)
    if not planned:
        summary["error"] = "Empty selection"
    try:
        for name, command, env in planned:
            directory = args.out / name
            directory.mkdir(parents=True, exist_ok=False)
            write(directory / "command.json", command)
            start = time.perf_counter()
            timed_out = False
            with (directory / "stdout.log").open("w") as stream:
                process = start_logged_process(
                    command,
                    cwd=root,
                    env=env,
                    log=stream,
                    process_group=True,
                )
                try:
                    code = process.wait(timeout=args.timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    code = kill_process(process)
                except BaseException:
                    kill_process(process)
                    raise
            row: Report = {
                "name": name,
                "returncode": code,
                "seconds": time.perf_counter() - start,
                "timed_out": timed_out,
            }
            stages.append(row)
            write(directory / "stage.json", row)
            write(args.out / "progress.json", {"complete": False, "stages": stages})
            print(json.dumps(row), flush=True)
            if timed_out:
                break
    except BaseException as exc:
        summary["error"] = repr(exc)
        raise
    finally:
        same = before == sources(root)
        summary.update(
            stages=stages,
            complete=bool(planned) and len(stages) == len(planned),
            source_unchanged=same,
            success=bool(planned)
            and len(stages) == len(planned)
            and all(s["returncode"] == 0 for s in stages)
            and same
            and "error" not in summary,
        )
        write(args.out / "summary.json", summary)
        hashes = {
            p.relative_to(args.out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(args.out.rglob("*"))
            if p.is_file() and p.name != "file_hashes.json"
        }
        write(args.out / "file_hashes.json", hashes)
        shutil.make_archive(str(args.out), "zip", args.out.parent, args.out.name)
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
