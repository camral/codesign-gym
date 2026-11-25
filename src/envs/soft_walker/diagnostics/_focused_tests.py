"""Shared focused pytest execution and evidence lifecycle; selection stays in each CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from envs.soft_walker.diagnostics._evidence import hash_files, write_archive, write_checksums
from envs.soft_walker.shared import source_group_files

Report = dict[str, Any]


def run_focused_tests(
    args: argparse.Namespace,
    output: Path,
    *,
    root: Path,
    src_root: Path,
    selection: list[str],
    schema: str,
    write_json: Callable[[Path, Any], None],
    stage_environment: Callable[[Path], dict[str, str]],
) -> int:
    """Run isolated precision stages and archive partial evidence on any failure."""
    files = sorted(
        p
        for folder in (
            "src/envs/soft_walker/engine",
            "tests/soft_walker",
            "src/envs/soft_walker/scripts",
            "src/envs/soft_walker/diagnostics",
        )
        for p in source_group_files(root, folder)
        if "__pycache__" not in p.parts
    )
    manifest: Report = {
        "schema": schema,
        "completed": False,
        "device_requested": args.device,
        "dtypes": args.dtypes,
        "offline_spaces_requested": args.offline_spaces,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "source_sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "selection": selection,
        "stages": [],
        "scope": "Infrastructure and isolated numerics; not a walking-gait qualification",
    }
    write_json(output / "manifest.json", manifest)
    passed = True
    try:
        for dtype in args.dtypes:
            stage = output / dtype
            stage.mkdir()
            env = os.environ.copy()
            env["PYTHONPATH"] = str(src_root) + os.pathsep + env.get("PYTHONPATH", "")
            env.update(
                JAX_PLATFORMS=args.device,
                CODESIGN_TEST_DTYPE=dtype,
                JAX_ENABLE_X64="1" if dtype == "float64" else "0",
            )
            if args.offline_spaces:
                env["CODESIGN_OFFLINE_SPACES"] = "1"
            else:
                env.pop("CODESIGN_OFFLINE_SPACES", None)
            env.update(stage_environment(stage))
            command = [
                sys.executable,
                "-m",
                "pytest",
                *selection,
                "-q",
                "--junitxml=" + str(stage / "junit.xml"),
            ]
            write_json(
                stage / "command.json",
                {
                    "argv": command,
                    "cwd": root,
                    "selected_environment": {
                        key: env.get(key)
                        for key in (
                            "JAX_PLATFORMS",
                            "JAX_ENABLE_X64",
                            "CUDA_VISIBLE_DEVICES",
                            "CODESIGN_TEST_DTYPE",
                            "CODESIGN_OFFLINE_SPACES",
                            "JAX_COMPILATION_CACHE_DIR",
                            "XLA_FLAGS",
                        )
                    },
                },
            )
            result: Report = {"dtype": dtype, "timed_out": False}
            start = time.perf_counter()
            with (stage / "stdout.log").open("w") as log:
                try:
                    process = subprocess.run(
                        command,
                        cwd=root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=args.timeout,
                    )
                    result["returncode"] = process.returncode
                except subprocess.TimeoutExpired:
                    result.update(timed_out=True, returncode=124)
                except OSError as error:
                    result.update(returncode=1, error=str(error))
            result["wall_s"] = time.perf_counter() - start
            if (stage / "junit.xml").exists():
                try:
                    suites = ET.parse(stage / "junit.xml").getroot()
                    result["junit"] = [suite.attrib for suite in suites.iter("testsuite")]
                except ET.ParseError as error:
                    result["junit_error"] = str(error)
            passed &= result["returncode"] == 0
            manifest["stages"].append(result)
            write_json(output / "manifest.json", manifest)
            print(json.dumps(result), flush=True)
        manifest.update(completed=True, passed=passed)
    finally:
        manifest.setdefault("passed", False)
        write_json(output / "manifest.json", manifest)
        hashes = hash_files(
            output,
            (p for p in sorted(output.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"),
            posix_names=False,
        )
        write_checksums(output / "SHA256SUMS", hashes)
        archive = output.with_name(output.name + ".zip")
        write_archive(output, archive, posix_names=False)
        print(f"Evidence: {archive}", flush=True)
    return 0 if manifest.get("passed") else 1
