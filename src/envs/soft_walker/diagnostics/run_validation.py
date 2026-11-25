#!/usr/bin/env python3
"""Run serial isolated evidence stages; ZIP evidence even after stage failure.

This parent uses only the standard library so missing JAX/CUDA dependencies and
native subprocess crashes remain diagnosable. No background delivery service.
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
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if __package__ in (None, ""):
    # Direct-file execution must select this checkout before package imports.
    _source_root = str(Path(__file__).resolve().parents[3])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)

from envs.soft_walker.diagnostics.validation_selection import (
    AFFECTED_TESTS,
    CHANGED_STAGES,
    FULL_STAGES,
    LEGACY_STAGES as LEGACY_STAGES,
    M3_STAGES as M3_STAGES,
    REASONS,
)
from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
    diagnostic_sources,
    require_source_checkout,
)

if TYPE_CHECKING:
    from envs.soft_walker.engine.types import (
        PathLike,
    )

ROOT = REPO_ROOT

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def write(path: PathLike, obj: Any) -> None:
    """Serialize evidence to the requested path using this command's existing JSON encoder."""
    Path(path).write_text(json.dumps(obj, indent=2) + "\n")


def main() -> int:
    """Command entry point. Run serial isolated evidence stages; ZIP evidence even after stage failure."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument(
        "--dtypes", nargs="+", choices=["float64", "float32"], default=["float64", "float32"]
    )
    p.add_argument(
        "--suite",
        choices=["changed", "full"],
        default="changed",
        help="changed: new tests plus explicit affected regressions; full: all milestones",
    )
    p.add_argument("--stages", nargs="+", choices=FULL_STAGES + ["affected"], default=None)
    p.add_argument(
        "--cache-dir",
        default=str(ROOT / ".jax_cache"),
        help="trusted local persistent JAX cache; never included in evidence ZIP",
    )
    p.add_argument(
        "--cold-cache",
        action="store_true",
        help="use a new unique cache namespace for cold-compile measurements",
    )
    p.add_argument("--no-cache", action="store_true")
    p.add_argument(
        "--case-filter",
        default="*",
        help="fnmatch filter for M3 audit/benchmark model names; does not filter pytest",
    )
    p.add_argument(
        "--unit-filter",
        default="",
        help="optional pytest -k expression; marks this as a filtered run",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument(
        "--timeout",
        type=float,
        default=1800,
        help="per-stage subprocess timeout; partial logs are retained",
    )
    p.add_argument("--dump-ir", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--log-compiles", action="store_true")
    p.add_argument(
        "--offline-spaces",
        action="store_true",
        help="explicit local-development adapter; NOT Gymnasium integration validation",
    )
    a = p.parse_args()
    a.stages = (
        a.stages
        if a.stages is not None
        else (CHANGED_STAGES if a.suite == "changed" else FULL_STAGES)
    )
    if a.cold_cache and a.no_cache:
        p.error("choose --cold-cache or --no-cache, not both")
    if a.repeats < 3:
        p.error("--repeats must be >=3")
    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        p.error("Output directory is not empty; choose a new evidence directory")
    try:
        require_source_checkout(ROOT)
    except ValueError as exc:
        p.error(str(exc))
    out.mkdir(parents=True, exist_ok=True)
    manifest: Report = {
        "format": "codesign-m3-evidence-v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "device": a.device,
        "dtypes": a.dtypes,
        "stages": [],
        "offline_spaces_explicit": a.offline_spaces,
        "source_sha256": {},
        "completed": False,
        "suite": a.suite,
        "selected_stages": a.stages,
        "unit_filter": a.unit_filter,
        "case_filter": a.case_filter,
        "affected_tests": AFFECTED_TESTS if "affected" in a.stages else [],
        "affected_reasons": REASONS,
        "full_suite_certification": a.stages == FULL_STAGES
        and not a.unit_filter
        and a.case_filter == "*",
    }
    cache = Path(a.cache_dir).expanduser().resolve()
    if a.cold_cache:
        import uuid

        cache = cache / ("cold-" + uuid.uuid4().hex)
    if not a.no_cache:
        cache.mkdir(parents=True, exist_ok=True)

    def cache_stats() -> Report:
        fs = list(cache.rglob("*")) if cache.exists() else []
        fs = [f for f in fs if f.is_file()]
        return {"files": len(fs), "bytes": sum(f.stat().st_size for f in fs)}

    manifest["compilation_cache"] = {
        "enabled": not a.no_cache,
        "path": str(cache),
        "cold_namespace": a.cold_cache,
        "before": cache_stats(),
        "trusted_local_only": True,
    }
    if out == cache or out in cache.parents:
        p.error("cache directory must be outside the evidence directory")
    files = sorted(
        list((ROOT / "src/envs/soft_walker/engine").rglob("*.py"))
        + list((ROOT / "src/envs/soft_walker/scripts").glob("*.py"))
        + diagnostic_sources(ROOT)
        + list((ROOT / "tests/soft_walker").glob("*.py"))
        + list((ROOT / "src/envs/soft_walker/examples").glob("*.py"))
        + list((ROOT / "src/envs/soft_walker/baselines").glob("*.json"))
        + [ROOT / "pyproject.toml", ROOT / "src/envs/soft_walker/requirements-tested.txt"]
    )
    for f in files:
        manifest["source_sha256"][str(f.relative_to(ROOT))] = hashlib.sha256(
            f.read_bytes()
        ).hexdigest()
    write(out / "manifest.json", manifest)
    for cmd, name in ((["nvidia-smi", "-q"], "nvidia-smi.txt"), (["lscpu"], "lscpu.txt")):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            (out / name).write_text(r.stdout + "\nSTDERR:\n" + r.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            (out / name).write_text(repr(exc) + "\n")
    monitor = None
    monitor_file = None
    child = None
    try:
        if a.device == "cuda" and shutil.which("nvidia-smi"):
            monitor_file = (out / "gpu_telemetry.csv").open("w")
            query = "timestamp,index,name,utilization.gpu,utilization.memory,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem"
            monitor = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv", "-lms", "500"],
                stdout=monitor_file,
                stderr=subprocess.STDOUT,
            )
        for dtype in a.dtypes:
            for stage in a.stages:
                dest = out / dtype / stage
                dest.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env.update(
                    {
                        "JAX_PLATFORMS": "cpu" if a.device == "cpu" else "cuda",
                        "JAX_ENABLE_X64": "1" if dtype == "float64" else "0",
                        "CODESIGN_TEST_DTYPE": dtype,
                        "CODESIGN_EVIDENCE_DIR": str(dest),
                        "JAX_TRACEBACK_FILTERING": "off",
                        "JAX_ENABLE_COMPILATION_CACHE": "false" if a.no_cache else "true",
                        "JAX_COMPILATION_CACHE_DIR": str(cache / a.device / dtype),
                        "CODESIGN_UNIT_FILTER": a.unit_filter,
                        "CODESIGN_CASE_FILTER": a.case_filter,
                        "PYTHONFAULTHANDLER": "1",
                        "PYTHONHASHSEED": "0",
                        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                        "PYTHONPATH": str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
                    }
                )
                env["CODESIGN_OFFLINE_SPACES"] = "1" if a.offline_spaces else "0"
                if a.log_compiles:
                    env["JAX_LOG_COMPILES"] = "1"
                    env["JAX_DEBUG_LOG_MODULES"] = "jax._src.compiler,jax._src.lru_cache"
                    env["JAX_EXPLAIN_CACHE_MISSES"] = "1"
                cmd = [
                    sys.executable,
                    "-u",
                    str(ROOT / "src/envs/soft_walker/diagnostics" / "worker.py"),
                    "--out",
                    str(dest),
                    "--stage",
                    stage,
                    "--device",
                    a.device,
                    "--dtype",
                    dtype,
                    "--repeats",
                    str(a.repeats),
                ]
                if a.dump_ir:
                    cmd.append("--dump-ir")
                if a.profile:
                    cmd.append("--profile")
                print(f"Running {a.device}/{dtype}/{stage}; log: {dest / 'stdout.log'}", flush=True)
                start = time.perf_counter()
                with (dest / "stdout.log").open("w") as log:
                    child = subprocess.Popen(
                        cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                    try:
                        rc = child.wait(timeout=a.timeout)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                        rc = 124
                        log.write("\nPARENT: stage timed out; subprocess killed.\n")
                row: Report = {
                    "dtype": dtype,
                    "stage": stage,
                    "returncode": rc,
                    "wall_s": time.perf_counter() - start,
                    "log": str((dest / "stdout.log").relative_to(out)),
                }
                text = (dest / "stdout.log").read_text(errors="replace").lower()
                row["cache_log_lines"] = {
                    "persistent_hits": text.count("persistent compilation cache hit"),
                    "persistent_misses": text.count("persistent compilation cache miss"),
                    "interpretation": "matching log lines, not a cache hit-rate estimator",
                }
                # Negative codes denote native signals; useful when Python cannot
                # write its own traceback (driver crash, XLA abort, OOM kill).
                if rc < 0:
                    row["signal"] = -rc
                manifest["stages"].append(row)
                write(out / "manifest.json", manifest)
                print(
                    f"Finished {dtype}/{stage}: returncode={rc}, wall={row['wall_s']:.3f}s",
                    flush=True,
                )
                child = None
        if {"float64", "float32"}.issubset(set(a.dtypes)) and "numerics" in a.stages:
            logpath = out / "precision_comparison.log"
            with logpath.open("w") as log:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "src/envs/soft_walker/diagnostics" / "compare_precision.py"),
                        str(out),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            manifest["stages"].append(
                {
                    "dtype": "mixed",
                    "stage": "precision_comparison",
                    "returncode": proc.returncode,
                    "log": logpath.name,
                }
            )
        if {"float64", "float32"}.issubset(set(a.dtypes)) and "m2_numerics" in a.stages:
            logpath = out / "m2_precision_comparison.log"
            with logpath.open("w") as log:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "src/envs/soft_walker/diagnostics" / "compare_m2_precision.py"),
                        str(out),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            manifest["stages"].append(
                {
                    "dtype": "mixed",
                    "stage": "m2_precision_comparison",
                    "returncode": proc.returncode,
                    "log": logpath.name,
                }
            )
        if {"float64", "float32"}.issubset(set(a.dtypes)) and "m3_numerics" in a.stages:
            logpath = out / "m3_precision_comparison.log"
            with logpath.open("w") as log:
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "src/envs/soft_walker/diagnostics" / "compare_m3_precision.py"),
                        str(out),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            manifest["stages"].append(
                {
                    "dtype": "mixed",
                    "stage": "m3_precision_comparison",
                    "returncode": proc.returncode,
                    "log": logpath.name,
                }
            )
        manifest["completed"] = True
    except KeyboardInterrupt:
        manifest["interrupted"] = True
        if child is not None:
            child.kill()
            child.wait()
    finally:
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
        if monitor_file is not None:
            monitor_file.close()
        passed = manifest["completed"] and all(x["returncode"] == 0 for x in manifest["stages"])
        manifest["status"] = (
            ("passed_with_offline_spaces_unvalidated" if a.offline_spaces else "passed")
            if passed
            else "failed_or_interrupted"
        )
        manifest["compilation_cache"]["after"] = cache_stats()
        write(out / "manifest.json", manifest)
        hashes = []
        for f in sorted(out.rglob("*")):
            if f.is_file() and f.name != "SHA256SUMS":
                hashes.append(
                    hashlib.sha256(f.read_bytes()).hexdigest() + "  " + str(f.relative_to(out))
                )
        (out / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
        archive = out.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    z.write(f, arcname=str(Path(out.name) / f.relative_to(out)))
        print(f"Evidence status: {manifest['status']}\nEvidence ZIP: {archive}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
