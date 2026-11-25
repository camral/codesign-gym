"""Run only the performance-patch tests and focused profiling stages.

Preserves logs on failure/timeouts and packages evidence automatically. It never
runs the full M1/M2/M3 suites or silently falls back from CUDA to CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Direct-file execution must select this checkout before package imports.
    _source_root = str(Path(__file__).resolve().parents[3])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)

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

ROOT = REPO_ROOT

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def digest(path: Path) -> str:
    """Return the SHA-256 digest of file bytes for artifact provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    """Serialize evidence to the requested path using this command's existing JSON encoder."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def source_hashes() -> dict[str, str]:
    """Hash the source files used to reproduce this command's evidence."""
    files: list[Path] = []
    for dirname in (
        "src/envs/soft_walker/engine",
        "src/envs/soft_walker/scripts",
        "src/envs/soft_walker/diagnostics",
        "tests/soft_walker",
    ):
        files.extend(source_group_files(ROOT, dirname))
    files.extend([ROOT / "pyproject.toml", ROOT / "src/envs/soft_walker/requirements-tested.txt"])
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def main() -> int:
    """Command entry point. Run only the performance-patch tests and focused profiling stages."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument(
        "--dtypes", nargs="+", choices=["float64", "float32"], default=["float64", "float32"]
    )
    p.add_argument(
        "--drivers", nargs="+", choices=["spring", "fem", "mpm2", "mpm3", "beam"], default=["mpm2"]
    )
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["incremental", "batched"],
        default=["incremental", "batched"],
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--size", choices=["small", "medium"], default="small")
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--log-compiles", action="store_true")
    p.add_argument(
        "--skip-tests",
        action="store_true",
        help="Profiling-only rerun; recorded as skipped, never a unit-test pass",
    )
    p.add_argument(
        "--offline-spaces", action="store_true", help="Explicit offline numerical development only"
    )
    p.add_argument(
        "--cpu-limit",
        type=int,
        default=0,
        help="Optional controlled affinity; 0 preserves workstation affinity",
    )
    args = p.parse_args()
    if args.repeats < 3 or args.batch_size < 1 or args.timeout <= 0 or args.cpu_limit < 0:
        p.error("Invalid repeat, batch, timeout or CPU-limit setting")
    out = args.out.expanduser().resolve()
    try:
        require_source_checkout(ROOT)
    except ValueError as exc:
        p.error(str(exc))
    out.mkdir(parents=True, exist_ok=False)
    if args.cpu_limit and hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[: args.cpu_limit])
    manifest: Report = {
        "schema": "codesign_performance_audit_v1",
        "completed": False,
        "passed": False,
        "utc": datetime.now(timezone.utc).isoformat(),
        "device": args.device,
        "dtypes": args.dtypes,
        "source_hashes": source_hashes(),
        "stages": [],
        "tests_skipped": args.skip_tests,
        "command": sys.argv,
        "scope": "New batching/solver-method tests and one affected vector-autoreset regression; NOT the full historical suite.",
    }
    write(out / "manifest.json", manifest)
    interrupted = None
    try:
        for dtype in dict.fromkeys(args.dtypes):
            env = dict(os.environ)
            env.update(
                JAX_PLATFORMS=args.device,
                JAX_ENABLE_X64="1" if dtype == "float64" else "0",
                CODESIGN_TEST_DTYPE=dtype,
                CODESIGN_OFFLINE_SPACES="1" if args.offline_spaces else "0",
                PYTHONPATH=str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", ""),
                PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                PYTHONHASHSEED="0",
            )
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("OMP_NUM_THREADS", "1")
            # Prefer a Linux-local, user-owned cache on WSL; honor an explicit
            # caller setting. Never include compiled caches in shared evidence.
            env.setdefault(
                "JAX_COMPILATION_CACHE_DIR",
                str(Path.home() / ".cache/codesign-jax" / args.device / dtype),
            )
            if args.log_compiles:
                env["JAX_LOG_COMPILES"] = "1"
            stages = [] if args.skip_tests else [("unit", ["--unit"])]
            stages.append(("linear", ["--driver", "linear"]))
            for kind in dict.fromkeys(args.drivers):
                for method in dict.fromkeys(args.methods):
                    if kind == "spring" and method != "incremental":
                        continue
                    stages.append((kind + "_" + method, ["--driver", kind, "--method", method]))
            for name, options in stages:
                stage = out / dtype / name
                stage.mkdir(parents=True)
                env["CODESIGN_EVIDENCE_DIR"] = str(stage)
                command = [
                    sys.executable,
                    "-u",
                    str(ROOT / "src/envs/soft_walker/diagnostics/performance_worker.py"),
                    "--out",
                    str(stage),
                    "--device",
                    args.device,
                    "--dtype",
                    dtype,
                ] + options
                if name != "unit":
                    command += [
                        "--repeats",
                        str(args.repeats),
                        "--batch-size",
                        str(args.batch_size),
                        "--size",
                        args.size,
                    ]
                    if args.trace:
                        command += ["--trace"]
                start = time.perf_counter()
                row: Report = {"dtype": dtype, "stage": name, "command": command}
                print(f"Starting {dtype}/{name}; log: {stage / 'stdout.log'}", flush=True)
                with (stage / "stdout.log").open("w") as log:
                    process = start_logged_process(
                        command,
                        cwd=ROOT,
                        env=env,
                        log=log,
                        process_group=os.name == "posix",
                    )
                    try:
                        code = process.wait(timeout=args.timeout)
                    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                        kill_process(process, process_group=os.name == "posix")
                        row["error"] = type(error).__name__
                        code = 124 if isinstance(error, subprocess.TimeoutExpired) else 130
                row.update(returncode=code, elapsed_s=time.perf_counter() - start)
                if not (stage / "stage_result.json").exists():
                    write(stage / "stage_result.json", {"completed": False, "passed": False, **row})
                manifest["stages"].append(row)
                write(out / "manifest.json", manifest)
                print(f"Finished {dtype}/{name}: exit={code}", flush=True)
                if code == 130:
                    raise KeyboardInterrupt
        manifest["completed"] = True
    except BaseException as error:
        interrupted = repr(error)
        manifest["error"] = interrupted
    finally:
        changed = source_hashes() != manifest["source_hashes"]
        manifest["source_changed_during_run"] = changed
        manifest["passed"] = bool(
            manifest["completed"]
            and manifest["stages"]
            and not changed
            and all(r["returncode"] == 0 for r in manifest["stages"])
        )
        write(out / "manifest.json", manifest)
        files = [p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"]
        (out / "SHA256SUMS").write_text(
            "".join(
                f"{digest(path)}  {path.relative_to(out).as_posix()}\n" for path in sorted(files)
            )
        )
        archive = out.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    z.write(path, Path(out.name) / path.relative_to(out))
        print(f"Evidence: {archive}; selected stages passed={manifest['passed']}", flush=True)
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
