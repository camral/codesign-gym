#!/usr/bin/env python3
"""Run new Warp checks in isolated processes and always archive the evidence.

This never selects the historical JAX suite. JAX is a CPU oracle in the explicit
parity stages, not an implementation dependency of the native solver.
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
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    diagnostic_sources,
    require_source_checkout,
)

if TYPE_CHECKING:
    from pathlib import Path

    from envs.soft_walker.engine.types import (
        Array,
    )

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def hashes(root: Path) -> dict[str, str]:
    """Collect source file digests for the evidence manifest."""
    paths = sorted((root / "src/envs/soft_walker/engine").rglob("*.py"))
    paths += diagnostic_sources(root)
    paths += sorted(p for p in (root / "src/envs/soft_walker/scripts").glob("*warp_native*.py"))
    paths += sorted((root / "tests/soft_walker").glob("test_warp_native*.py"))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main() -> int:
    """Command entry point. Run new Warp checks in isolated processes and always archive the evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--dtypes", nargs="+", choices=("float64", "float32"), default=["float64", "float32"]
    )
    parser.add_argument("--drivers", nargs="+", choices=("mpm2", "mpm3"), default=["mpm2", "mpm3"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout < 1:
        parser.error("repeats and timeout must be positive")
    if args.audit and args.device != "cuda":
        parser.error("--audit requires CUDA")
    if args.skip_tests and args.skip_parity and not args.audit:
        parser.error("Empty validation selection")
    root = REPO_ROOT
    out = args.out.resolve()
    try:
        require_source_checkout(root)
    except ValueError as exc:
        parser.error(str(exc))
    out.mkdir(parents=True, exist_ok=False)
    packages: Report = {}
    for name in ("warp-lang", "jax", "jaxlib", "numpy", "scipy", "gymnasium"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest: Report = dict(
        schema=1,
        selection=vars(args) | {"out": str(out)},
        source_hashes=hashes(root),
        python=sys.version,
        platform=platform.platform(),
        packages=packages,
        environment={
            key: os.environ[key]
            for key in ("CUDA_VISIBLE_DEVICES", "XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR")
            if key in os.environ
        },
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary: Report = dict(success=False, stages=[])

    def run(name: str, command: Array, env: dict[str, str]) -> bool:
        log = out / f"{name}.log"
        start = time.perf_counter()
        row: Report = dict(name=name, command=command)
        with log.open("w") as stream:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            process = None
            try:
                process = start_logged_process(
                    command,
                    cwd=root,
                    env=env,
                    log=stream,
                    process_group=True,
                )
                row["returncode"] = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                if process is not None:
                    kill_process(process)
                row.update(returncode=124, timed_out=True)
            except BaseException as error:
                if process is not None and process.poll() is None:
                    kill_process(process)
                row.update(
                    returncode=130 if isinstance(error, KeyboardInterrupt) else 1,
                    interrupted=isinstance(error, KeyboardInterrupt),
                    exception=traceback.format_exc(),
                )
        row["seconds"] = time.perf_counter() - start
        summary["stages"].append(row)
        (out / "progress.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(row), flush=True)
        if row.get("timed_out") or row.get("interrupted"):
            raise RuntimeError(
                "Timed out or interrupted; skipping later device work to avoid compounding a possible stalled GPU"
            )
        return row["returncode"] == 0

    try:
        # Fail early on device unavailability, before compiling unrelated oracles.
        env = dict(
            os.environ,
            PYTHONPATH=str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
        )
        probe = (
            "import warp as w; w.init(); d=w.get_device(%r); print(d); print(w.__version__)"
            % args.device
        )
        device_ok = run("device", [sys.executable, "-c", probe], env)
        if device_ok:
            for dtype in dict.fromkeys(args.dtypes):
                env.update(
                    WARP_TEST_DTYPE=dtype,
                    WARP_TEST_DEVICE=args.device,
                    JAX_PLATFORMS="cpu",
                    JAX_ENABLE_X64="true" if dtype == "float64" else "false",
                )
                if args.audit:
                    name = dtype + "_audit"
                    accepted = run(
                        name,
                        [
                            sys.executable,
                            "src/envs/soft_walker/diagnostics/warp_native_worker.py",
                            "--mode",
                            "audit",
                            "--device",
                            args.device,
                            "--dtype",
                            dtype,
                            "--out",
                            str(out / name),
                        ],
                        env,
                    )
                    if not accepted:
                        break  # Do not run larger physical tests after a failed residency gate.
                if not args.skip_tests:
                    tests = [
                        str(p.relative_to(root))
                        for p in sorted((root / "tests/soft_walker").glob("test_warp_native*.py"))
                    ]
                    run(
                        dtype + "_units",
                        [
                            sys.executable,
                            "-m",
                            "pytest",
                            "--noconftest",
                            "-q",
                            *tests,
                            "--junitxml=" + str(out / f"{dtype}_junit.xml"),
                        ],
                        env,
                    )
                if not args.skip_parity:
                    for driver in dict.fromkeys(args.drivers):
                        for law in ("active_strain", "active_stress"):
                            name = f"{dtype}_{driver}_{law}"
                            run(
                                name,
                                [
                                    sys.executable,
                                    "src/envs/soft_walker/diagnostics/warp_native_worker.py",
                                    "--mode",
                                    "parity",
                                    "--device",
                                    args.device,
                                    "--dtype",
                                    dtype,
                                    "--dim",
                                    driver[-1],
                                    "--law",
                                    law,
                                    "--repeats",
                                    str(args.repeats),
                                    "--out",
                                    str(out / name),
                                ],
                                env,
                            )
                    name = dtype + "_ladder"
                    run(
                        name,
                        [
                            sys.executable,
                            "src/envs/soft_walker/diagnostics/warp_native_worker.py",
                            "--mode",
                            "ladder",
                            "--device",
                            args.device,
                            "--dtype",
                            dtype,
                            "--out",
                            str(out / name),
                        ],
                        env,
                    )
        summary["source_unchanged"] = manifest["source_hashes"] == hashes(root)
        summary["success"] = (
            bool(summary["stages"])
            and all(r["returncode"] == 0 for r in summary["stages"])
            and summary["source_unchanged"]
        )
    except BaseException:
        summary["exception"] = traceback.format_exc()
        print(summary["exception"], file=sys.stderr)
    finally:
        summary["source_unchanged"] = manifest["source_hashes"] == hashes(root)
        if not summary["source_unchanged"]:
            summary["success"] = False
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        checks = {
            str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*"))
            if p.is_file()
        }
        (out / "SHA256SUMS").write_text("".join(f"{v}  {k}\n" for k, v in checks.items()))
        shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
        print("Evidence archive:", str(out) + ".zip", flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
