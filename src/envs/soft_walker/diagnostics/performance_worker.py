"""One isolated precision/device stage, with evidence even on initialization failure."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Direct-file execution must select this checkout before package imports.
    _source_root = str(Path(__file__).resolve().parents[3])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)

from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
    require_source_checkout,
)

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def write(path: Path, value: Any) -> None:
    """Serialize evidence to the requested path using this command's existing JSON encoder."""
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def main() -> int:
    """Command entry point. One isolated precision/device stage, with evidence even on initialization failure."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--dtype", choices=["float64", "float32"], required=True)
    p.add_argument("--unit", action="store_true")
    args, rest = p.parse_known_args()
    if args.unit:
        try:
            require_source_checkout(REPO_ROOT)
        except ValueError as exc:
            p.error(str(exc))
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    status: Report = {"completed": False, "passed": False, "command": sys.argv}
    try:
        import jax

        jax.config.update("jax_default_matmul_precision", "highest")
        device = jax.devices("gpu" if args.device == "cuda" else "cpu")[0]
        if jax.default_backend() != device.platform:
            raise RuntimeError("Requested backend is not the default; refusing fallback")
        import envs.soft_walker.engine.spaces as spaces

        packages: Report = {}
        for name in (
            "jax",
            "jaxlib",
            "numpy",
            "scipy",
            "gymnasium",
            "diffrax",
            "optimistix",
            "equinox",
            "lineax",
        ):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        env_keys = (
            "JAX_PLATFORMS",
            "JAX_ENABLE_X64",
            "JAX_COMPILATION_CACHE_DIR",
            "XLA_FLAGS",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "CUDA_VISIBLE_DEVICES",
            "CODESIGN_OFFLINE_SPACES",
            "JAX_LOG_COMPILES",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "JAX_TRANSFER_GUARD",
            "JAX_TRANSFER_GUARD_DEVICE_TO_HOST",
        )
        write(
            args.out / "environment.json",
            {
                "python": sys.version,
                "platform": platform.platform(),
                "packages": packages,
                "device": str(device),
                "device_kind": device.device_kind,
                "backend": jax.default_backend(),
                "x64": jax.config.jax_enable_x64,
                "gymnasium_real": spaces.USING_GYMNASIUM,
                "environment": {key: os.environ.get(key) for key in env_keys},
                "affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
            },
        )
        if args.device == "cuda":
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
                (args.out / "nvidia_smi.txt").write_text(result.stdout + result.stderr)
            except Exception as error:
                write(args.out / "nvidia_smi_error.json", {"error": repr(error)})
        if args.unit:
            import pytest

            tests = [
                "tests/soft_walker/test_performance.py",
                # Existing regression tied directly to the changed vector API.
                "tests/soft_walker/test_m3_drivers.py::test_m3_vector_scalar_autoreset",
            ]
            status["selected_tests"] = tests
            code = int(pytest.main(tests + ["-q", "--junitxml=" + str(args.out / "junit.xml")]))
        else:
            from envs.soft_walker.diagnostics.profile_performance import main as profile_main

            sys.argv = [
                "profile_performance.py",
                "--out",
                str(args.out),
                "--dtype",
                args.dtype,
            ] + rest
            profile_main()
            code = 0
        status.update(completed=True, passed=code == 0, exitcode=code)
    except BaseException as error:
        traceback.print_exc()
        status.update(completed=True, passed=False, exitcode=1, error=repr(error))
        code = 1
    finally:
        status["elapsed_s"] = time.perf_counter() - start
        write(args.out / "stage_result.json", status)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
