"""One precision/device/stage per process, preserving crashes and diagnostics."""

from __future__ import annotations

import argparse
import importlib.metadata as md
import os
import platform
import resource
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

ROOT = REPO_ROOT
sys.path.insert(0, str(ROOT / "src"))

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def main() -> int:
    """Command entry point. One precision/device/stage per process, preserving crashes and diagnostics."""
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], required=True)
    p.add_argument("--dtype", required=True)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--dump-ir", action="store_true")
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    if a.stage in ("unit", "m2_unit", "m3_unit", "affected"):
        try:
            require_source_checkout(ROOT)
        except ValueError as exc:
            p.error(str(exc))
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["CODESIGN_EVIDENCE_DIR"] = str(out)
    from envs.soft_walker.shared import evidence_record as record, evidence_write_json as write_json

    versions = {x.metadata.get("Name", "unknown"): x.version for x in md.distributions()}
    allow = {
        "JAX_PLATFORMS",
        "JAX_ENABLE_X64",
        "JAX_DEFAULT_MATMUL_PRECISION",
        "JAX_DISABLE_JIT",
        "JAX_LOG_COMPILES",
        "JAX_TRACEBACK_FILTERING",
        "JAX_ENABLE_COMPILATION_CACHE",
        "JAX_COMPILATION_CACHE_DIR",
        "JAX_DEBUG_NANS",
        "JAX_DEBUG_INFS",
        "XLA_FLAGS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "CODESIGN_OFFLINE_SPACES",
        "CODESIGN_TEST_DTYPE",
        "CODESIGN_EVIDENCE_DIR",
        "CODESIGN_UNIT_FILTER",
        "CODESIGN_CASE_FILTER",
        "JAX_DEBUG_LOG_MODULES",
        "JAX_EXPLAIN_CACHE_MISSES",
        "OMP_NUM_THREADS",
        "OMP_PROC_BIND",
        "OMP_PLACES",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "LD_LIBRARY_PATH",
        "PYTHONHASHSEED",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    }
    metadata: Report = {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": dict(sorted(versions.items())),
        "command": sys.argv,
        "environment": {k: v for k, v in os.environ.items() if k in allow},
        "stage": a.stage,
        "requested_device": a.device,
        "requested_dtype": a.dtype,
    }
    if Path("/proc/cpuinfo").exists():
        metadata["cpu_model"] = next(
            (
                x.split(":", 1)[1].strip()
                for x in Path("/proc/cpuinfo").read_text().splitlines()
                if x.startswith("model name")
            ),
            "unknown",
        )
    if hasattr(os, "sched_getaffinity"):
        metadata["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    metadata["cgroup_limits"] = {
        str(p): p.read_text().strip()
        for p in (Path("/sys/fs/cgroup/cpu.max"), Path("/sys/fs/cgroup/memory.max"))
        if p.exists()
    }
    write_json(out / "environment.json", metadata)
    record("stage_start", stage=a.stage, dtype=a.dtype, device=a.device)
    t = time.perf_counter()
    try:
        import jax

        jax.config.update("jax_default_matmul_precision", "highest")
        devices = jax.devices("gpu" if a.device == "cuda" else "cpu")
        if not devices or (a.device == "cuda" and devices[0].platform != "gpu"):
            raise RuntimeError("Requested CUDA device was not obtained; CPU fallback is forbidden")
        if a.dtype == "float64" and not jax.config.x64_enabled:
            raise RuntimeError("float64 silently disabled")
        from envs.soft_walker.engine.spaces import USING_GYMNASIUM

        metadata.update(
            {
                "jax_backend": jax.default_backend(),
                "jax_x64": jax.config.x64_enabled,
                "gymnasium_real": USING_GYMNASIUM,
                "jax_matmul_precision": str(jax.config.jax_default_matmul_precision),
                "devices": [
                    {
                        "id": d.id,
                        "platform": d.platform,
                        "kind": d.device_kind,
                        "description": str(d),
                    }
                    for d in devices
                ],
            }
        )
        try:
            metadata["device_memory_stats_before"] = devices[0].memory_stats()
        except Exception as exc:
            metadata["memory_stats_unavailable"] = repr(exc)
        write_json(out / "environment.json", metadata)
        record(
            "preflight_pass",
            **{k: metadata[k] for k in ["jax_backend", "jax_x64", "gymnasium_real", "devices"]},
        )
        if a.stage in ("unit", "m2_unit", "m3_unit", "affected"):
            import pytest

            discovered = sorted((ROOT / "tests/soft_walker").glob("test_*.py"))
            if a.stage == "affected":
                from envs.soft_walker.diagnostics.validation_selection import AFFECTED_TESTS

                files = AFFECTED_TESTS
            elif a.stage == "unit":
                files = [
                    str(p) for p in discovered if not p.name.startswith(("test_m2", "test_m3"))
                ]
            else:
                files = [str(p) for p in discovered if p.name.startswith("test_" + a.stage[:2])]
            write_json(
                out / "test_selection.json",
                {"files_or_nodes": files, "filter": os.environ.get("CODESIGN_UNIT_FILTER", "")},
            )
            if os.environ.get("CODESIGN_UNIT_FILTER"):
                files = files + ["-k", os.environ["CODESIGN_UNIT_FILTER"]]
            rc = pytest.main(
                files + ["-v", "--tb=long", "--durations=0", f"--junitxml={out / 'junit.xml'}"]
            )
        elif a.stage == "numerics":
            from envs.soft_walker.diagnostics.audit_numerics import audit_main

            rc = audit_main(out, a.dtype)
        elif a.stage == "benchmark":
            from envs.soft_walker.diagnostics.benchmark import benchmark_main

            rc = benchmark_main(out, a.dtype, a.repeats, a.dump_ir, a.profile)
        elif a.stage == "m2_numerics":
            from envs.soft_walker.diagnostics.audit_m2 import audit_main

            rc = audit_main(out, a.dtype)
        elif a.stage == "m2_mechanics":
            from envs.soft_walker.diagnostics.mechanics_m2 import mechanics_main

            rc = mechanics_main(out, a.dtype)
        elif a.stage == "m2_benchmark":
            from envs.soft_walker.diagnostics.benchmark_m2 import benchmark_main

            rc = benchmark_main(out, a.dtype, a.repeats, a.dump_ir, a.profile)
        elif a.stage == "m3_numerics":
            from envs.soft_walker.diagnostics.audit_m3 import audit_main

            rc = audit_main(out, a.dtype)
        elif a.stage == "m3_benchmark":
            from envs.soft_walker.diagnostics.benchmark_m3 import benchmark_main

            rc = benchmark_main(out, a.dtype, a.repeats, a.dump_ir, a.profile)
        else:
            raise ValueError("unknown stage")
        stats = {
            "exitcode": int(rc),
            "wall_s": time.perf_counter() - t,
            "host_ru_maxrss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        try:
            stats["device_memory_stats_after"] = devices[0].memory_stats()
        except Exception:
            pass
        write_json(out / "stage_result.json", stats)
        record("stage_complete", **stats)
        return int(rc)
    except BaseException as exc:
        traceback.print_exc()
        error = {
            "exitcode": 1,
            "exception": repr(exc),
            "traceback": traceback.format_exc(),
            "wall_s": time.perf_counter() - t,
        }
        write_json(out / "stage_result.json", error)
        record("stage_exception", **error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
