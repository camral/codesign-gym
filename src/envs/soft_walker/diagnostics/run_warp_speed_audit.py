#!/usr/bin/env python3
"""Archive a focused Warp speed/parity experiment, including failed stages.

No historical suite or optimizer search runs here. Native and optional JAX
workers run sequentially in isolated processes; compilation is excluded from
warm execution samples. For CUDA, submission auditing is enabled unless it is
explicitly disabled for an external profiler (whose result is not a certificate).
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
from typing import TYPE_CHECKING, Any, cast

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
        PathLike,
    )

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def write(path: PathLike, value: Any) -> None:
    """Serialize evidence to the requested path using this command's existing JSON encoder."""
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def sources(root: Path) -> dict[str, str]:
    """Collect source file digests rooted at the supplied repository."""
    files = list((root / "src/envs/soft_walker/engine").rglob("*.py"))
    files += diagnostic_sources(root)
    files += list((root / "src/envs/soft_walker/scripts").glob("*warp*py"))
    files += list((root / "tests/soft_walker").glob("test_warp_native*py"))
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(set(files))
    }


def main() -> int:
    """Command entry point. Archive a focused Warp speed/parity experiment, including failed stages."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument(
        "--dtypes", nargs="+", choices=("float64", "float32"), default=["float64", "float32"]
    )
    p.add_argument("--dims", nargs="+", type=int, choices=(2, 3), default=[2, 3])
    p.add_argument(
        "--variants",
        nargs="+",
        choices=(
            "legacy",
            "adaptive",
            "cached",
            "optimized",
            "dynamic",
            "dynamic_cached",
            "dynamic_optimized",
            "linear_reuse",
            "compact",
            "compact_reuse",
            "block",
            "all",
        ),
        default=["adaptive", "optimized"],
    )
    p.add_argument("--cells", type=int, default=1)
    p.add_argument("--compact-nodes", type=int, default=0)
    p.add_argument("--steps", type=int)
    p.add_argument("--settling", type=int)
    p.add_argument("--substeps", type=int)
    p.add_argument("--checkpoint-stride", type=int, default=16)
    p.add_argument("--bundle", type=Path)
    p.add_argument("--commands", type=Path)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--gradient", action="store_true")
    p.add_argument("--compare-jax", action="store_true")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--skip-benchmark", action="store_true")
    p.add_argument("--skip-ladder", action="store_true")
    p.add_argument(
        "--no-audit",
        action="store_true",
        help="For external CUPTI/Nsight only; no residency certificate",
    )
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    from envs.soft_walker.diagnostics.warp_speed_worker import resolve_protocol

    try:
        a = resolve_protocol(a)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    if (
        min(a.cells, a.steps, a.substeps, a.checkpoint_stride, a.repeats, a.timeout) < 1
        or a.settling < 0
    ):
        p.error("Invalid counts")
    if a.skip_tests and a.skip_benchmark and a.skip_ladder:
        p.error("Empty selection")
    if a.bundle and (len(a.dtypes) != 1 or len(a.dims) != 1):
        p.error("An exported bundle must use exactly one precision/dimension")
    if a.commands and not a.bundle:
        p.error("--commands requires --bundle")
    root = REPO_ROOT
    out = a.out.resolve()
    try:
        require_source_checkout(root)
    except ValueError as exc:
        p.error(str(exc))
    out.mkdir(parents=True, exist_ok=False)
    env = dict(
        os.environ,
        PYTHONPATH=str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    packages: Report = {}
    for name in ("warp-lang", "jax", "jaxlib", "numpy", "scipy", "gymnasium"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    manifest: Report = dict(
        selection=vars(a),
        source_hashes=sources(root),
        packages=packages,
        python=sys.version,
        platform=platform.platform(),
        started=time.time(),
        environment={
            key: env[key]
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "XLA_FLAGS",
                "JAX_COMPILATION_CACHE_DIR",
                "CODESIGN_OFFLINE_SPACES",
            )
            if key in env
        },
        comparison_scope="Same resolved fields, initial state, timestep, tolerance/budget and diagnostic loss; not the locomotion reward",
    )
    write(out / "manifest.json", manifest)
    summary: Report = dict(success=False, stages=[], comparisons=[])

    def run(name: str, command: Array) -> bool:
        start = time.perf_counter()
        row: Report = dict(name=name, command=command)
        write(out / "phase.json", row)
        process = None
        with (out / (name + ".log")).open("w") as f:
            try:
                process = start_logged_process(
                    command,
                    cwd=root,
                    env=env,
                    log=f,
                    process_group=True,
                )
                row["returncode"] = process.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                kill_process(cast(subprocess.Popen[bytes], process))
                row.update(returncode=124, timed_out=True)
            except BaseException:
                if process is not None and process.poll() is None:
                    kill_process(process)
                row.update(returncode=1, exception=traceback.format_exc())
        row["seconds"] = time.perf_counter() - start
        summary["stages"].append(row)
        write(out / "progress.json", summary)
        print(json.dumps(row), flush=True)
        if row.get("timed_out"):
            raise RuntimeError("Timed out; no further device work will be launched")
        return row["returncode"] == 0

    try:
        if shutil.which("nvidia-smi"):
            try:
                q = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, timeout=15)
                (out / "nvidia-smi.txt").write_text(q.stdout + q.stderr)
            except Exception as exc:
                write(out / "nvidia-smi_error.json", dict(error=repr(exc)))
        if not run(
            "device",
            [
                sys.executable,
                "-c",
                f"import warp as w; w.init(); print(w.__version__); print(w.get_device({a.device!r}))",
            ],
        ):
            raise RuntimeError("Requested device unavailable; no fallback")
        for dtype in dict.fromkeys(a.dtypes):
            env.update(
                WARP_TEST_DTYPE=dtype,
                WARP_TEST_DEVICE=a.device,
                JAX_ENABLE_X64="true" if dtype == "float64" else "false",
                JAX_PLATFORMS="cpu",
            )
            if not a.skip_tests:
                tests = [
                    "tests/soft_walker/test_warp_native_speed.py",
                    "tests/soft_walker/test_warp_native_speed_merge.py",
                    "tests/soft_walker/test_warp_native_contact.py",
                    "tests/soft_walker/test_warp_native_krylov.py",
                    "tests/soft_walker/test_warp_native_support.py",
                ]
                if not run(
                    dtype + "_tests",
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "--noconftest",
                        "-q",
                        *tests,
                        "-o",
                        "junit_family=xunit1",
                        "--junitxml=" + str(out / (dtype + "_tests.xml")),
                    ],
                ):
                    continue

            def command(mode: str, dim: int, destination: Path) -> list[str]:
                cmd = [
                    sys.executable,
                    "src/envs/soft_walker/diagnostics/warp_speed_worker.py",
                    "--mode",
                    mode,
                    "--device",
                    a.device,
                    "--dtype",
                    dtype,
                    "--dim",
                    str(dim),
                    "--compact-nodes",
                    str(a.compact_nodes),
                    "--cells",
                    str(a.cells),
                    "--steps",
                    str(a.steps),
                    "--settling",
                    str(a.settling),
                    "--substeps",
                    str(a.substeps),
                    "--checkpoint-stride",
                    str(a.checkpoint_stride),
                    "--repeats",
                    str(a.repeats),
                    "--out",
                    str(destination),
                ]
                if a.bundle:
                    cmd += ["--bundle", str(a.bundle.resolve())]
                if a.commands:
                    cmd += ["--commands", str(a.commands.resolve())]
                if a.gradient:
                    cmd += ["--gradient"]
                return cmd

            if not a.skip_benchmark:
                for dim in dict.fromkeys(a.dims):
                    name = f"{dtype}_d{dim}"
                    native_out = out / (name + "_native")
                    cmd = command("native", dim, native_out) + ["--variants", *a.variants]
                    if a.device == "cuda" and not a.no_audit:
                        cmd += ["--audit"]
                    if not run(name + "_native", cmd):
                        continue
                    if a.compare_jax:
                        jax_out = out / (name + "_jax")
                        if run(name + "_jax", command("jax", dim, jax_out)):
                            import numpy as np

                            from envs.soft_walker.diagnostics.warp.checks import compare_array

                            mode = "value_gradient" if a.gradient else "forward"
                            reference = np.load(jax_out / (mode + ".npz"), allow_pickle=False)
                            for variant in a.variants:
                                arrays = np.load(
                                    native_out / variant / (mode + ".npz"), allow_pickle=False
                                )
                                comparisons = {}
                                for key in reference.files:
                                    comparisons[key] = compare_array(
                                        arrays[key], reference[key], name=key, precision=dtype
                                    )
                                summary["comparisons"].append(
                                    dict(
                                        precision=dtype,
                                        dim=dim,
                                        variant=variant,
                                        arrays=comparisons,
                                    )
                                )
                                write(out / "comparisons.json", summary["comparisons"])
            if not a.skip_ladder:
                # The ladder is an independently specified short fixture. It
                # must not inherit a possibly 3D, long-horizon demo command file.
                ladder_cmd = [
                    sys.executable,
                    "src/envs/soft_walker/diagnostics/warp_speed_worker.py",
                    "--mode",
                    "ladder",
                    "--device",
                    a.device,
                    "--dtype",
                    dtype,
                    "--out",
                    str(out / (dtype + "_ladder")),
                ]
                run(dtype + "_ladder", ladder_cmd)
        summary["success"] = bool(summary["stages"]) and all(
            r["returncode"] == 0 for r in summary["stages"]
        )
    except BaseException:
        summary["exception"] = traceback.format_exc()
        print(summary["exception"], file=sys.stderr, flush=True)
    finally:
        summary["sources_unchanged"] = manifest["source_hashes"] == sources(root)
        summary["success"] = summary["success"] and summary["sources_unchanged"]
        summary["seconds"] = time.time() - manifest["started"]
        write(out / "summary.json", summary)
        entries = {
            str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "SHA256SUMS"
        }
        (out / "SHA256SUMS").write_text("".join(f"{h}  {p}\n" for p, h in entries.items()))
        shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
        print("Archive:", str(out) + ".zip", flush=True)
    return 0 if summary["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
