"""M3 changed-path finite-difference ladders with explicit order/plateau gates."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from envs.soft_walker.diagnostics.m3_cases import diagnostic_value, driver_case, spring_case
from envs.soft_walker.shared import (
    ENGINE_ROOT as ENGINE_ROOT,
    REPO_ROOT as REPO_ROOT,
    SCRIPTS_ROOT as SCRIPTS_ROOT,
    SOFT_ROOT as SOFT_ROOT,
    SRC_ROOT as SRC_ROOT,
    TESTS_ROOT as TESTS_ROOT,
    DifferenceReport,
    evidence_record as record,
    evidence_write_json as write_json,
    finite_difference_ladder,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike, DTypeLike

    from envs.soft_walker.engine.types import (
        Array,
        PathLike,
    )

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def audit_main(out: PathLike, dtype: DTypeLike) -> int:
    """Run this precision-specific numerical audit, write its evidence, and return a failure exit code."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(dtype)
    is64 = dtype == np.dtype("float64")
    reports: list[DifferenceReport] = []
    failures: list[str] = []
    arrays = {}
    configs = {}
    conditioning = {}

    def run(
        name: str,
        roll: Callable[..., Any],
        theta: Array,
        directions: Sequence[tuple[str, ArrayLike]],
        kind: str,
    ) -> Report:
        def with_aux(theta: Array) -> tuple[Array, Report]:
            r = roll(theta)
            state = r["state"] if kind == "spring" else r["state"].body
            value = diagnostic_value(state, kind)
            return jnp.where(r["metrics"]["valid"], value, jnp.nan), r

        # Reuse one executable for baseline, FD primal values, and saved state.
        # FD still uses only f(theta+h*d)-f(theta-h*d); extra gradients returned
        # at perturbed points are ignored. No AD derivative enters the quotient.
        vg = jax.jit(jax.value_and_grad(with_aux, has_aux=True))
        theta = jnp.asarray(theta, dtype)
        value = lambda theta: vg(theta)[0][0]
        t = time.perf_counter()
        (val, result), grad = jax.block_until_ready(vg(theta))
        compile_s = time.perf_counter() - t
        arrays[name + "_gradient"] = np.asarray(grad)
        arrays[name + "_value"] = np.asarray(val)
        arrays[name + "_theta"] = np.asarray(theta)
        record("m3_gradient", name=name, gradient=grad, first_compile_and_execute_s=compile_s)
        h = 0.64 / 2.0 ** np.arange(15 if is64 else 11)
        for label, d in directions:
            direction = jnp.asarray(d, dtype)
            ad = float(jnp.vdot(grad, direction))
            ladder = finite_difference_ladder(
                value, theta, direction, steps=h, autodiff=ad, name=name + "/" + label
            )
            order = [
                r["observed_order"] is not None and 1.55 < r["observed_order"] < 2.45
                for r in ladder["rows"]
            ]
            window = any(a and b for a, b in zip(order, order[1:])) if is64 else any(order)
            threshold = 3e-5 if is64 else 1e-3
            accurate = [
                np.isfinite(r["relative_error"]) and r["relative_error"] < threshold
                for r in ladder["rows"]
            ]
            plateau = not is64 and any(all(accurate[i : i + 3]) for i in range(len(accurate) - 2))
            passed = bool(
                np.isfinite(ad)
                and abs(ad) > 1e-10
                and all(np.isfinite(r["central"]) for r in ladder["rows"])
                and ladder["best_relative_error"] < threshold
                and (window or plateau)
            )
            ladder.update(
                {
                    "passed": passed,
                    "relative_threshold": threshold,
                    "absolute_threshold": 0.0,
                    "second_order_window": window,
                    "order_required": True,
                    "roundoff_plateau": bool(plateau and not window),
                    "cross_precision_required": bool(plateau and not window),
                }
            )
            reports.append(ladder)
            if not passed:
                failures.append(ladder["name"])
            print(
                f"{ladder['name']}: AD={ad:.6g} rel={ladder['best_relative_error']:.3e} order={window} plateau={plateau and not window} passed={passed}",
                flush=True,
            )
            write_json(
                out / "gradient_ladders.json",
                {"reports": reports, "failures": failures, "dtype": str(dtype)},
            )
        return result

    import fnmatch
    import os

    selected = [
        k
        for k in ["spring", "fem", "mpm2", "mpm3", "beam"]
        if fnmatch.fnmatchcase(k, os.environ.get("CODESIGN_CASE_FILTER", "*"))
    ]
    if not selected:
        raise ValueError("No M3 numerical cases match the filter")
    for kind in selected:
        fn, roll, theta, meta, env, _ = (
            spring_case(dtype)
            if kind == "spring"
            else driver_case(kind, dtype, steps=12 if kind == "beam" else 3)
        )
        directions: list[tuple[str, ArrayLike]] = [
            (n, np.eye(len(theta))[i]) for i, n in enumerate(meta["parameters"])
        ]
        directions.append(
            (
                "mixed",
                [0.3, -0.2, 0.4, -0.25, 0.65, -0.45]
                if kind == "spring"
                else [0.3, -0.2, 0.4, -0.25, 0.15, 0.65, -0.45],
            )
        )
        r = run(kind + "_driver", roll, theta, directions, kind)
        s = r["state"] if kind == "spring" else r["state"].body
        arrays[kind + "_velocity"] = np.asarray(s.v)
        arrays[kind + "_observation"] = np.asarray(r["observations"][-1])
        arrays[kind + "_objective"] = np.asarray(r["metrics"]["objective"])
        if kind == "beam":
            arrays["beam_rotation"] = np.asarray(s.rotation)
            arrays["beam_omega"] = np.asarray(s.omega)
        if kind.startswith("mpm"):
            arrays[kind + "_F"] = np.asarray(s.F)
            arrays[kind + "_C"] = np.asarray(s.C)
        configs[kind] = meta
        write_json(
            out / (kind + "_rollout.json"),
            {
                "config": meta,
                "state": s._asdict(),
                "metrics": r["metrics"],
                "diagnostics": r["diagnostics"],
                "completed": r.get("completed", r["metrics"]["valid"]),
            },
        )
        if not bool(r["metrics"]["valid"]):
            failures.append(kind + "/invalid_rollout")
        g = np.abs(arrays[kind + "_driver_gradient"])
        nz = g[g > 1e-12]
        target = np.median(nz) if nz.size else 0.0
        conditioning[kind] = {
            "parameters": meta["parameters"],
            "absolute_gradient": g,
            "max_to_min_nonzero": float(nz.max() / nz.min()) if nz.size else None,
            "suggested_coordinate_multiplier": np.clip(target / np.maximum(g, 1e-12), 0.1, 10.0),
            "automatic_rescaling": False,
        }
        env.close()
        import gc

        jax.clear_caches()
        gc.collect()
    np.savez_compressed(out / "reference_arrays.npz", **arrays)
    write_json(out / "case_configs.json", configs)
    write_json(out / "gradient_conditioning.json", conditioning)
    summary: Report = {
        "passed": not failures,
        "failures": failures,
        "dtype": str(dtype),
        "gradient_ladders": len(reports),
        "roundoff_limited_cases": [r["name"] for r in reports if r["roundoff_plateau"]],
        "scope": "new complete driver paths, including beam and MPM3D; fixed topology, smooth branches only",
        "full_regression": False,
        "selected_cases": selected,
    }
    write_json(out / "summary.json", summary)
    return int(bool(failures))
