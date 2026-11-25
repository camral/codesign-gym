#!/usr/bin/env python3
"""Full short MPM VJP/FD check including independent particle-volume derivatives.

CPU JAX float64 is the numerical oracle, never a native runtime dependency.
No broad benchmark sweep or optimizer search is launched by this worker.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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
)

if TYPE_CHECKING:
    from pathlib import Path

    from envs.soft_walker.engine.physics import mpm
    from envs.soft_walker.engine.types import (
        Array,
    )
    from envs.soft_walker.engine.warp_native.audit_types import NativeSubmissionReport
    from envs.soft_walker.engine.warp_native.host_types import EpisodeResult

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def main() -> int:
    """Command entry point. Full short MPM VJP/FD check including independent particle-volume derivatives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=["float32", "float64"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or (args.audit and args.device != "cuda"):
        parser.error("Positive repetitions required; auditing requires CUDA")
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_ENABLE_X64"] = "1"
    status: Report = {
        "success": False,
        "dtype": args.dtype,
        "native_device": args.device,
        "oracle": "CPU JAX float64, including independent volume variables",
    }
    started = time.perf_counter()
    try:
        import jax
        import jax.numpy as jnp
        import numpy as np
        import warp as wp

        from envs.soft_walker.diagnostics.warp.checks import contact_fixture
        from envs.soft_walker.engine.physics import mpm
        from envs.soft_walker.engine.warp_native.episode import Episode
        from envs.soft_walker.engine.warp_native.model import DeviceModel
        from envs.soft_walker.engine.warp_native.mpm_conversion import to_jax

        bundle = contact_fixture(2, args.dtype, "active_strain", "slope")
        bundle.config = replace(
            bundle.config,
            mass_weighted=True,
            krylov_method="adaptive",
            cache_tangent=True,
            parameter_adjoint="mass",
        )
        command = np.asarray([[0.4], [-0.2], [0.15]], args.dtype)
        direction = np.asarray([[0.1], [-0.07], [0.08]], args.dtype)
        owner = DeviceModel(bundle, args.device)
        episode = Episode(
            owner,
            command,
            settling_steps=1,
            checkpoint_stride=2,
            velocity_weight=0.1,
            work_weight=0.05,
            reuse_replay_roots=True,
        )

        def checked(grad: bool = False) -> EpisodeResult:
            report = episode.run(gradient=grad)
            if (
                report["status"] != 0
                or not np.isfinite(report["loss"])
                or (grad and report["adjoint_status"] != 0)
            ):
                raise RuntimeError(f"Invalid execution: {report}")
            return report

        first = checked(True)
        gradient = episode.gradients()
        ng = {name: gradient["model"][name].copy() for name in ("young", "strength", "volume")}
        ng["commands"] = gradient["commands"].copy()
        outputs = {
            k: getattr(episode.final, k).numpy().copy()
            for k in ("x", "v", "F", "C", "actuator", "ledger")
        }
        np.savez_compressed(
            args.out / "native_initial.npz",
            **outputs,
            **{"gradient_" + k: v for k, v in ng.items()},
        )
        status["first_native"] = first
        (args.out / "progress.json").write_text(json.dumps(status, indent=2))
        reference_bundle = contact_fixture(2, "float64", "active_strain", "slope")
        # Copy actual rounded native inputs before casting up; do not compare
        # float32 input rounding to a subtly different float64 fixture.
        reference_bundle.fields = {
            k: (v.astype("float64") if v.dtype.kind == "f" else v.copy())
            for k, v in bundle.fields.items()
        }
        reference_bundle.initial = {
            k: (v.astype("float64") if v.dtype.kind == "f" else v.copy())
            for k, v in cast("dict[str, Array]", bundle.initial).items()
        }
        reference_bundle.terrain = {
            k: (v.astype("float64") if hasattr(v, "dtype") and v.dtype.kind == "f" else v)
            for k, v in cast("dict[str, Array]", bundle.terrain).items()
        }
        reference_bundle.config = bundle.config
        model, initial, grid, cfg = to_jax(reference_bundle)
        p = {k: jnp.asarray(bundle.fields[k], jnp.float64) for k in ("young", "strength", "volume")}
        p["commands"] = jnp.asarray(command, jnp.float64)

        def objective(values: dict[str, Array]) -> tuple[Array, tuple[mpm.State, Array]]:
            cloud = model.cloud._replace(volume=values["volume"])
            mass = model.alpha * model.density * values["volume"]
            m = model._replace(
                cloud=cloud, mass=mass, young=values["young"], strength=values["strength"]
            )
            actions = jnp.concatenate(
                (jnp.zeros((1, bundle.channels), jnp.float64), values["commands"])
            )

            def step(
                carry: tuple[mpm.State, Array], a: Array
            ) -> tuple[tuple[mpm.State, Array], Array]:
                state, travel = carry
                new, diag = mpm.trial(
                    m, state, a, jnp.asarray(bundle.config.dt, jnp.float64), grid, cfg
                )
                return (new, travel + bundle.config.dt * new.v), diag.code

            (final, travel), codes = jax.lax.scan(
                step, (initial, jnp.zeros_like(initial.v)), actions
            )
            loss = (
                jnp.sum(mass * (-travel[:, 0] / 0.2 + 0.1 * jnp.sum(final.v**2, axis=-1)))
                / jnp.sum(mass)
                + 0.05 * final.work_positive
            )
            return loss, (final, codes)

        t = time.perf_counter()
        (loss, (reference, codes)), jg = jax.jit(jax.value_and_grad(objective, has_aux=True))(p)
        jax.block_until_ready(loss)
        status["oracle_compile_and_run_seconds"] = time.perf_counter() - t
        if np.any(np.asarray(codes) != 0):
            raise RuntimeError("Independent JAX root failed")
        arrays = {"native_loss": np.asarray([first["loss"]]), "jax_loss": np.asarray([loss])}
        errors = {}
        for name in ng:
            a = ng[name].astype("float64")
            b = np.asarray(jg[name])
            rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
            limit = 3e-3 if args.dtype == "float32" else 3e-7
            passed = bool(np.linalg.norm(a - b) <= limit * np.linalg.norm(b) + 1e-24)
            errors[name] = {"relative_l2": rel, "passed": passed, "limit": limit}
            arrays["native_gradient_" + name] = ng[name]
            arrays["jax_gradient_" + name] = b
        status["comparisons"] = errors
        np.savez_compressed(args.out / "parity.npz", **arrays)
        if not all(row["passed"] for row in errors.values()):
            raise AssertionError(f"Independent gradient comparison failed: {errors}")
        volume_direction = bundle.fields["volume"] * np.asarray(
            [0.25, -0.18, 0.13, -0.09], args.dtype
        )
        originals = {k: bundle.fields[k].copy() for k in ("young", "strength", "volume")}
        ad = float(
            np.sum(ng["commands"] * direction)
            + np.sum(ng["young"] * originals["young"] * 0.8)
            + np.sum(ng["strength"] * originals["strength"] * 0.1)
            + np.sum(ng["volume"] * volume_direction)
        )

        def value(theta: Array) -> float:
            episode.set_commands(command + theta * direction)
            owner.model.young.assign((originals["young"] * np.exp(0.8 * theta)).astype(args.dtype))
            owner.model.strength.assign(
                (originals["strength"] * (1 + 0.1 * theta)).astype(args.dtype)
            )
            owner.model.volume.assign(
                (originals["volume"] + theta * volume_direction).astype(args.dtype)
            )
            return checked()["loss"]

        rows = []
        for h in 0.4 * 0.5 ** np.arange(9):
            plus, minus = value(float(h)), value(float(-h))
            fd = (plus - minus) / (2 * h)
            err = abs(fd - ad)
            rows.append(
                {
                    "h": float(h),
                    "plus": plus,
                    "minus": minus,
                    "central_difference": float(fd),
                    "absolute_error": float(err),
                    "relative_error": float(err / max(abs(ad), 1e-30)),
                }
            )
            (args.out / "ladder_progress.json").write_text(
                json.dumps({"ad": ad, "rows": rows}, indent=2)
            )
        orders = [
            float(np.log2(a["absolute_error"] / b["absolute_error"]))
            if min(a["absolute_error"], b["absolute_error"]) > 0
            else None
            for a, b in zip(rows, rows[1:])
        ]
        windows = [
            i
            for i in range(len(orders) - 1)
            if all(v is not None and 1.5 <= v <= 2.5 for v in orders[i : i + 2])
        ]
        best = min(r["relative_error"] for r in rows)
        passed = bool(
            best < (3e-5 if args.dtype == "float64" else 3e-3)
            and (windows or args.dtype == "float32")
        )
        ladder = {
            "ad": ad,
            "rows": rows,
            "orders": orders,
            "second_order_windows": windows,
            "best_relative_error": best,
            "passed": passed,
            "float32_plateau_without_window": args.dtype == "float32" and not bool(windows),
        }
        (args.out / "ladder.json").write_text(json.dumps(ladder, indent=2))
        if not passed:
            raise AssertionError(f"Directional ladder failed: {ladder}")
        for k, v in originals.items():
            getattr(owner.model, k).assign(v)
        episode.set_commands(command)
        _repeated = checked(True)
        for k in ng:
            g = (
                episode.gradients()["commands"]
                if k == "commands"
                else episode.gradients()["model"][k]
            )
            np.testing.assert_allclose(
                g, ng[k], rtol=3e-5 if args.dtype == "float32" else 1e-10, atol=1e-12
            )
        timings = {}
        for grad in (False, True):
            checked(grad)
            timings["gradient" if grad else "forward"] = [
                checked(grad) for _ in range(args.repeats)
            ]
        status.update(ladder=ladder, timings=timings, diagnostics=episode.step.diagnostics())
        if args.audit:
            from envs.soft_walker.engine.execution.cupti import DriverAudit

            audits = {}
            for grad in (False, True):
                checked(grad)
                expected = episode.loss.numpy().copy()
                expected_gradient = episode.gradients()["commands"].copy() if grad else None
                stream = wp.get_stream(owner.device)
                wp.synchronize_stream(stream)
                with DriverAudit() as observer:
                    wp.capture_launch(
                        episode.graphs["gradient" if grad else "forward"], stream=stream
                    )
                    wp.synchronize_stream(stream)
                report = cast("NativeSubmissionReport", observer.report())
                report["numerical_match"] = bool(
                    np.allclose(episode.loss.numpy(), expected, rtol=2e-5, atol=1e-12)
                )
                if grad:
                    report["numerical_match"] &= bool(
                        np.allclose(
                            episode.gradients()["commands"],
                            expected_gradient,
                            rtol=3e-4,
                            atol=1e-10,
                        )
                    )
                audits["gradient" if grad else "forward"] = report
                (args.out / "audits.json").write_text(json.dumps(audits, indent=2, default=str))
                if not report["passed"] or not report["numerical_match"]:
                    raise RuntimeError("Whole-execution audit rejected the invocation")
            status["audits"] = audits
        status["success"] = True
    except BaseException:
        status["exception"] = traceback.format_exc()
        print(status["exception"], flush=True)
    finally:
        status["seconds"] = time.perf_counter() - started
        (args.out / "result.json").write_text(json.dumps(status, indent=2, default=str))
        print(json.dumps(status, indent=2, default=str), flush=True)
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
