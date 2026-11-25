#!/usr/bin/env python3
"""Small native/JAX equation-parity and complete-episode derivative probe.

The CPU JAX oracle is deliberately outside the timed native computation.
GPU forward and reverse episodes are each captured as a complete device graph.
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

import numpy as np

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

    from envs.soft_walker.engine.types import (
        Array,
    )
    from envs.soft_walker.engine.warp_native.audit_types import NativeSubmissionReport

# Serialized reports can contain backend-specific auxiliary diagnostic trees.
Report = dict[str, Any]


def main() -> int:
    """Command entry point. Small native/JAX equation-parity and complete-episode derivative probe."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--physics", choices=["spring", "beam", "fem"], required=True)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ["JAX_ENABLE_X64"] = "1" if args.dtype == "float64" else "0"
    start = time.perf_counter()
    status: Report = {
        "success": False,
        "physics": args.physics,
        "dtype": args.dtype,
        "device": args.device,
        "jax_oracle": "cpu",
    }
    try:
        import jax
        import jax.numpy as jnp
        import warp as wp

        from envs.soft_walker.diagnostics.warp.solid_fixtures import fixture, reference_functions
        from envs.soft_walker.engine.warp_native.solid_episode import SolidEpisode
        from envs.soft_walker.engine.warp_native.solid_model import SolidDeviceModel
        from envs.soft_walker.engine.warp_native.solid_step import SolidStep

        device = wp.get_device("cuda:0" if args.device == "cuda" else args.device)
        if args.device.startswith("cuda") and not device.is_cuda:
            raise RuntimeError("CUDA fallback forbidden")
        bundle, jmodel, jstate, jconfig, _, _ = fixture(args.physics, args.dtype, True)
        # Excite bending/shear instead of only checking a rest-state tangent.
        if args.physics == "beam":
            from envs.soft_walker.engine.physics import beam

            angles = np.zeros((len(bundle.fields["mass"]), 3), args.dtype)
            angles[:, 2] = np.linspace(-0.025, 0.04, len(angles))
            rotation = np.asarray(beam.exp_so3(jnp.asarray(angles)))
            cast("dict[str, Array]", bundle.initial)["rotation"] = rotation
            jstate = cast(beam.State, jstate)._replace(rotation=jnp.asarray(rotation))
        # Tighten the primal solve for a derivative ladder, not the simulation defaults.
        # Otherwise a ~1e-8 control derivative is obscured by a 3e-6 root residual.
        if args.dtype == "float32":
            bundle.config = replace(bundle.config, atol32=3e-8)
        status["probe_root_atol"] = (
            bundle.config.atol64 if args.dtype == "float64" else bundle.config.atol32
        )
        owner = SolidDeviceModel(bundle, device)
        step = SolidStep(owner)
        command = np.asarray([0.3, -0.2], args.dtype)
        step.command.assign(command)
        rng = np.random.default_rng(173)
        seeds = {}
        for name in ("u", "v", "rotation", "omega", "actuator", "ledger"):
            array = getattr(step.output, name)
            seed = rng.normal(size=array.numpy().shape).astype(args.dtype)
            if name == "ledger":
                seed = np.asarray([0.03, 0.2, 0.1, 0.0], args.dtype)
            seeds[name] = seed
        seed_arrays = {
            getattr(step.output, k): wp.array(v, dtype=getattr(step.output, k).dtype, device=device)
            for k, v in seeds.items()
        }

        def one_step() -> None:
            step.forward(record_derivatives=True)
            step.backward(seed_arrays)

        compile_start = time.perf_counter()
        if device.is_cuda:
            with wp.ScopedCapture(device=device, force_module_load=False) as cap:
                one_step()
            wp.capture_launch(cap.graph)
        else:
            one_step()
        wp.synchronize_device(device)
        status["one_step_compile_and_first_seconds"] = time.perf_counter() - compile_start
        assert step.code.numpy()[0] == 0, step.diagnostics()
        q = step.q.numpy().ravel()
        native_grad = step.command.grad.numpy().copy()
        residual, outputs = reference_functions(
            args.physics, jmodel, jstate, jconfig, bundle.config.dt
        )
        jq = jnp.asarray(q)
        jc = jnp.asarray(command)
        reference_residual = np.asarray(residual(jq, jc))
        expected = jax.jit(outputs)(jq, jc)
        arrays = {}
        errors = {}
        rtol = 5e-4 if args.dtype == "float32" else 2e-8
        atol = 4e-7 if args.dtype == "float32" else 2e-11
        for name, ref in expected.items():
            actual = getattr(step.output, name).numpy()
            ref = np.asarray(ref)
            np.testing.assert_allclose(actual, ref, rtol=rtol, atol=atol)
            arrays["native_" + name] = actual
            arrays["jax_" + name] = ref
            errors[name] = float(np.max(np.abs(actual - ref), initial=0))
        assert np.max(np.abs(reference_residual)) < (5e-5 if args.dtype == "float32" else 2e-8), (
            reference_residual
        )
        weights = {k: jnp.asarray(v) for k, v in seeds.items()}

        def objective(z: Array, a: Array) -> Array:
            values = outputs(z, a)
            return sum(jnp.sum(values[k] * weights[k]) for k in weights)

        jacobian = jax.jit(jax.jacfwd(residual, argnums=0))(jq, jc)
        control_jac = jax.jit(jax.jacfwd(residual, argnums=1))(jq, jc)
        dq, da = jax.jit(jax.grad(objective, argnums=(0, 1)))(jq, jc)
        multiplier = np.linalg.solve(np.asarray(jacobian).T, np.asarray(dq))
        reference_grad = np.asarray(da) - np.asarray(control_jac).T @ multiplier
        np.testing.assert_allclose(native_grad, reference_grad, rtol=rtol, atol=atol)
        arrays["native_control_grad"] = native_grad
        arrays["jax_control_grad"] = reference_grad
        # True nonsymmetric tangent and transpose, not only a symmetric test.
        direction = rng.normal(size=step.q.numpy().shape).astype(args.dtype)
        x = wp.array(direction, dtype=step.k.qvec, device=device)
        out = wp.zeros_like(x)
        for trans in (False, True):

            def tangent_program() -> None:
                step.refresh_linearization()
                step.apply_tangent(x, out, out, 1.0, 0.0, trans)

            if device.is_cuda:
                with wp.ScopedCapture(device=device, force_module_load=False) as tangent_capture:
                    tangent_program()
                wp.capture_launch(tangent_capture.graph)
            else:
                tangent_program()
            wp.synchronize_device(device)
            product = (
                np.asarray(jacobian).T @ direction.ravel()
                if trans
                else np.asarray(jacobian) @ direction.ravel()
            )
            np.testing.assert_allclose(out.numpy().ravel(), product, rtol=rtol, atol=atol)
        status["one_step_parity"] = errors
        theta = np.asarray([[0.7, 0.4], [-0.4, 0.2]], args.dtype)
        commands = np.tanh(theta)
        episode = SolidEpisode(
            owner,
            commands,
            settling_steps=1,
            checkpoint_stride=2,
            velocity_weight=0.17,
            work_weight=0.09,
            reuse_replay_roots=True,
        )
        run = episode.run(gradient=True)
        assert run["status"] == 0, run
        grad = episode.gradients()
        direction = np.asarray([[0.8, 0.6], [0.4, 0.3]], args.dtype)
        direction /= np.linalg.norm(direction)
        fields = (
            ["stiffness"]
            if args.physics == "spring"
            else ["young"]
            if args.physics == "fem"
            else ["extension", "bending"]
        )
        originals = {name: getattr(owner.model, name).numpy().copy() for name in fields}
        material_direction = 0.37
        ad = float(np.sum(grad["commands"] * direction * (1 - commands**2)))
        ad += sum(
            float(np.sum(grad["model"][name] * value)) * material_direction
            for name, value in originals.items()
        )
        status["episode_ad_before_ladder"] = ad
        status["episode_control_gradient"] = grad["commands"].tolist()
        samples = []
        epsilons = (
            [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025]
            if args.dtype == "float64"
            else [1.28, 0.64, 0.32, 0.16, 0.08, 0.04, 0.02, 0.01]
        )
        for h in epsilons:
            losses = []
            for sign in (-1, 1):
                episode.set_commands(np.tanh(theta + sign * h * direction).astype(args.dtype))
                for name, value in originals.items():
                    getattr(owner.model, name).assign(
                        (value * np.exp(sign * h * material_direction)).astype(args.dtype)
                    )
                result = episode.run()
                assert result["status"] == 0, result
                losses.append(result["loss"])
            finite_difference = (losses[1] - losses[0]) / (2 * h)
            samples.append(
                dict(
                    h=h,
                    minus=losses[0],
                    plus=losses[1],
                    finite_difference=finite_difference,
                    absolute_error=abs(finite_difference - ad),
                    relative_error=abs(finite_difference - ad)
                    / max(abs(finite_difference), abs(ad), 1e-20),
                )
            )
        episode.set_commands(commands)
        for name, value in originals.items():
            getattr(owner.model, name).assign(value)
        _again = episode.run(True)
        again_grad = episode.gradients()
        np.testing.assert_allclose(again_grad["commands"], grad["commands"], rtol=rtol, atol=atol)
        best = min(samples, key=lambda row: row["relative_error"])
        assert best["relative_error"] < (7e-3 if args.dtype == "float32" else 2e-4), samples
        orders = [
            np.log(a["absolute_error"] / b["absolute_error"]) / np.log(a["h"] / b["h"])
            if min(a["absolute_error"], b["absolute_error"]) > 0
            else None
            for a, b in zip(samples, samples[1:])
        ]
        status["episode"] = run
        status["ladder"] = dict(
            ad=ad,
            samples=samples,
            orders=orders,
            observed_second_order=any(v is not None and 1.5 < v < 2.5 for v in orders),
        )
        status["source_scope"] = (
            "resolved fields + command derivatives; no topology derivative claimed"
        )
        arrays["episode_commands_grad"] = grad["commands"]
        arrays["episode_loss"] = np.asarray([run["loss"]], args.dtype)
        arrays["reference_residual"] = reference_residual
        np.savez_compressed(args.out / "parity.npz", **arrays)
        if args.audit:
            if not device.is_cuda:
                raise ValueError("Residency audit requires CUDA")
            from envs.soft_walker.engine.execution.cupti import DriverAudit

            reports = {}
            for with_gradient in (False, True):
                checked = episode.run(with_gradient)
                if checked["status"] != 0 or (with_gradient and checked["adjoint_status"] != 0):
                    raise RuntimeError("Residency warmup failed numerically")
                reference_loss = episode.loss.numpy().copy()
                stream = wp.get_stream(device)
                wp.synchronize_stream(stream)
                with DriverAudit() as observer:
                    wp.capture_launch(
                        episode.graphs["gradient" if with_gradient else "forward"], stream=stream
                    )
                    wp.synchronize_stream(stream)
                report = cast("NativeSubmissionReport", observer.report())
                report["output_matches"] = bool(
                    np.allclose(episode.loss.numpy(), reference_loss, rtol=1e-5, atol=1e-12)
                )
                reports["gradient" if with_gradient else "forward"] = report
                (args.out / "audits.json").write_text(json.dumps(reports, indent=2, default=str))
                if not report["passed"] or not report["output_matches"]:
                    raise RuntimeError("Native whole-episode submission audit failed")
            status["audits"] = reports
        status["diagnostics"] = step.diagnostics()
        status["success"] = True
    except BaseException:
        status["exception"] = traceback.format_exc()
        print(status["exception"], flush=True)
    finally:
        status["seconds"] = time.perf_counter() - start
        (args.out / "result.json").write_text(json.dumps(status, indent=2, allow_nan=False))
        print(json.dumps(status, indent=2, allow_nan=False), flush=True)
    return 0 if status["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
