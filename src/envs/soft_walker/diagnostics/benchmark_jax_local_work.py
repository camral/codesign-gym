"""Compare optional JAX local-work optimizations on a fixed biped MPM problem.

Run with ``python -m envs.soft_walker.diagnostics.benchmark_jax_local_work``.
All modes use identical physics, commands, timestep, and acceptance tolerances.
Compilation is reported separately; warm forward and value-and-gradient times
include a device synchronization. This bounded benchmark is not a walking test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np

if __package__ in (None, ""):
    # Direct-file execution must select this checkout before package imports.
    _source_root = str(Path(__file__).resolve().parents[3])
    if _source_root not in sys.path:
        sys.path.insert(0, _source_root)

from envs.soft_walker.engine.numerics.solvers import NewtonConfig
from envs.soft_walker.engine.physics import mpm
from envs.soft_walker.engine.physics.common import DynamicsConfig
from envs.soft_walker.engine.physics.contact import PlaneContact
from envs.soft_walker.presets.locomotion import LeggedDecoder


class TimingRecord(TypedDict):
    """Synchronized wall times in seconds, with compilation kept separate."""

    compile_and_first_seconds: float
    warm_seconds: list[float]
    warm_median_seconds: float


def _timings(function: Callable[[], object], repeats: int) -> TimingRecord:
    """Measure synchronized calls after compilation and a separate unrecorded warmup."""
    start = time.perf_counter()
    jax.block_until_ready(function())
    compile_and_first = time.perf_counter() - start
    jax.block_until_ready(function())
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(function())
        samples.append(time.perf_counter() - start)
    return {
        "compile_and_first_seconds": compile_and_first,
        "warm_seconds": samples,
        "warm_median_seconds": float(np.median(samples)),
    }


def main() -> None:
    """Construct one fixed physical fixture and write comparable timing/correctness evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "full_grid",
            "compact",
            "compact_tangent",
            "compact_tangent_reuse",
            "blocks_predictor",
            "blocks_newton",
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.steps < 1 or arguments.repeats < 1:
        parser.error("steps and repeats must be positive")
    jax.config.update("jax_enable_x64", arguments.dtype == "float64")
    devices = jax.devices("gpu" if arguments.device == "cuda" else "cpu")
    with jax.default_device(devices[0]):
        dtype = np.dtype(arguments.dtype)
        cloud = mpm.uniform_cloud(
            cell_shape=(6, 6), spacing=0.025, origin=(0.025, 0.001), dtype=dtype
        )
        decoder = LeggedDecoder(2, dtype)
        embedding = jax.tree.map(lambda x: jnp.asarray(x, dtype), decoder.base())
        model = mpm.instantiate(cloud, decoder.decode(embedding, cloud.points), channels=4)
        state = mpm.initial_state(model)
        state = state._replace(F=state.F.at[:, 0, 1].set(0.03).at[:, 1, 1].set(0.985))
        grid = mpm.Grid(shape=(48, 32), spacing=0.025, origin=(-0.15, -0.15))
        config = DynamicsConfig(
            gravity=9.81,
            actuator_tau=0.0,
            actuation_model="active_stress",
            mpm_mass_weighted=True,
            max_halvings=0,
            contact=PlaneContact(enabled=True, stiffness=1e6, friction=0.8),
            newton=NewtonConfig(
                gmres_method="fused",
                preconditioner_side="right",
                restart=40,
                max_restarts=20,
                linear_refinements=2,
            ),
        )
        capacity = 128
        modes = {
            "full_grid": config,
            "compact": replace(config, mpm_compact_nodes=capacity),
            "compact_tangent": replace(config, mpm_compact_nodes=capacity, mpm_cache_tangent=True),
            "compact_tangent_reuse": replace(
                config,
                mpm_compact_nodes=capacity,
                mpm_cache_tangent=True,
                newton=replace(config.newton, linear_work_reuse=True),
            ),
            "blocks_predictor": replace(
                config,
                mpm_compact_nodes=capacity,
                mpm_cache_tangent=True,
                mpm_block_preconditioner=True,
                newton=replace(config.newton, linear_work_reuse=True),
            ),
            "blocks_newton": replace(
                config,
                mpm_compact_nodes=capacity,
                mpm_cache_tangent=True,
                mpm_block_preconditioner=True,
                mpm_preconditioner_update="newton",
                newton=replace(config.newton, linear_work_reuse=True),
            ),
        }
        if arguments.modes:
            modes = {name: modes[name] for name in arguments.modes}
        parameters = jnp.asarray([0.35, -0.30, 0.15, -0.25, 0.0, 0.0], dtype)
        reference_state = None
        reference_gradient = None
        records: dict[str, object] = {}
        report: dict[str, object] = {
            "device": str(devices[0]),
            "dtype": arguments.dtype,
            "jax_version": jax.__version__,
            "fixture": "LeggedDecoder biped, deformed contact-loaded start",
            "particles": len(model.mass),
            "active_particles": int(jnp.sum(model.active)),
            "grid": asdict(grid),
            "steps": arguments.steps,
            "dt_seconds": 0.001,
            "repeats": arguments.repeats,
            "reference_parameters": np.asarray(parameters).tolist(),
            "reference_mode": next(iter(modes)),
            "state_and_gradient_agreement": {
                "rtol": 3e-5 if dtype.name == "float32" else 2e-8,
                "atol": 2e-7 if dtype.name == "float32" else 2e-10,
            },
            "modes": records,
        }
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        for name, numerical in modes.items():

            def rollout(p: jax.Array) -> tuple[mpm.State, jax.Array]:
                """Advance identical dynamics while collecting acceptance and Newton-work counts."""
                body = model._replace(
                    young=model.young * jnp.exp(p[4]),
                    mass=model.mass * jnp.exp(p[5]),
                    density=model.density * jnp.exp(p[5]),
                )

                def step(current: mpm.State, _: None) -> tuple[mpm.State, jax.Array]:
                    """Run a complete resident trial without changing solver acceptance."""
                    updated, information = mpm.trial(
                        body, current, p[:4], jnp.asarray(0.001, dtype), grid, numerical
                    )
                    return updated, jnp.stack(
                        (
                            information.accepted,
                            information.nonlinear_iterations,
                            information.linear_solves,
                            information.max_linear_residual,
                            information.code,
                        )
                    )

                return jax.lax.scan(step, state, None, length=arguments.steps)

            def loss(p: jax.Array) -> jax.Array:
                """Differentiate commands and physical materials through the matched rollout."""
                out, information = rollout(p)
                value = 100 * jnp.mean(out.x[:, 0]) + 0.1 * jnp.sum(out.v**2) + out.work_positive
                return jnp.where(jnp.all(information[:, 0] == 1), value, jnp.nan)

            forward = jax.jit(rollout)
            gradient = jax.jit(jax.value_and_grad(loss))
            forward_timing = _timings(lambda: forward(parameters), arguments.repeats)
            gradient_timing = _timings(lambda: gradient(parameters), arguments.repeats)
            out, info = jax.device_get(forward(parameters))
            value, derivative = jax.device_get(gradient(parameters))
            if (
                not np.all(info[:, 0] == 1)
                or not np.isfinite(value)
                or not np.isfinite(derivative).all()
            ):
                raise RuntimeError(
                    f"{name}: physical or implicit-gradient acceptance failed: {info}"
                )
            flattened = np.concatenate([np.asarray(x).reshape(-1) for x in out])
            if not np.isfinite(flattened).all():
                raise RuntimeError(f"{name}: nonfinite physical state")
            if reference_state is None:
                reference_state, reference_gradient = flattened, derivative
            tolerance = 3e-5 if dtype.name == "float32" else 2e-8
            absolute = 2e-7 if dtype.name == "float32" else 2e-10
            np.testing.assert_allclose(flattened, reference_state, rtol=tolerance, atol=absolute)
            np.testing.assert_allclose(
                derivative, reference_gradient, rtol=tolerance, atol=absolute
            )
            records[name] = {
                "configuration": asdict(numerical),
                "forward": forward_timing,
                "value_and_gradient": gradient_timing,
                "loss": float(value),
                "gradient": derivative.tolist(),
                "step_diagnostics": info.tolist(),
                "maximum_state_difference": float(np.max(np.abs(flattened - reference_state))),
            }
            arguments.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(
                f"{name}: forward={forward_timing['warm_median_seconds']:.6f}s "
                f"value+gradient={gradient_timing['warm_median_seconds']:.6f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
