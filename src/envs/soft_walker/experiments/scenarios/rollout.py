"""Terrain-aware diagnostics and memory-bounded open/closed-loop rollouts.

The simulator transitions are the existing drivers; this is experiment support,
not a second integrator. Contact checks use the actual discrete contact sites.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from envs.soft_walker.contracts.metrics import scenario_metric_layout
from envs.soft_walker.engine.drivers import environment as drivers
from envs.soft_walker.engine.drivers.model_access import physical_state, task_positions as positions
from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv
from envs.soft_walker.engine.drivers.task import BodyConfig, TaskState
from envs.soft_walker.engine.physics import beam, mpm, spring
from envs.soft_walker.engine.physics.common import StepInfo
from envs.soft_walker.engine.physics.terrain import Terrain, gap_normal
from envs.soft_walker.engine.types import (
    AdvanceKernel,
    Array,
    BodyModel,
    DriverState,
    PhysicsModel,
    ScenarioStep,
    ScenarioStepData,
)
from envs.soft_walker.experiments.scenarios.controllers import action
from envs.soft_walker.experiments.scenarios.objectives import running_cost, terminal_cost
from envs.soft_walker.presets.scenarios.config import Scenario
from envs.soft_walker.shared import region_tags


def surface_samples(
    env: Spring2DEnv, model: PhysicsModel, state: DriverState
) -> tuple[jax.Array, jax.Array, jax.Array, float | jax.Array]:
    """Return contact quadrature positions, k*area coefficients, and site owners."""
    body = physical_state(env, state)
    xyz = positions(env, model, state)
    if not hasattr(env, "kind"):
        return (
            xyz,
            cast(spring.Model, model).contact_k,
            jnp.arange(len(xyz)),
            env.config.contact_width,
        )
    stiffness = cast(BodyConfig, env.config).dynamics.contact.stiffness
    if env.kind.startswith("mpm"):
        offset = jnp.einsum(
            "pij,fj->pfi", cast(mpm.State, body).F, cast(mpm.Model, model).cloud.face_offset
        )
        surface = xyz[:, None, :] + offset
    elif env.kind == "beam":
        offset = jnp.einsum(
            "pij,fj->pfi", cast(beam.State, body).rotation, cast(beam.Model, model).topology.offsets
        )
        surface = xyz[:, None, :] + offset
    else:
        return (
            xyz,
            stiffness * cast(BodyModel, model).contact_area,
            jnp.arange(len(xyz)),
            cast(BodyConfig, env.config).dynamics.contact.width,
        )
    owners = jnp.repeat(jnp.arange(len(xyz)), surface.shape[1])
    return (
        surface.reshape(-1, xyz.shape[-1]),
        stiffness * cast(BodyModel, model).contact_area.reshape(-1),
        owners,
        cast(BodyConfig, env.config).dynamics.contact.width,
    )


def columns(dim: int) -> list[str]:
    """Name scenario metric columns in their fixed record order."""
    return scenario_metric_layout(dim).columns()


def measure(
    env: Spring2DEnv,
    model: PhysicsModel,
    state: DriverState,
    tags: tuple[jax.Array, jax.Array] | None = None,
) -> jax.Array:
    """Collect task and contact metrics from the committed physical state."""
    body = physical_state(env, state)
    xyz = positions(env, model, state)
    torso, feet = region_tags(env, model) if tags is None else tags
    mass = jnp.where(model.active, model.mass, 0.0)
    total = jnp.maximum(jnp.sum(mass), 1e-20)
    velocity = jnp.einsum("n,nd->d", mass, body.v) / total
    torso_mass = jnp.where(torso, mass, 0.0)
    center = jnp.einsum("n,nd->d", torso_mass, xyz) / jnp.maximum(jnp.sum(torso_mass), 1e-20)
    center_gap, _ = gap_normal(center, cast(Terrain, model.terrain))
    sites, coefficients, owner, width = surface_samples(env, model, state)
    gap, _ = gap_normal(sites, cast(Terrain, model.terrain))
    exposed = coefficients > 0
    depth = width * jax.nn.softplus(-gap / width)
    normal_force = jnp.sum(coefficients * depth * jax.nn.sigmoid(-gap / width))
    energy = 0.5 * jnp.sum(coefficients * depth**2)
    # vmap independent groups, not consecutive physical timesteps.
    foot_gap = jax.vmap(lambda f: jnp.min(jnp.where(f[owner] & exposed, gap, jnp.inf)))(feet)
    foot_velocity = jax.vmap(
        lambda f: (
            jnp.sum(jnp.where(f, mass * body.v[:, 0], 0.0))
            / jnp.maximum(jnp.sum(jnp.where(f, mass, 0.0)), 1e-20)
        )
    )(feet)
    displacement = (
        cast(TaskState, state).displacement
        if hasattr(env, "kind")
        else cast(spring.State, state).com_displacement
    )
    disp3 = jnp.pad(displacement, (0, 3 - displacement.size))
    return jnp.concatenate(
        (
            jnp.stack(
                (
                    body.time,
                    *disp3,
                    body.work_positive,
                    body.work_signed,
                    state.status.astype(xyz.dtype),
                    center_gap,
                    jnp.linalg.norm(velocity),
                )
            ),
            foot_gap,
            foot_velocity,
            jnp.stack(
                (
                    normal_force / (total * 9.81),
                    jnp.min(jnp.where(exposed, gap, jnp.inf)),
                    jnp.min(jnp.where(exposed & torso[owner], gap, jnp.inf)),
                    velocity[1],
                    energy,
                )
            ),
        )
    )


def bind_step(env: Spring2DEnv, case: Scenario) -> ScenarioStep:
    """Return (model, state, controller, integer_index) -> state and diagnostics."""
    advance: AdvanceKernel
    if hasattr(env, "kind"):
        advance = lambda m, s, a: drivers.advance(
            env.kind,
            cast(BodyModel, m),
            cast(TaskState, s),
            a,
            cast(BodyConfig, env.config),
            env._grid,
        )
    else:
        advance = lambda m, s, a: spring.advance(
            cast(spring.Model, m), cast(spring.State, s), a, env.config
        )

    def step(
        model: PhysicsModel, state: DriverState, parameters: jax.Array, index: jax.Array
    ) -> tuple[DriverState, ScenarioStepData]:
        """Evaluate the configured controller, advance physics, and record committed diagnostics."""
        observation = (
            env._observe(model, state)[0] if case.controller in ("feedback", "neural") else None
        )
        command = action(case, parameters, index * env.config.control_dt, observation)
        new, information = advance(model, state, command)
        if hasattr(env, "kind"):
            info = cast(tuple[StepInfo, jax.Array], information)[0]
            solver = jnp.stack(
                (
                    info.code,
                    info.max_residual,
                    info.max_linear_residual,
                    info.min_J,
                    info.support_valid,
                    info.nonlinear_iterations,
                )
            )
        else:
            information = cast(spring.Diagnostics, information)
            solver = jnp.stack(
                (
                    information.solver_code,
                    information.max_residual,
                    jnp.asarray(-1.0, parameters.dtype),
                    information.min_area_ratio,
                    jnp.asarray(1.0, parameters.dtype),
                    information.nonlinear_iterations,
                )
            )
        return new, (measure(env, model, new), command, positions(env, model, new), solver)

    return step


def bind_chunk(
    env: Spring2DEnv, case: Scenario, chunk_size: int = 10
) -> Callable[
    [PhysicsModel, DriverState, jax.Array, jax.Array], tuple[DriverState, ScenarioStepData]
]:
    """Record only chunk_size intervals. Host can persist each accepted chunk."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    step = bind_step(env, case)

    def chunk(
        model: PhysicsModel, initial: DriverState, parameters: jax.Array, start: jax.Array
    ) -> tuple[DriverState, ScenarioStepData]:
        """Scan a bounded action-index range and return per-step evidence on device."""

        def one(state: DriverState, index: jax.Array) -> tuple[DriverState, ScenarioStepData]:
            """Advance valid in-horizon states; otherwise emit a stationary diagnostic record."""

            def run(_: None) -> tuple[DriverState, ScenarioStepData]:
                """Evaluate the normal scenario step for one valid action index."""
                return step(model, state, parameters, index)

            def stop(_: None) -> tuple[DriverState, ScenarioStepData]:
                """Retain state and mark skipped dynamics while preserving fixed history shapes."""
                return state, (
                    measure(env, model, state),
                    jnp.zeros(case.channels, parameters.dtype),
                    positions(env, model, state),
                    -jnp.ones(6, parameters.dtype),
                )

            return jax.lax.cond((index < case.steps) & (state.status == 0), run, stop, None)

        return jax.lax.scan(one, initial, start + jnp.arange(chunk_size))

    return chunk


def bind_loss(
    env: Spring2DEnv, case: Scenario, block_size: int = 10
) -> Callable[[PhysicsModel, DriverState, jax.Array], jax.Array]:
    """Checkpointed control loss. Fixed design/reset; no position history output.

    This loss is a training aid, not the physical terminal objective definition.
    Numerical invalidity produces NaN so optimizers cannot prefer failed trials.
    """
    step = bind_step(env, case)
    blocks = math.ceil(case.steps / block_size)

    def loss(model: PhysicsModel, initial: DriverState, parameters: jax.Array) -> jax.Array:
        """Scan checkpointed actions and return a dimensionless objective or NaN for invalid
        rollouts.
        """
        zero = jnp.asarray(0.0, parameters.dtype)

        def one(
            carry: tuple[DriverState, jax.Array], index: jax.Array
        ) -> tuple[tuple[DriverState, jax.Array], None]:
            """Accumulate an interval cost only for a valid in-horizon state."""
            state, cost = carry

            def run(_: None) -> tuple[DriverState, jax.Array]:
                """Advance one valid interval and accumulate terrain-aware height and travel
                shaping.
                """
                new, data = step(model, state, parameters, index)
                row = data[0]
                increment = running_cost(row, case, zero)
                return new, cost + increment

            return jax.lax.cond(
                (index < case.steps) & (state.status == 0), run, lambda _: carry, None
            ), None

        def block(
            carry: tuple[DriverState, jax.Array], indices: jax.Array
        ) -> tuple[tuple[DriverState, jax.Array], None]:
            """Checkpoint a fixed group of intervals and retain only its final carry."""
            return jax.lax.scan(jax.checkpoint(one), carry, indices)[0], None

        (final, cost), _ = jax.lax.scan(
            jax.checkpoint(block),
            (initial, zero),
            jnp.arange(blocks * block_size).reshape(blocks, block_size),
        )
        row = measure(env, model, final)
        objective = terminal_cost(row, case, env.config.work_ref, cost)
        valid = (final.status == 0) & jnp.isclose(
            row[scenario_metric_layout(case.dim).time],
            case.steps * env.config.control_dt,
            rtol=5e-4,
            atol=1e-6,
        )
        return jnp.where(valid, objective, jnp.nan)

    return loss


def assess(rows: Array, case: Scenario, *, design_mode: str = "hard") -> dict[str, object]:
    """Separate numerical validity, walking, and observed discrete-contact flight."""
    rows = np.asarray(rows)
    layout = scenario_metric_layout(case.dim)
    dt = case.effective_control_dt
    valid = (
        rows.ndim == 2
        and len(rows) == case.steps + 1
        and np.isfinite(rows).all()
        and np.all(rows[:, layout.status] == 0)
        and np.isclose(rows[-1, layout.time], case.steps * dt, rtol=5e-4, atol=1e-6)
    )
    # Exclude initialized sample and the first commanded cycle for qualification.
    start = min(case.steps_per_cycle + 1, len(rows))
    post = rows[start:]
    feet = post[:, layout.foot_gaps]
    swings = np.any(feet > 0.003, axis=0)
    stances = np.any(feet < 0.001, axis=0)
    airborne = (
        (post[:, layout.normal_force_over_weight] < 0.01)
        & (post[:, layout.all_surface_gap] > 0.003)
        & (post[:, layout.status] == 0)
    )
    spans = []
    index = 0
    while index < len(airborne):
        if not airborne[index]:
            index += 1
            continue
        first = index
        while index < len(airborne) and airborne[index]:
            index += 1
        # Conservative duration between samples, not the number of samples * dt.
        duration = float(post[index - 1, layout.time] - post[first, layout.time])
        spans.append(
            {
                "start_s": float(post[first, layout.time]),
                "duration_s": duration,
                "landed_in_recording": bool(index < len(airborne)),
            }
        )
    flights = [s for s in spans if s["duration_s"] >= 0.02 and s["landed_in_recording"]]
    endpoints = rows[case.steps_per_cycle :: case.steps_per_cycle, layout.dx]
    increments = np.diff(np.r_[0.0, endpoints])
    post_distance = float(np.sum(increments[1:]))
    torso_clear = bool(len(post) and np.all(post[:, layout.torso_surface_gap] > 0.003))
    walking = (
        valid
        and design_mode == "hard"
        and len(increments) >= 6
        and post_distance >= 0.25 * case.size[0]
        and np.count_nonzero(increments[1:] > 0) >= 4
        and bool(np.all(swings & stances))
        and torso_clear
    )
    return {
        "valid_rollout": bool(valid),
        "walking_gate_passed": bool(walking),
        "observed_airborne_intervals": spans,
        "completed_flights_at_least_20ms": len(flights),
        "repeated_airborne_observed": bool(valid and len(flights) >= 2 and design_mode == "hard"),
        "each_foot_swings": swings.tolist(),
        "each_foot_stances": stances.tolist(),
        "torso_clear": torso_clear,
        "per_cycle_distance_m": increments.tolist(),
        "post_startup_distance_m": post_distance,
        "total_distance_m": float(rows[-1, layout.dx]) if len(rows) else None,
        "positive_work_j": float(rows[-1, layout.work_positive]) if len(rows) else None,
        "minimum_recorded_normal_force_over_weight": float(
            np.min(rows[:, layout.normal_force_over_weight])
        )
        if len(rows)
        else None,
        "qualification": "discrete contact sites; spatial/time refinement and negative controls still required",
    }
