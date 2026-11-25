"""Memory-bounded, device-resident experiment rollouts through existing physics.

The public factory/reset API creates and validates the robot. These functions
bind to that environment; they do not define a second simulator. Optimization
and replay use the same action clock, initial state and numerical transition.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from envs.soft_walker.contracts.metrics import (
    FlightMetric as Flight,
    LocomotionMetricLayout as Metric,
    LocomotionSolver,
    flight_metric_columns,
    locomotion_metric_columns,
)
from envs.soft_walker.engine.drivers import environment as drivers
from envs.soft_walker.engine.drivers.model_access import (
    physical_state,
    task_positions as positions,
    task_reference_positions as reference_positions,
)
from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv
from envs.soft_walker.engine.drivers.task import BodyConfig, TaskState
from envs.soft_walker.engine.physics import beam, mpm, spring
from envs.soft_walker.engine.physics.common import StepInfo
from envs.soft_walker.engine.physics.terrain import Terrain
from envs.soft_walker.engine.types import (
    AdvanceKernel,
    Array,
    BodyModel,
    DriverState,
    ExperimentHistory,
    ExperimentResult,
    PhysicsModel,
)
from envs.soft_walker.presets.locomotion import DemoConfig
from envs.soft_walker.shared import (
    periodic_action,
    region_tags,
)

from .policies import (
    COVARIANCE_NORM2_FLOOR,
    FLIGHT_QUALIFICATION,
    LEGACY_WORK_REFERENCES,
    MASS_DENOMINATOR_FLOOR,
    ROLLOUT_TIME_ATOL,
    ROLLOUT_TIME_RTOL,
    TRAVEL_HEIGHT_SMOOTHING_FRACTION,
    WALKING_QUALIFICATION,
    WEIGHT_DENOMINATOR_FLOOR,
)


@dataclass(frozen=True)
class ObjectiveConfig:
    """Training loss, deliberately separate from authoritative physical metrics.

    Travel keeps the original expression. Hop averages lift, uprightness and
    torso-ground terms after the first cycle (all samples for a one-cycle run),
    rewards progress over the same window, charges total positive work, and
    penalizes final lateral drift in 3D.
    The settled initial COM height anchors lift. It can reward standing taller;
    only independent contact/clearance diagnostics qualify sampled flight.
    Weights are dimensionless after the stated normalizations. Values are the
    inherited exploratory protocol; original empirical calibration was not recorded.
    """

    work_weight: float = 0.001
    height_weight: float = 0.25
    lateral_weight: float = 0.1
    # Initial gait discovery uses all displacement. Frozen acceptance separately
    # excludes one startup cycle and checks multiple subsequent cycles.
    minimum_torso_height_fraction: float = 0.45
    name: str = "travel"
    hop_progress_weight: float = 0.25
    hop_lift_weight: float = 2.0
    hop_upright_weight: float = 2.0
    hop_torso_ground_weight: float = 0.25
    hop_lift_smoothing_fraction: float = 0.05
    hop_torso_gap_fraction: float = 0.02

    def __post_init__(self) -> None:
        """Select a documented exploratory objective without changing physics metrics."""
        if self.name not in ("travel", "hop"):
            raise ValueError("Objective must be travel or hop")


def objective_record(objective: ObjectiveConfig, demo: DemoConfig) -> dict[str, object]:
    """Persist the exact exploratory loss independently of physical assessment."""
    from dataclasses import asdict

    return {
        "schema": "locomotion_objective_v1",
        **asdict(objective),
        "startup_cycles_excluded": 1 if objective.name == "hop" and demo.cycles > 1 else 0,
        "formula": (
            "-dx/L + work_weight*positive_work/work_ref + height_weight*height_cost + lateral_weight*(dz/L)^2"
            if objective.name == "travel"
            else "-hop_progress_weight*post_startup_dx/L "
            "- hop_lift_weight*mean(hop_lift_smoothing_fraction*softplus((COM_y-initial_COM_y)/(hop_lift_smoothing_fraction*H))) "
            "+ hop_upright_weight*mean(1-upright_cosine) "
            "+ hop_torso_ground_weight*mean(softplus((hop_torso_gap_fraction*H-torso_surface_gap)/(hop_torso_gap_fraction*H))^2) "
            "+ work_weight*positive_work/work_ref"
            + (" + lateral_weight*(dz/L)^2" if demo.dim == 3 else "")
        ),
        "qualification": "Exploratory optimization objective; lift reward can favor standing taller without flight. Flight is assessed separately from actual discrete contact sites.",
    }


def _weighted_mean(model: PhysicsModel, value: jax.Array, mask: jax.Array) -> jax.Array:
    """Compute a masked mass-weighted mean of node or particle vectors."""
    mass = jnp.where(mask, model.mass, 0.0)
    return jnp.einsum("n,nd->d", mass, value) / jnp.maximum(jnp.sum(mass), MASS_DENOMINATOR_FLOOR)


def sample_metrics(
    env: Spring2DEnv, model: PhysicsModel, state: DriverState, torso: jax.Array, feet: jax.Array
) -> jax.Array:
    """Small diagnostic vector, independent of observation padding capacity.

    Columns: time, dx, dy, dz, positive/signed work, status, torso height,
    speed norm, then per-foot lower-surface clearance and horizontal velocity.
    Foot clearances are geometry diagnostics, not a rigid/static-contact claim.
    """
    body = physical_state(env, state)
    x = positions(env, model, state)
    disp = (
        cast(TaskState, state).displacement
        if hasattr(env, "kind")
        else cast(spring.State, state).com_displacement
    )
    disp3 = jnp.pad(disp, (0, 3 - disp.size))
    center = _weighted_mean(model, x, torso)
    velocity = _weighted_mean(model, body.v, model.active)
    bottom = x[:, 1]
    if hasattr(env, "kind") and env.kind.startswith("mpm"):
        # Lower face of the represented particle cell, transformed by F.
        hp = cast(mpm.Grid, env._grid).spacing / cast(BodyConfig, env.config).particles_per_axis
        bottom = bottom - 0.5 * hp * jnp.sum(jnp.abs(cast(mpm.State, body).F[:, 1, :]), axis=-1)
    elif hasattr(env, "kind") and env.kind == "beam":
        bottom = bottom - 0.5 * cast(beam.Model, model).topology.h
    clearance = jnp.stack([jnp.min(jnp.where(f, bottom, jnp.inf)) for f in feet])
    foot_vx = jnp.stack([_weighted_mean(model, body.v, f)[0] for f in feet])
    return jnp.concatenate(
        (
            jnp.stack(
                (
                    body.time,
                    *disp3,
                    body.work_positive,
                    body.work_signed,
                    state.status.astype(x.dtype),
                    center[1],
                    jnp.linalg.norm(velocity),
                )
            ),
            clearance,
            foot_vx,
        )
    )


def metric_columns(dim: int) -> list[str]:
    """Name the recorded metric columns for the selected spatial dimension."""
    return locomotion_metric_columns(dim)


def flight_columns() -> list[str]:
    """Name discrete-contact and body-state columns used for flight assessment."""
    return flight_metric_columns()


def sample_flight_metrics(
    env: Spring2DEnv, model: PhysicsModel, state: DriverState, torso: jax.Array | None = None
) -> jax.Array:
    """Measure actual exposed contact sites; marker positions alone cannot show flight.

    The torso orientation uses reference-coordinate covariance, without an SVD.
    In 2D it is the signed rotation of the reference horizontal torso direction.
    In 3D it is the inclination of the reference vertical direction. Contact uses
    the same exposed face-centroid quadrature and smooth force law as the driver.
    """
    from envs.soft_walker.engine.physics.terrain import gap_normal

    body = physical_state(env, state)
    xyz = positions(env, model, state)
    torso = region_tags(env, model)[0] if torso is None else torso
    mass = jnp.where(model.active, model.mass, 0.0)
    total = jnp.maximum(jnp.sum(mass), MASS_DENOMINATOR_FLOOR)
    center = jnp.sum(mass[:, None] * xyz, axis=0) / total
    velocity = jnp.sum(mass[:, None] * body.v, axis=0) / total
    ref = reference_positions(env, model)
    torso_mass = jnp.where(torso, mass, 0.0)
    torso_total = jnp.maximum(jnp.sum(torso_mass), MASS_DENOMINATOR_FLOOR)
    ref_center = jnp.sum(torso_mass[:, None] * ref, axis=0) / torso_total
    torso_center = jnp.sum(torso_mass[:, None] * xyz, axis=0) / torso_total
    dim = xyz.shape[-1]
    axis = 0 if dim == 2 else 1
    direction = jnp.sum(
        torso_mass[:, None] * (ref[:, axis] - ref_center[axis])[:, None] * (xyz - torso_center),
        axis=0,
    )
    length = jnp.sqrt(jnp.sum(direction**2) + COVARIANCE_NORM2_FLOOR)
    upright = jnp.clip(direction[axis] / length, -1.0, 1.0)
    angle = (
        jnp.arctan2(direction[1], direction[0])
        if dim == 2
        else jnp.arctan2(
            jnp.sqrt(direction[0] ** 2 + direction[2] ** 2 + COVARIANCE_NORM2_FLOOR), direction[1]
        )
    )
    if not hasattr(env, "kind"):
        sites, coefficients = xyz, cast(spring.Model, model).contact_k
        owners = jnp.arange(len(xyz))
        width, gravity = env.config.contact_width, env.config.gravity
    else:
        contact = cast(BodyConfig, env.config).dynamics.contact
        width, gravity = contact.width, cast(BodyConfig, env.config).dynamics.gravity
        if env.kind.startswith("mpm"):
            offset = jnp.einsum(
                "pij,fj->pfi", cast(mpm.State, body).F, cast(mpm.Model, model).cloud.face_offset
            )
            sites = xyz[:, None, :] + offset
        elif env.kind == "beam":
            offset = jnp.einsum(
                "pij,fj->pfi",
                cast(beam.State, body).rotation,
                cast(beam.Model, model).topology.offsets,
            )
            sites = xyz[:, None, :] + offset
        else:
            sites = xyz
        owners = (
            jnp.repeat(jnp.arange(len(xyz)), sites.shape[1])
            if sites.ndim == 3
            else jnp.arange(len(xyz))
        )
        sites = sites.reshape(-1, dim)
        coefficients = (contact.stiffness * cast(BodyModel, model).contact_area).reshape(-1)
    gaps, _ = gap_normal(sites, cast(Terrain, model.terrain))
    exposed = coefficients > 0
    depth = width * jax.nn.softplus(-gaps / width)
    force = jnp.sum(coefficients * depth * jax.nn.sigmoid(-gaps / width))
    weight = total * gravity
    return jnp.stack(
        (
            body.time,
            center[1],
            velocity[1],
            force,
            weight,
            force / jnp.maximum(weight, WEIGHT_DENOMINATOR_FLOOR),
            jnp.min(jnp.where(exposed, gaps, jnp.inf)),
            jnp.min(jnp.where(exposed & torso[owners], gaps, jnp.inf)),
            angle,
            upright,
        )
    )


def hop_sample_terms(
    flight: jax.Array, initial_height: jax.Array, height: float, objective: ObjectiveConfig
) -> jax.Array:
    """Smooth dimensionless lift/upright/torso terms; no discrete flight reward."""
    smooth = objective.hop_lift_smoothing_fraction
    gap = objective.hop_torso_gap_fraction * height
    return jnp.stack(
        (
            -objective.hop_lift_weight
            * smooth
            * jax.nn.softplus((flight[Flight.com_height] - initial_height) / (smooth * height)),
            objective.hop_upright_weight * (1.0 - flight[Flight.torso_upright]),
            objective.hop_torso_ground_weight
            * jax.nn.softplus((gap - flight[Flight.torso_surface_gap]) / gap) ** 2,
        )
    )


def objective_components(
    metrics: Array,
    flight: jax.Array,
    demo: DemoConfig,
    objective: ObjectiveConfig,
    *,
    work_ref: float | None = None,
) -> dict[str, float]:
    """Recompute loss terms from recorded rows without advancing physics.

    Live callers pass their environment's positive work reference, in joules.
    Omitting it preserves the historical dimension-specific preset interpretation
    for old records that did not save the normalizer. An explicit nonpositive or
    nonfinite reference is rejected instead of silently changing the objective.
    """
    rows = np.asarray(metrics)
    work_ref = LEGACY_WORK_REFERENCES[demo.dim] if work_ref is None else work_ref
    if not math.isfinite(work_ref) or work_ref <= 0:
        raise ValueError("work_ref must be finite and positive (joules)")
    if objective.name == "travel":
        shortage = TRAVEL_HEIGHT_SMOOTHING_FRACTION * np.logaddexp(
            0.0,
            (objective.minimum_torso_height_fraction * demo.size[1] - rows[1:, Metric.torso_height])
            / (TRAVEL_HEIGHT_SMOOTHING_FRACTION * demo.size[1]),
        )
        terms = dict(
            forward=-rows[-1, Metric.dx] / demo.size[0],
            work=objective.work_weight * rows[-1, Metric.work_positive] / work_ref,
            torso_height=objective.height_weight * np.mean(shortage**2),
            lateral=objective.lateral_weight * (rows[-1, Metric.dz] / demo.size[0]) ** 2,
        )
    else:
        f = np.asarray(flight)
        first = demo.steps_per_cycle if demo.cycles > 1 else 0
        post = f[first + 1 :]
        smooth = objective.hop_lift_smoothing_fraction
        gap = objective.hop_torso_gap_fraction * demo.size[1]
        terms = dict(
            forward=-objective.hop_progress_weight
            * (rows[-1, Metric.dx] - rows[first, Metric.dx])
            / demo.size[0],
            lift=-objective.hop_lift_weight
            * np.mean(
                smooth
                * np.logaddexp(
                    0.0,
                    (post[:, Flight.com_height] - f[0, Flight.com_height])
                    / (smooth * demo.size[1]),
                )
            ),
            upright=objective.hop_upright_weight * np.mean(1.0 - post[:, Flight.torso_upright]),
            torso_ground=objective.hop_torso_ground_weight
            * np.mean(np.logaddexp(0.0, (gap - post[:, Flight.torso_surface_gap]) / gap) ** 2),
            work=objective.work_weight * rows[-1, Metric.work_positive] / work_ref,
            lateral=(
                objective.lateral_weight * (rows[-1, Metric.dz] / demo.size[0]) ** 2
                if demo.dim == 3
                else 0.0
            ),
        )
    return {
        **{key: float(value) for key, value in terms.items()},
        "total": float(sum(terms.values())),
    }


def flight_assessment(
    flight: jax.Array, demo: DemoConfig, *, valid_rollout: bool
) -> dict[str, object]:
    """Qualify sampled flight separately from walking and optimization loss."""
    rows = np.asarray(flight)
    finite = (
        rows.ndim == 2
        and rows.shape == (demo.steps + 1, len(flight_columns()))
        and np.isfinite(rows).all()
    )
    if not finite:
        return {
            "valid_rollout": bool(valid_rollout),
            "diagnostics_finite_and_complete": False,
            "repeated_airborne_observed": False,
            "observed_airborne_intervals": [],
            "completed_flights_at_least_20ms": 0,
        }
    policy = FLIGHT_QUALIFICATION
    first = policy.startup_cycles * demo.steps_per_cycle + 1 if demo.cycles > 1 else 1
    post = rows[first:]
    force_free = (post[:, Flight.weight] > 0) & (
        post[:, Flight.normal_force_over_weight] < policy.force_fraction
    )
    spans, index, episode, completed = [], 0, 0, 0
    while index < len(post):
        if not force_free[index]:
            index += 1
            continue
        begin_episode = index
        while index < len(post) and force_free[index]:
            index += 1
        landed = bool(
            index < len(post)
            and post[index, Flight.normal_force_over_weight] >= policy.force_fraction
        )
        episode += 1
        cursor, qualifies = begin_episode, False
        while cursor < index:
            if post[cursor, Flight.all_surface_gap] <= policy.surface_clearance:
                cursor += 1
                continue
            begin = cursor
            while (
                cursor < index and post[cursor, Flight.all_surface_gap] > policy.surface_clearance
            ):
                cursor += 1
            duration = float(post[cursor - 1, Flight.time] - post[begin, Flight.time])
            qualifies = qualifies or duration >= policy.minimum_duration - policy.duration_atol
            spans.append(
                {
                    "start_s": float(post[begin, Flight.time]),
                    "end_s": float(post[cursor - 1, Flight.time]),
                    "duration_s": duration,
                    "landed_in_recording": landed,
                    "contact_free_episode": episode,
                }
            )
        completed += int(qualifies and landed)
    return {
        "valid_rollout": bool(valid_rollout),
        "diagnostics_finite_and_complete": True,
        "startup_cycles_excluded": policy.startup_cycles if demo.cycles > 1 else 0,
        "force_threshold_over_actual_weight": policy.force_fraction,
        "minimum_all_surface_gap_m": policy.surface_clearance,
        "observed_airborne_intervals": spans,
        "completed_flights_at_least_20ms": completed,
        "repeated_airborne_observed": bool(
            valid_rollout and completed >= policy.completed_episodes
        ),
        "minimum_recorded_normal_force_over_weight": float(
            rows[:, Flight.normal_force_over_weight].min()
        ),
        "maximum_all_surface_gap_m": float(rows[:, Flight.all_surface_gap].max()),
        "maximum_abs_torso_angle_degrees": float(
            np.degrees(np.abs(rows[:, Flight.torso_angle])).max()
        ),
        "qualification": "Observed at control boundaries using exposed contact face centroids, not the entire geometric surface. Spatial/time refinement and negative controls remain separate.",
    }


def bind_experiment(
    env: Spring2DEnv,
    demo: DemoConfig,
    *,
    record: bool = False,
    block_size: int = 5,
    objective: ObjectiveConfig = ObjectiveConfig(),
) -> Callable[[PhysicsModel, DriverState, jax.Array], ExperimentResult]:
    """Return rollout(model, initialized_state, controller) -> result.

    The fixed material model and zero-command settling are evaluated once for
    control optimization. For joint design optimization, pass a differentiably
    instantiated model/settled state instead; they are dynamic arguments.
    Block checkpointing bounds stored large states. Only replay stores positions;
    optimization accumulates a small metric vector and losses, not observations.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")
    blocks = math.ceil(demo.steps / block_size)
    cfg = env.config
    advance: AdvanceKernel
    if hasattr(env, "kind"):
        advance = lambda m, s, a: drivers.advance(
            env.kind, cast(BodyModel, m), cast(TaskState, s), a, cast(BodyConfig, cfg), env._grid
        )
    else:
        advance = lambda m, s, a: spring.advance(
            cast(spring.Model, m), cast(spring.State, s), a, cfg
        )

    def rollout(
        model: PhysicsModel, initial: DriverState, parameters: jax.Array
    ) -> ExperimentResult:
        """Roll out one fixed model and controller, optionally recording metrics and geometry."""
        torso, feet = region_tags(env, model)
        initial_height = (
            sample_flight_metrics(env, model, initial, torso)[Flight.com_height]
            if objective.name == "hop"
            else None
        )
        skip = demo.steps_per_cycle if objective.name == "hop" and demo.cycles > 1 else 0

        def one(
            carry: tuple[DriverState, jax.Array], index: jax.Array
        ) -> tuple[tuple[DriverState, jax.Array], ExperimentHistory]:
            """Advance an in-horizon, valid state or preserve the checkpoint carry unchanged."""
            state, height_cost = carry

            def accepted_step(
                _: None,
            ) -> tuple[DriverState, jax.Array, jax.Array, jax.Array, jax.Array]:
                """Evaluate one periodic command and accumulate objective terms plus solver
                diagnostics.
                """
                action = periodic_action(parameters, index * cfg.control_dt, demo.period)
                new, diag = advance(model, state, action)
                if hasattr(env, "kind"):
                    root = cast(tuple[StepInfo, jax.Array], diag)[0]
                    numerical = jnp.stack(
                        (
                            root.code,
                            root.max_residual,
                            root.max_linear_residual,
                            root.min_J,
                            root.support_valid,
                            root.max_penetration,
                            root.nonlinear_iterations,
                        )
                    )
                else:
                    diag = cast(spring.Diagnostics, diag)
                    numerical = jnp.stack(
                        (
                            diag.solver_code,
                            diag.max_residual,
                            jnp.asarray(-1.0, parameters.dtype),
                            diag.min_area_ratio,
                            jnp.asarray(1.0, parameters.dtype),
                            diag.max_penetration,
                            diag.nonlinear_iterations,
                        )
                    )
                row = sample_metrics(env, model, new, torso, feet)
                if objective.name == "hop":
                    flight = sample_flight_metrics(env, model, new, torso)
                    previous_dx = (
                        cast(TaskState, state).displacement[0]
                        if hasattr(env, "kind")
                        else cast(spring.State, state).com_displacement[0]
                    )
                    increment = -objective.hop_progress_weight * (
                        row[Metric.dx] - previous_dx
                    ) / cfg.length_ref + jnp.sum(
                        hop_sample_terms(flight, initial_height, cfg.size[1], objective)
                    ) / (demo.steps - skip)
                    cost = height_cost + jnp.where(index >= skip, increment, 0.0)
                else:
                    shortage = (
                        jax.nn.softplus(
                            (
                                objective.minimum_torso_height_fraction * cfg.size[1]
                                - row[Metric.torso_height]
                            )
                            / (TRAVEL_HEIGHT_SMOOTHING_FRACTION * cfg.size[1])
                        )
                        * TRAVEL_HEIGHT_SMOOTHING_FRACTION
                    )
                    cost = height_cost + shortage**2 / demo.steps
                # Keep true failure visible; freeze without silently restarting.
                return new, cost, row, action, numerical

            def padding(_: None) -> tuple[DriverState, jax.Array, jax.Array, jax.Array, jax.Array]:
                """Retain the carry and emit sentinel metrics after the horizon or numerical
                failure.
                """
                return (
                    state,
                    height_cost,
                    sample_metrics(env, model, state, torso, feet),
                    jnp.zeros(env.channels, parameters.dtype),
                    -jnp.ones(7, parameters.dtype),
                )

            new, cost, row, action, numerical = jax.lax.cond(
                (index < demo.steps) & (state.status == 0), accepted_step, padding, None
            )
            data = (
                (
                    row,
                    action,
                    positions(env, model, new),
                    numerical,
                    sample_flight_metrics(env, model, new, torso),
                )
                if record
                else None
            )
            return (new, cost), data

        def block(
            carry: tuple[DriverState, jax.Array], indices: jax.Array
        ) -> tuple[tuple[DriverState, jax.Array], ExperimentHistory]:
            """Checkpoint one fixed block of action indices without retaining intermediate large
            states.
            """
            return jax.lax.scan(jax.checkpoint(one), carry, indices)

        indices = jnp.arange(blocks * block_size).reshape(blocks, block_size)
        (final, height_cost), history = jax.lax.scan(
            jax.checkpoint(block), (initial, jnp.asarray(0.0, parameters.dtype)), indices
        )
        row = sample_metrics(env, model, final, torso, feet)
        if objective.name == "hop":
            loss = height_cost + objective.work_weight * row[Metric.work_positive] / cfg.work_ref
            if demo.dim == 3:
                loss = loss + objective.lateral_weight * (row[Metric.dz] / cfg.length_ref) ** 2
        else:
            loss = (
                -row[Metric.dx] / cfg.length_ref
                + objective.work_weight * row[Metric.work_positive] / cfg.work_ref
                + objective.height_weight * height_cost
                + objective.lateral_weight * (row[Metric.dz] / cfg.length_ref) ** 2
            )
        valid = (
            (final.status == 0)
            & jnp.isfinite(loss)
            & jnp.isclose(
                physical_state(env, final).time,
                demo.steps * cfg.control_dt,
                rtol=ROLLOUT_TIME_RTOL,
                atol=ROLLOUT_TIME_ATOL,
            )
        )
        result: ExperimentResult = {
            "loss": jnp.where(valid, loss, jnp.nan),
            "valid": valid,
            "final_metrics": row,
            "state": final,
        }
        if record:
            result["metrics"] = history[0].reshape(-1, history[0].shape[-1])[: demo.steps]
            result["actions"] = history[1].reshape(-1, env.channels)[: demo.steps]
            result["positions"] = history[2].reshape((-1,) + history[2].shape[2:])[: demo.steps]
            result["solver_diagnostics"] = history[3].reshape(-1, LocomotionSolver.width)[
                : demo.steps
            ]
            result["flight_metrics"] = history[4].reshape(-1, len(flight_columns()))[: demo.steps]
        return result

    return rollout


def gait_assessment(metrics: Array, demo: DemoConfig) -> dict[str, object]:
    """Conservative frozen-replay checks; motion alone is not labelled walking."""
    rows = np.asarray(metrics)
    policy = WALKING_QUALIFICATION
    valid = (
        len(rows) == demo.steps and np.isfinite(rows).all() and np.all(rows[:, Metric.status] == 0)
    )
    cycle_end = rows[demo.steps_per_cycle - 1 :: demo.steps_per_cycle, Metric.dx]
    increments = np.diff(np.r_[0.0, cycle_end])
    post = increments[policy.startup_cycles :]
    progress = float(np.sum(post)) if len(post) else 0.0
    feet = rows[:, Metric(demo.dim).foot_gaps]
    threshold = max(policy.minimum_swing_clearance, demo.cell_size * policy.swing_cell_fraction)
    swings = np.any(feet > threshold, axis=0)
    stances = np.any(feet < threshold / policy.stance_threshold_divisor, axis=0)
    torso_clear = (
        bool(np.min(rows[:, Metric.torso_height]) > demo.size[1] * policy.torso_height_fraction)
        if len(rows)
        else False
    )
    passes = (
        valid
        and len(post) >= policy.minimum_cycles
        and progress >= policy.minimum_progress * demo.size[0]
        and np.count_nonzero(post > 0) >= policy.positive_cycles
        and bool(np.all(swings & stances))
        and torso_clear
    )
    return {
        "valid_rollout": bool(valid),
        "walking_gate_passed": bool(passes),
        "startup_cycles_excluded": policy.startup_cycles,
        "post_startup_cycles": len(post),
        "post_startup_distance_m": progress,
        "post_startup_body_lengths": progress / demo.size[0],
        "per_cycle_distance_m": increments.tolist(),
        "each_foot_swings": swings.tolist(),
        "each_foot_stances": stances.tolist(),
        "torso_clear": torso_clear,
        "qualification": "candidate; longer replay/refinement and controls are separate checks",
    }
