"""Bind the legged locomotion protocol to the generic resident executor."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

from envs.soft_walker.contracts.metrics import (
    FlightMetric as Flight,
    LocomotionMetricLayout as Metric,
)
from envs.soft_walker.engine.physics import spring
from envs.soft_walker.engine.types import (
    DriverState,
    PhysicsModel,
)

if TYPE_CHECKING:
    from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv
    from envs.soft_walker.engine.drivers.task import TaskState
    from envs.soft_walker.experiments.locomotion.rollout import ObjectiveConfig
    from envs.soft_walker.presets.locomotion import DemoConfig

from envs.soft_walker.engine.execution.program import EpisodeProgram, bind_episode

from .policies import TRAVEL_HEIGHT_SMOOTHING_FRACTION


def bind_demo_episode(
    env: Spring2DEnv,
    demo: DemoConfig,
    *,
    objective: ObjectiveConfig | None = None,
    record_stride: int = 1,
    checkpoint_block: int = 5,
) -> EpisodeProgram:
    """Preserve the user's travel/hop objectives and active-stress demo variants."""
    from envs.soft_walker.engine.drivers.model_access import task_positions as positions
    from envs.soft_walker.experiments.locomotion.rollout import (
        ObjectiveConfig,
        hop_sample_terms,
        sample_flight_metrics,
        sample_metrics,
    )
    from envs.soft_walker.shared import periodic_action, region_tags

    objective = ObjectiveConfig() if objective is None else objective
    skip = demo.steps_per_cycle if objective.name == "hop" and demo.cycles > 1 else 0

    def measure(model: PhysicsModel, state: DriverState) -> jax.Array:
        """Compute the established locomotion metric vector using model-derived region tags."""
        torso, feet = region_tags(env, model)
        return sample_metrics(env, model, state, torso, feet)

    def running(
        model: PhysicsModel,
        initial: DriverState,
        previous: DriverState,
        new: DriverState,
        row: jax.Array,
        index: jax.Array,
    ) -> jax.Array:
        """Evaluate the selected hopping or clearance objective without changing its weights."""
        if objective.name == "hop":
            torso = region_tags(env, model)[0]
            flight = sample_flight_metrics(env, model, new, torso)
            initial_height = sample_flight_metrics(env, model, initial, torso)[Flight.com_height]
            previous_dx = (
                cast("TaskState", previous).displacement[0]
                if hasattr(env, "kind")
                else cast(spring.State, previous).com_displacement[0]
            )
            value = -objective.hop_progress_weight * (
                row[Metric.dx] - previous_dx
            ) / env.config.length_ref + jnp.sum(
                hop_sample_terms(flight, initial_height, env.config.size[1], objective)
            ) / (demo.steps - skip)
            return jnp.where(index >= skip, value, 0.0)
        shortage = TRAVEL_HEIGHT_SMOOTHING_FRACTION * jax.nn.softplus(
            (
                objective.minimum_torso_height_fraction * env.config.size[1]
                - row[Metric.torso_height]
            )
            / (TRAVEL_HEIGHT_SMOOTHING_FRACTION * env.config.size[1])
        )
        return shortage**2 / demo.steps

    def terminal(
        model: PhysicsModel,
        initial: DriverState,
        final: DriverState,
        row: jax.Array,
        cost: jax.Array,
    ) -> jax.Array:
        """Combine the selected objective with work and, in 3D, lateral penalties."""
        if objective.name == "hop":
            loss = cost + objective.work_weight * row[Metric.work_positive] / env.config.work_ref
            if demo.dim == 3:
                loss = (
                    loss + objective.lateral_weight * (row[Metric.dz] / env.config.length_ref) ** 2
                )
            return loss
        return (
            -row[Metric.dx] / env.config.length_ref
            + objective.work_weight * row[Metric.work_positive] / env.config.work_ref
            + objective.height_weight * cost
            + objective.lateral_weight * (row[Metric.dz] / env.config.length_ref) ** 2
        )

    return bind_episode(
        env,
        steps=demo.steps,
        action=lambda m, s, p, i: periodic_action(p, i * env.config.control_dt, demo.period),
        measure=measure,
        sample=lambda m, s: {
            "positions": positions(env, m, s),
            "flight": sample_flight_metrics(env, m, s),
        },
        running_cost=running,
        terminal_loss=terminal,
        record_stride=record_stride,
        checkpoint_block=checkpoint_block,
        description={"family": "demo", "objective": objective.name},
    )
