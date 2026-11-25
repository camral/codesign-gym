"""Bind the scenario protocol to the generic resident executor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax

from envs.soft_walker.engine.types import (
    DriverState,
    PhysicsModel,
)

if TYPE_CHECKING:
    from envs.soft_walker.engine.drivers.spring_environment import Spring2DEnv
    from envs.soft_walker.presets.scenarios.config import Scenario

from envs.soft_walker.engine.execution.program import EpisodeProgram, bind_episode


def bind_scenario_episode(
    env: Spring2DEnv, case: Scenario, *, record_stride: int = 1, checkpoint_block: int = 10
) -> EpisodeProgram:
    """Preserve the existing scenario training objective and controller semantics."""
    from envs.soft_walker.engine.drivers.model_access import task_positions as positions
    from envs.soft_walker.experiments.scenarios.controllers import action as controller_action
    from envs.soft_walker.experiments.scenarios.objectives import running_cost, terminal_cost
    from envs.soft_walker.experiments.scenarios.rollout import measure

    def action(
        model: PhysicsModel, state: DriverState, parameters: jax.Array, index: jax.Array
    ) -> jax.Array:
        """Evaluate the scenario controller at physical time, including requested observations."""
        observation = (
            env._observe(model, state)[0] if case.controller in ("feedback", "neural") else None
        )
        return controller_action(case, parameters, index * env.config.control_dt, observation)

    def running(
        model: PhysicsModel,
        initial: DriverState,
        previous: DriverState,
        new: DriverState,
        row: jax.Array,
        index: jax.Array,
    ) -> jax.Array:
        """Accumulate the existing height/clearance cost per scenario control interval."""
        return running_cost(row, case)

    def terminal(
        model: PhysicsModel,
        initial: DriverState,
        final: DriverState,
        row: jax.Array,
        cost: jax.Array,
    ) -> jax.Array:
        """Combine normalized forward displacement, mechanical work, and accumulated cost."""
        return terminal_cost(row, case, env.config.work_ref, cost)

    return bind_episode(
        env,
        steps=case.steps,
        action=action,
        measure=lambda m, s: measure(env, m, s),
        sample=lambda m, s: {"positions": positions(env, m, s)},
        running_cost=running,
        terminal_loss=terminal,
        record_stride=record_stride,
        checkpoint_block=checkpoint_block,
        description={"family": "scenario", "case": case.to_dict()},
    )
