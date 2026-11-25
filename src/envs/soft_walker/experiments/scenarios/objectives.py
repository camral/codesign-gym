"""Shared traceable shaping loss for the existing scenario training paths.

The inherited coefficients are experiment heuristics, with no recorded fitting
procedure. They are separate from physical rewards and success/flight criteria.
Changing them is an objective change, not a numerical refactor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax

from envs.soft_walker.contracts.metrics import scenario_metric_layout

if TYPE_CHECKING:
    from envs.soft_walker.presets.scenarios.config import Scenario


@dataclass(frozen=True)
class ScenarioObjective:
    """Dimensionless weights and SI clearance scales of the historical loss."""

    clearance_gain: float = 0.05  # Dimensionless softplus output scaling.
    target_clearance_m: float = 0.045
    clearance_transition_m: float = 0.005
    collapse_weight: float = 0.25
    height_reward_weight: float = 0.15
    positive_work_weight: float = 0.001


DEFAULT_SCENARIO_OBJECTIVE = ScenarioObjective()


def running_cost(
    row: jax.Array,
    case: Scenario,
    zero: float | jax.Array = 0.0,
    objective: ScenarioObjective = DEFAULT_SCENARIO_OBJECTIVE,
) -> jax.Array:
    """Return one interval's normalized clearance/vertical-displacement cost.

    ``row`` uses the saved scenario metric layout. ``zero`` preserves the calling
    path's existing scalar precision for non-hopping gaits. This pure helper
    performs no state validity checks or I/O; callers retain those transactions.
    """
    layout = scenario_metric_layout(case.dim)
    shortage = objective.clearance_gain * jax.nn.softplus(
        (objective.target_clearance_m - row[layout.torso_center_gap])
        / objective.clearance_transition_m
    )
    reward_height = row[layout.dy] / case.size[1] if case.gait in ("bound", "pronk") else zero
    return (
        objective.collapse_weight * shortage**2 - objective.height_reward_weight * reward_height
    ) / case.steps


def terminal_cost(
    row: jax.Array,
    case: Scenario,
    work_reference_j: float,
    accumulated_cost: jax.Array,
    objective: ScenarioObjective = DEFAULT_SCENARIO_OBJECTIVE,
) -> jax.Array:
    """Combine forward displacement, positive work and already accumulated shaping.

    Positions are metres and work is joules; normalization produces a
    dimensionless scalar. Finite-state and final-time validation stay with the
    caller, preserving ordinary versus resident episode failure handling.
    """
    layout = scenario_metric_layout(case.dim)
    return (
        -row[layout.dx] / case.size[0]
        + objective.positive_work_weight * row[layout.work_positive] / work_reference_j
        + accumulated_cost
    )
