"""Action-hold configuration without changing the physics timestep.

An engine horizon always counts policy actions. Holding each action longer
therefore extends the simulated horizon; reset settling retains its original
configuration and duration. No solver or timestep tolerance is changed here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .drivers.task import BodyConfig
    from .physics.spring_config import SpringConfig

from dataclasses import replace

ConfigT = TypeVar("ConfigT", "SpringConfig", "BodyConfig")


def positive_step_count(value: object, name: str = "physics_steps_per_action") -> int:
    """Validate a plain positive integer without silently accepting bools."""
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive Python integer")
    return value


def with_action_hold(config: ConfigT, physics_steps_per_action: int | None = None) -> ConfigT:
    """Return a JAX driver configuration with fixed nominal ``config.dt``.

    Callers retain ``config`` for reset initialization: its ``settle_steps``
    count the original control intervals, independently of the new action hold.
    """
    if physics_steps_per_action is None:
        return config
    count = positive_step_count(physics_steps_per_action)
    if count == config.substeps:
        return config
    return replace(config, substeps=count, control_dt=config.dt * count)


def native_action_hold(
    substeps: int | None = None, physics_steps_per_action: int | None = None
) -> int:
    """Resolve the native legacy substeps name and explicit action-hold name."""
    if substeps is not None:
        positive_step_count(substeps, "substeps")
    if physics_steps_per_action is not None:
        positive_step_count(physics_steps_per_action)
        if substeps is not None and substeps != physics_steps_per_action:
            raise ValueError(
                "substeps and physics_steps_per_action must agree when both are supplied"
            )
        return physics_steps_per_action
    return 1 if substeps is None else substeps
