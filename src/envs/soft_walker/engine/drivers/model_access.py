"""Shared read-only access to heterogeneous body model/state records."""

from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

from ..physics import beam, fem, mpm, spring
from ..types import BodyModel, DriverState, PhysicsModel, PhysicsState
from .task import TaskState

if TYPE_CHECKING:
    from .spring_environment import Spring2DEnv


def reference(model: PhysicsModel, kind: str | None = None) -> jax.Array:
    """Read reference coordinates, inferring the family when kind is omitted.

    The explicit kind argument preserves historical body-driver calls; resolved
    bundles also include spring models and can dispatch on their record types.
    """
    if isinstance(model, spring.Model):
        return model.reference
    if kind == "fem" or (kind is None and isinstance(model, fem.Model)):
        return cast(fem.Model, model).mesh.reference
    if kind == "beam" or (kind is None and isinstance(model, beam.Model)):
        return cast(beam.Model, model).topology.reference
    return cast(mpm.Model, model).cloud.reference


def positions(model: PhysicsModel, state: PhysicsState, kind: str | None = None) -> jax.Array:
    """Return world coordinates from displacement or particle-position state."""
    absolute = (
        isinstance(state, (spring.State, mpm.State))
        if kind is None
        else kind == "spring" or kind.startswith("mpm")
    )
    return (
        cast(spring.State | mpm.State, state).x
        if absolute
        else reference(model, kind) + cast(beam.State | fem.State, state).u
    )


def module(kind: str) -> ModuleType:
    """Select the numerical physics module for an already validated driver kind."""
    return mpm if kind.startswith("mpm") else (fem if kind == "fem" else beam)


def channel_mask(model: BodyModel) -> jax.Array:
    """Return the channels connected to occupied material or active beam links."""
    if hasattr(model, "strain_map"):
        model = cast(beam.Model, model)
        links = (model.alpha[model.topology.edges[:, 0]] > 0) & (
            model.alpha[model.topology.edges[:, 1]] > 0
        )
        return jnp.any((jnp.abs(model.strain_map) > 0) & links[:, None], axis=0)
    return jnp.any(
        (model.alpha * model.strength)[:, None]
        * jax.nn.one_hot(model.channel, len(model.channels), dtype=model.alpha.dtype)
        > 0,
        axis=0,
    )


def physical_state(env: Spring2DEnv, state: DriverState) -> PhysicsState:
    """Unwrap task ledgers to obtain the underlying physical body state."""
    return cast(TaskState, state).body if hasattr(env, "kind") else cast(spring.State, state)


def task_positions(env: Spring2DEnv, model: PhysicsModel, state: DriverState) -> jax.Array:
    """Read current world positions in metres without advancing the simulation."""
    if hasattr(env, "kind"):
        return positions(cast(BodyModel, model), cast(TaskState, state).body, env.kind)
    return cast(spring.State, state).x


def task_reference_positions(env: Spring2DEnv, model: PhysicsModel) -> jax.Array:
    """Read immutable material reference positions from the selected driver."""
    if hasattr(env, "kind"):
        return reference(cast(BodyModel, model), env.kind)
    return cast(spring.Model, model).reference
