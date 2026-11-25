"""Scenario settings and objective/reward semantics, independent of the optimizer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..physics.common import DynamicsConfig
from ..physics.contact import PlaneContact
from ..physics.terrain import TerrainConfig
from ..types import BodyState, PhysicsModel


@dataclass(frozen=True)
class BodyConfig:
    """Static discretization, control timing, and task scales for beam/FEM/MPM drivers.

    Lengths are metres and times seconds. Each control transition integrates
    ``substeps`` steps of ``control_dt / substeps``; ``horizon`` counts control
    transitions. Construction rejects incompatible dimensions and physical scales.
    """

    grid_shape: tuple[int, ...] = (2, 1, 1)
    size: tuple[float, ...] = (0.1, 0.05, 0.05)
    thickness: float = 0.01
    control_dt: float = 0.02
    substeps: int = 80  # .25 ms trial profile, not an accuracy certification
    settle_steps: int = 5
    horizon: int = 250
    initial_clearance: float = 0.001
    particles_per_axis: int = 2
    grid_margin: int = 2
    grid_padding: tuple[tuple[int, int], ...] | None = None
    observation_profile: str = "full"
    length_ref: float = 0.2
    velocity_ref: float = 1.0
    work_ref: float = 4.0
    energy_weight: float = 1.0
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    dynamics: DynamicsConfig = field(
        default_factory=lambda: DynamicsConfig(contact=PlaneContact(enabled=True), max_halvings=0)
    )

    def __post_init__(self) -> None:
        """Validate discretization, control horizon, and consistent physical scales."""
        dim = len(self.grid_shape)
        if (
            dim not in (2, 3)
            or len(self.size) != dim
            or any(type(x) is not int or x < 1 for x in self.grid_shape)
        ):
            raise ValueError("grid_shape and size must be matching positive 2D/3D tuples")
        vals = (
            *self.size,
            self.thickness,
            self.control_dt,
            self.length_ref,
            self.velocity_ref,
            self.work_ref,
        )
        if not all(math.isfinite(x) and x > 0 for x in vals):
            raise ValueError("scales must be finite and positive")
        if (
            not math.isfinite(self.initial_clearance)
            or self.initial_clearance < 0
            or not math.isfinite(self.energy_weight)
            or self.energy_weight < 0
        ):
            raise ValueError("invalid clearance or energy weight")
        for k in ("substeps", "horizon", "particles_per_axis", "grid_margin"):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError(k + " must be a positive integer")
        if type(self.settle_steps) is not int or self.settle_steps < 0:
            raise ValueError("settle_steps must be nonnegative")
        if self.grid_padding is not None:
            if len(self.grid_padding) != dim or any(
                len(p) != 2 or any(type(v) is not int or v < 2 for v in p)
                for p in self.grid_padding
            ):
                raise ValueError("grid_padding needs >=2 lower/upper cells per axis")
        if self.observation_profile not in ("compact", "full"):
            raise ValueError("observation_profile must be compact or full")
        if self.dynamics.vertical_axis != 1 or self.dynamics.contact.vertical_axis != 1:
            raise ValueError("M3 drivers use +y vertical")
        if self.dynamics.velocity_ref != self.velocity_ref:
            raise ValueError("driver and dynamics velocity scales must match")

    @property
    def dt(self) -> float:
        """Nominal physics timestep in seconds."""
        return self.control_dt / self.substeps


class TaskState(NamedTuple):
    """Immutable JAX task pytree wrapping the selected driver's physical state.

    Displacement and initial center of mass have shape ``(dim,)`` in metres.
    Effort accumulates squared commands over seconds; scalar status is zero for
    accepted states. Array placement follows the wrapped body state.
    """

    body: BodyState
    displacement: jax.Array
    effort: jax.Array
    initial_com: jax.Array
    status: jax.Array

    @property
    def v(self) -> jax.Array:
        """Read the physical body velocity without changing its shape."""
        return self.body.v

    @property
    def time(self) -> jax.Array:
        """Read the committed physical time in seconds."""
        return self.body.time


def mean(model: PhysicsModel, array: jax.Array) -> jax.Array:
    """Compute a mass-weighted mean while masking inactive reference sites."""
    a = jnp.where(model.active.reshape(model.active.shape + (1,) * (array.ndim - 1)), array, 0.0)
    return jnp.einsum("n,n...->...", model.mass, a) / jnp.where(
        jnp.sum(model.mass) > 0, jnp.sum(model.mass), 1.0
    )


def metrics(state: TaskState, cfg: BodyConfig) -> dict[str, jax.Array]:
    """Return physical task measures, validity, and normalized objectives."""
    valid = (
        (state.status == 0)
        & jnp.all(jnp.isfinite(state.displacement))
        & jnp.isfinite(state.body.work_positive)
    )
    distance = state.displacement[0]
    vector = jnp.stack((-distance / cfg.length_ref, state.body.work_positive / cfg.work_ref))
    scalar = vector[0] + cfg.energy_weight * vector[1]
    return {
        "distance_m": distance,
        "work_positive_j": state.body.work_positive,
        "work_signed_j": state.body.work_signed,
        "effort_s": state.effort,
        "objective_vector": jnp.where(valid, vector, jnp.nan),
        "objective": jnp.where(valid, vector, jnp.nan),
        "scalar_objective": jnp.where(valid, scalar, jnp.nan),
        "valid": valid,
        "dissipated_j": state.body.dissipated,
        "time_s": state.body.time,
        "status": state.status,
    }


def reward(before: TaskState, after: TaskState, cfg: BodyConfig) -> jax.Array:
    """Return accepted forward displacement minus positive work for one transition."""
    r = (after.displacement[0] - before.displacement[0]) / cfg.length_ref - cfg.energy_weight * (
        after.body.work_positive - before.body.work_positive
    ) / cfg.work_ref
    return jnp.where(after.status == 0, r, jnp.asarray(0.0, r.dtype))
