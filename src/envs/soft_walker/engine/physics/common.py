"""Small immutable mechanics profiles and shared step diagnostics."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from numbers import Integral
from typing import NamedTuple, TypeVar

import jax
import jax.numpy as jnp

from ...contracts.status_codes import (
    BINARY_STATE_FAILED,
    BINARY_STATE_OK,
    ROOT_OK,
    STEP_INVALID_GEOMETRY,
    STEP_INVALID_SUPPORT,
    STEP_REJECTED,
)
from ..numerics.solvers import NewtonConfig, RootInfo
from ..types import DriverState, PhysicsState
from .contact import PlaneContact

StateT = TypeVar("StateT", bound="DriverState | PhysicsState")


@dataclass(frozen=True)
class DynamicsConfig:
    """Shared physical actuation, contact, integration, and solve configuration.

    Optional MPM compaction bounds grid buffers by ``mpm_compact_nodes``; zero
    retains full-grid buffers, and insufficient capacity rejects the trial.
    Local tangent caches retain exact constitutive/element derivatives within
    each linearization. Node-block preconditioning is MPM-only and can freeze
    blocks at the predictor or rebuild them at each Newton iterate and final
    adjoint root. All numerical-work options default off and preserve the
    selected equation, material/control derivatives, and acceptance tolerances.
    """

    gravity: float = 9.81
    vertical_axis: int = 1
    velocity_ref: float = 1.0
    actuator_tau: float = 0.05
    max_log_stretch: float = 0.2
    min_J: float = 0.1
    contact: PlaneContact = field(default_factory=PlaneContact)
    newton: NewtonConfig = field(default_factory=NewtonConfig)
    max_halvings: int = 2
    mpm_mass_weighted: bool = False  # equivalent root in kinetic-energy-scaled coordinates
    actuation_model: str = "active_strain"
    active_stress_ratio: float = 0.4  # stress amplitude / passive shear modulus
    mpm_compact_nodes: int = 0  # zero retains the full background-grid solver buffers
    mpm_cache_tangent: bool = False
    mpm_block_preconditioner: bool = False
    mpm_preconditioner_update: str = "predictor"
    fem_cache_tangent: bool = False
    beam_cache_tangent: bool = False

    def __post_init__(self) -> None:
        """Validate timestep retry, contact, and actuator-model settings."""
        if not math.isfinite(self.gravity):
            raise ValueError("gravity must be finite")
        scales = (self.velocity_ref, self.max_log_stretch, self.min_J)
        if not all(math.isfinite(value) for value in scales) or min(scales) <= 0:
            raise ValueError("Physical scales/admissibility must be finite and positive")
        if self.actuation_model not in ("active_strain", "active_stress"):
            raise ValueError("Unknown continuum actuation model")
        if (
            not math.isfinite(self.actuator_tau)
            or self.actuator_tau < 0
            or (self.actuator_tau == 0 and self.actuation_model != "active_stress")
        ):
            raise ValueError("actuator_tau must be positive, or zero for direct active stress")
        if not math.isfinite(self.active_stress_ratio) or not 0 <= self.active_stress_ratio < 1:
            raise ValueError("active_stress_ratio must be finite and in [0, 1)")
        if (
            isinstance(self.max_halvings, bool)
            or not isinstance(self.max_halvings, Integral)
            or not 0 <= self.max_halvings <= 4
        ):
            raise ValueError("max_halvings must be an integer in 0..4")
        if type(self.mpm_compact_nodes) is not int or self.mpm_compact_nodes < 0:
            raise ValueError("mpm_compact_nodes must be a nonnegative integer")
        if type(self.mpm_cache_tangent) is not bool:
            raise ValueError("mpm_cache_tangent must be boolean")
        if type(self.mpm_block_preconditioner) is not bool:
            raise ValueError("mpm_block_preconditioner must be boolean")
        if self.mpm_preconditioner_update not in ("predictor", "newton"):
            raise ValueError("mpm_preconditioner_update must be predictor or newton")
        if type(self.fem_cache_tangent) is not bool or type(self.beam_cache_tangent) is not bool:
            raise ValueError("Element tangent-cache options must be boolean")


class StepInfo(NamedTuple):
    """Device scalar diagnostics from accepted and rejected physical trials."""

    accepted: jax.Array
    rejected: jax.Array
    nonlinear_iterations: jax.Array
    linear_solves: jax.Array
    max_residual: jax.Array
    max_linear_residual: jax.Array
    line_search_backtracks: jax.Array
    min_J: jax.Array
    support_valid: jax.Array
    max_penetration: jax.Array
    code: jax.Array


def info_from_root(
    root: RootInfo, minJ: jax.Array, support: jax.Array, penetration: jax.Array, ok: jax.Array
) -> StepInfo:
    """Translate nonlinear diagnostics and geometric checks into a step report."""
    d = root.residual.dtype
    code = jnp.where(
        ~support,
        float(STEP_INVALID_SUPPORT),
        jnp.where(minJ <= 0, float(STEP_INVALID_GEOMETRY), root.code),
    )
    code = jnp.where((~ok) & (code == ROOT_OK), float(STEP_REJECTED), code)
    return StepInfo(
        ok.astype(d),
        (~ok).astype(d),
        root.iterations,
        root.linear_solves,
        root.residual,
        root.linear_residual,
        root.backtracks,
        minJ,
        support.astype(d),
        penetration,
        code,
    )


def combine_info(a: StepInfo, b: StepInfo) -> StepInfo:
    """Combine committed retry diagnostics while retaining extrema and counters."""
    return StepInfo(
        a.accepted + b.accepted,
        a.rejected + b.rejected,
        a.nonlinear_iterations + b.nonlinear_iterations,
        a.linear_solves + b.linear_solves,
        jnp.maximum(a.max_residual, b.max_residual),
        jnp.maximum(a.max_linear_residual, b.max_linear_residual),
        a.line_search_backtracks + b.line_search_backtracks,
        jnp.minimum(a.min_J, b.min_J),
        jnp.minimum(a.support_valid, b.support_valid),
        jnp.maximum(a.max_penetration, b.max_penetration),
        jnp.maximum(a.code, b.code),
    )


def advance_retry(
    trial: Callable[[StateT, jax.Array, jax.Array], tuple[StateT, StepInfo]],
    state: StateT,
    command: jax.Array,
    dt: jax.Array,
    cfg: DynamicsConfig,
) -> tuple[StateT, StepInfo, jax.Array]:
    """Transactional substep bisection, fixed replay capacity, no silent repair.

    trial(state, command, dt) returns (state, diagnostics). A failed parent trial
    is discarded. If either half ultimately fails, the WHOLE requested interval
    rolls back. dt_trace records accepted leaves; failed intervals return zeros.
    Discrete branch choices are differentiated piecewise, never smoothed away.
    """

    def recurse(s: StateT, h: jax.Array, depth: int) -> tuple[StateT, StepInfo, jax.Array]:
        """Attempt an interval, recursively bisecting failed trials within the fixed depth budget."""
        out, info = trial(s, command, h)
        length = 2**depth
        trace = jnp.zeros(length, s.v.dtype).at[0].set(h)
        if depth == 0:
            ok = out.status == BINARY_STATE_OK
            return out, info, jnp.where(ok, trace, 0.0)

        def split(_: None) -> tuple[StateT, StepInfo, jax.Array]:
            """Evaluate both half intervals and roll the full interval back if either ultimately
            fails.
            """
            a, ia, ta = recurse(s, h / 2, depth - 1)
            b, ib, tb = recurse(a, h / 2, depth - 1)
            good = (a.status == BINARY_STATE_OK) & (b.status == BINARY_STATE_OK)
            info2 = combine_info(ia, ib)
            # Include failed parent work in cost diagnostics; terminal code is
            # successful if recovery succeeded. No failed state is committed.
            info2 = info2._replace(
                rejected=info2.rejected + info.rejected,
                nonlinear_iterations=info2.nonlinear_iterations + info.nonlinear_iterations,
                linear_solves=info2.linear_solves + info.linear_solves,
                line_search_backtracks=info2.line_search_backtracks + info.line_search_backtracks,
                code=jnp.where(good, float(ROOT_OK), jnp.maximum(ia.code, ib.code)),
            )
            restored = jax.tree.map(lambda n, o: jnp.where(good, n, o), b, s)._replace(
                status=jnp.where(good, BINARY_STATE_OK, BINARY_STATE_FAILED).astype(jnp.int32)
            )
            return restored, info2, jnp.where(good, jnp.concatenate((ta, tb)), 0.0)

        return jax.lax.cond(
            out.status == BINARY_STATE_OK, lambda _: (out, info, trace), split, operand=None
        )

    return recurse(state, dt, cfg.max_halvings)


def surface_exposure(
    alpha: jax.Array, neighbor: jax.Array, has_neighbor: jax.Array, smoothing: float = 0.0
) -> jax.Array:
    """Compute exposed reference-area weights from neighbouring occupancy."""
    other = jnp.where(has_neighbor, alpha[neighbor], 0.0)
    p = jnp.maximum(alpha[:, None] - other, 0.0)
    # Unlike alpha*(1-alpha_neighbor), uniform gray interiors do not create
    # O(1/h) fictitious surface area under refinement.
    if smoothing > 0:
        return p * p / (p + smoothing)
    return p
