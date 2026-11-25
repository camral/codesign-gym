"""Immutable spring geometry, integration, contact, and task configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..numerics.selection import resolve_integrator

if TYPE_CHECKING:
    from .terrain import TerrainConfig


@dataclass(frozen=True)
class SpringConfig:
    """Immutable spring geometry, integration, contact, and objective settings."""

    grid_shape: tuple[int, int] = (4, 2)
    size: tuple[float, float] = (0.2, 0.1)
    thickness: float = 0.01
    control_dt: float = 0.02
    substeps: int = 10
    settle_steps: int = 5  # 0.10 s in M1; not a claim of static equilibrium.
    horizon: int = 50
    integrator: str = "backward_euler"
    linear_solver: str = "dense"  # explicit M1 reference or M2 gmres; no hidden switch
    gmres_method: str = "incremental"
    gmres_preconditioner: str = "diagonal"
    gmres_refinement_steps: int = 2
    gravity: float = 9.81
    ground: bool = True
    ground_height: float = 0.0
    initial_clearance: float = 0.002
    damping_ratio: float = 0.08
    friction: float = 0.6
    friction_velocity: float = 0.01
    contact_modulus: float = 200000.0  # pressure-per-penetration = modulus / Lref
    contact_width: float = 0.0005
    actuator_tau: float = 0.05
    max_log_stretch: float = 0.2
    newton_max_steps: int = 20
    line_search_steps: int = 8
    root_tol64: float = 2e-10
    root_tol32: float = 4e-06
    min_length_ratio: float = 0.1
    min_area_ratio: float = 0.03
    max_dense_dofs: int = 128  # enforced M1 limit, not a scalable solver.
    length_ref: float = 0.2
    velocity_ref: float = 1.0
    work_ref: float = 4.0
    energy_weight: float = 1.0
    terrain: TerrainConfig | None = (
        None  # optional TerrainConfig; None preserves the M1 flat-plane path
    )
    # Explicit matrix-free settings; defaults preserve the former NewtonConfig defaults.
    gmres_restart: int = 20
    gmres_max_restarts: int = 12
    linear_rtol64: float = 2e-10
    linear_rtol32: float = 2e-5
    armijo: float = 0.0001
    gmres_linear_work_reuse: bool = False
    spring_cache_tangent: bool = False

    def __post_init__(self) -> None:
        """Validate geometry, timestep, contact, and solver settings at construction."""
        if type(self.spring_cache_tangent) is not bool:
            raise ValueError("spring_cache_tangent must be boolean")
        if self.terrain is not None:
            from .terrain import TerrainConfig

            if not isinstance(self.terrain, TerrainConfig):
                raise ValueError("terrain must be TerrainConfig or None")
        if len(self.grid_shape) != 2 or any(type(n) is not int or n < 1 for n in self.grid_shape):
            raise ValueError("grid_shape must contain two positive Python integers")
        if len(self.size) != 2:
            raise ValueError("size must have length two")
        positive = (
            *self.size,
            self.thickness,
            self.control_dt,
            self.friction_velocity,
            self.contact_width,
            self.actuator_tau,
            self.length_ref,
            self.velocity_ref,
            self.work_ref,
            self.root_tol32,
            self.root_tol64,
            self.max_log_stretch,
            self.linear_rtol64,
            self.linear_rtol32,
        )
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError("physical scales and tolerances must be finite and positive")
        for n in (self.substeps, self.horizon, self.newton_max_steps, self.line_search_steps):
            if type(n) is not int or n < 1:
                raise ValueError("iteration counts must be positive integers")
        if type(self.settle_steps) is not int or self.settle_steps < 0:
            raise ValueError("settle_steps must be nonnegative")
        if self.integrator not in ("backward_euler", "kvaerno3", "kvaerno3_native"):
            raise ValueError("Unknown integrator")
        if self.linear_solver not in ("dense", "gmres"):
            raise ValueError("linear_solver must be dense or gmres")
        if any(type(n) is not int or n < 1 for n in (self.gmres_restart, self.gmres_max_restarts)):
            raise ValueError("GMRES restart capacities must be positive integers")
        if not math.isfinite(self.armijo) or not 0 < self.armijo < 1:
            raise ValueError("armijo must be finite and in (0, 1)")
        if self.gmres_preconditioner not in ("none", "diagonal"):
            raise ValueError("gmres_preconditioner must be none or diagonal")
        if (
            type(self.gmres_refinement_steps) is not int
            or not 0 <= self.gmres_refinement_steps <= 3
        ):
            raise ValueError("gmres_refinement_steps must be an integer in 0..3")
        if self.gmres_method not in ("incremental", "batched", "fused"):
            raise ValueError("Unknown gmres_method")
        if type(self.gmres_linear_work_reuse) is not bool:
            raise ValueError("gmres_linear_work_reuse must be boolean")
        if self.gmres_linear_work_reuse and self.gmres_method != "fused":
            raise ValueError("gmres_linear_work_reuse requires gmres_method='fused'")
        if self.linear_solver == "dense" and self.gmres_method != "incremental":
            raise ValueError("Nondefault gmres_method requires linear_solver='gmres'")
        if self.integrator == "kvaerno3" and self.linear_solver != "dense":
            raise ValueError(
                "Unmodified Diffrax stage adjoint is dense; use kvaerno3_native for matrix-free"
            )
        if self.integrator == "kvaerno3_native" and self.linear_solver != "gmres":
            raise ValueError("kvaerno3_native requires explicit linear_solver=gmres")
        if any(
            not math.isfinite(v) or v < 0
            for v in (
                self.gravity,
                self.damping_ratio,
                self.friction,
                self.contact_modulus,
                self.initial_clearance,
                self.energy_weight,
            )
        ):
            raise ValueError("gravity, damping, contact, clearance and weight must be nonnegative")
        if not math.isfinite(self.ground_height):
            raise ValueError("ground_height must be finite")
        if not (0 < self.min_length_ratio < 1 and 0 < self.min_area_ratio < 1):
            raise ValueError("admissibility thresholds must be in (0, 1)")
        if not (1 <= self.max_dense_dofs <= 128):
            raise ValueError("M1 dense displacement/velocity budget must be <=128")
        ndof = 2 * (self.grid_shape[0] + 1) * (self.grid_shape[1] + 1)
        if self.linear_solver == "dense" and ndof > self.max_dense_dofs:
            raise ValueError(
                f"M1 dense solver capacity exceeded: {ndof}>{self.max_dense_dofs}; use M2 scalable solvers later"
            )

    @property
    def dt(self) -> float:
        """Nominal physics timestep in seconds, independent of the control interval."""
        return self.control_dt / self.substeps


def with_integrator(config: SpringConfig | None, integrator: str | None) -> SpringConfig:
    """Choose an existing JAX spring integrator without mutating caller settings.

    An omitted configuration uses the historical default, except that an
    explicit native Kvaerno request constructs its required GMRES configuration.
    A supplied configuration retains every solver/tolerance value; incompatible
    choices raise its existing validation error. No override preserves object
    identity, which also preserves existing initialization/cache behavior.
    """
    name = resolve_integrator(
        integrator,
        backend="jax",
        physics="spring",
        configured=config.integrator if config is not None else "backward_euler",
    )
    if config is None:
        solver = "gmres" if name == "kvaerno3_native" else "dense"
        return SpringConfig(integrator=name, linear_solver=solver)
    return config if config.integrator == name else replace(config, integrator=name)
