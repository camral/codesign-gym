"""JAX-independent configuration for the native MPM numerical slice."""

import math
from dataclasses import asdict, dataclass

from ._solver_limits import validate_adaptive_restart


@dataclass(frozen=True)
class MPMConfig:
    """Fixed native MPM integration and solver parameters.

    Distances are metres, times seconds, and contact stiffness is N/m³.
    Grid topology and iteration capacities become graph constants. Invalid
    dimensions, scales or unsupported solver combinations raise at construction;
    no clipping or automatic solver substitution occurs. The dynamic adaptive
    solver's effective restart is checked against its implementation capacity
    when the bundle supplies its resolved degrees of freedom.
    """

    grid_shape: tuple[int, ...] = (8, 8)

    origin: tuple[float, ...] = (-0.05, -0.05)

    spacing: float = 0.025

    dt: float = 0.0005

    gravity: float = 9.81

    vertical_axis: int = 1

    velocity_ref: float = 1.0

    actuator_tau: float = 0.05

    max_log_stretch: float = 0.2

    actuation_model: str = "active_strain"

    active_stress_ratio: float = 0.4

    mass_weighted: bool = False

    parameter_adjoint: str = "mass"  # equivalent residual without predictor normalization

    min_J: float = 0.1

    contact_enabled: bool = False

    contact_height: float = 0.0

    contact_stiffness: float = 1e6

    contact_width: float = 0.0005

    friction: float = 0.6

    friction_width: float = 0.01

    max_newton: int = 16

    max_backtracks: int = 10

    krylov_method: str = "adaptive"

    cache_tangent: bool = False

    compact_nodes: int = 0  # zero preserves conservative support capacity

    linear_work_reuse: bool = False  # checked zero start / unchanged residual reuse

    block_preconditioner: bool = False

    restart: int = 20

    max_restarts: int = 12

    linear_refinements: int = 0

    armijo: float = 0.0001

    atol64: float = 2e-11

    atol32: float = 3e-6

    linear_rtol64: float = 2e-10

    linear_rtol32: float = 2e-5

    def __post_init__(self) -> "None":
        """Reject unsupported solver combinations and nonfinite physical scales."""

        dim = len(self.grid_shape)

        if dim not in (2, 3) or len(self.origin) != dim:
            raise ValueError("Grid and origin must have dimension 2 or 3")

        if any(type(v) is not int or v < 4 for v in self.grid_shape):
            raise ValueError("Grid extents must be integer node counts >= 4")

        if math.prod(self.grid_shape) >= 2**31 - 1:
            raise ValueError("Grid addressing exceeds int32 capacity")

        numeric = [v for v in asdict(self).values() if isinstance(v, (int, float))]

        if not all(math.isfinite(v) for v in numeric + list(self.origin)):
            raise ValueError("Nonfinite numerical parameter")

        if not 0 <= self.vertical_axis < dim:
            raise ValueError("Vertical axis is outside the spatial dimension")

        if self.parameter_adjoint not in ("mass", "velocity"):
            raise ValueError("parameter_adjoint must be mass or velocity")

        if type(self.compact_nodes) is not int or self.compact_nodes < 0:
            raise ValueError("compact_nodes must be a nonnegative integer")

        if type(self.linear_work_reuse) is not bool or type(self.block_preconditioner) is not bool:
            raise ValueError("Solver optimization choices must be booleans")

        if self.block_preconditioner and not self.cache_tangent:
            raise ValueError("Block preconditioning requires cache_tangent=True")

        if type(self.cache_tangent) is not bool:
            raise ValueError("cache_tangent must be a boolean")

        if self.krylov_method not in ("legacy", "adaptive", "adaptive_dynamic"):
            raise ValueError("krylov_method must be legacy, adaptive, or adaptive_dynamic")

        if self.actuation_model not in ("active_strain", "active_stress"):
            raise ValueError("Unsupported actuation law")

        if self.actuator_tau < 0 or (
            self.actuator_tau == 0 and self.actuation_model != "active_stress"
        ):
            raise ValueError("Zero actuator_tau is supported only for active stress")

        if not 0 <= self.active_stress_ratio < 1 or not 0 < self.armijo < 1:
            raise ValueError("Invalid actuation or line-search bound")

        positive = (
            self.spacing,
            self.dt,
            self.velocity_ref,
            self.min_J,
            self.max_log_stretch,
            self.contact_width,
            self.friction_width,
            self.atol64,
            self.atol32,
            self.linear_rtol64,
            self.linear_rtol32,
        )

        if min(positive) <= 0 or min(self.contact_stiffness, self.friction) < 0:
            raise ValueError("Invalid physical scale or solver tolerance")

        if type(self.linear_refinements) is not int or not 0 <= self.linear_refinements <= 3:
            raise ValueError("linear_refinements must be in 0..3")

        for v in (self.max_newton, self.max_backtracks, self.restart, self.max_restarts):
            if type(v) is not int or v < 1:
                raise ValueError("Iteration capacities must be positive integers")

    def validate_solver_capacity(self, scalar_dofs: int) -> None:
        """Check the selected implementation after resolving the DOF clamp.

        A requested restart above the implementation limit remains valid when
        the existing DOF clamp reduces the effective restart below that limit.
        Legacy and adaptive-capturable GMRES do not share this restriction.
        """
        if self.krylov_method == "adaptive_dynamic":
            validate_adaptive_restart(self.restart, scalar_dofs)

    @property
    def dim(self) -> "int":
        """Spatial dimension encoded by the background-grid extents."""

        return len(self.grid_shape)

    def to_dict(self) -> "dict[str, float | int | str | tuple[int, ...] | tuple[float, ...]]":
        """Return serializable configuration values without changing tuple fields."""

        return asdict(self)
