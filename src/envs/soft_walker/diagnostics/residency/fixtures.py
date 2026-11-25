"""Dimensionless inputs for the inherited compiler/capture diagnostic probes.

These are small synthetic problems, not physical defaults or solver-wide policy.
Their values preserve the original capture regression fixtures; no empirical
calibration record is available. NumPy and JAX share these input data while their
reference algorithms remain independent.
"""

from dataclasses import dataclass

INITIAL_PROBE_INPUT = 0.23  # Nonzero scalar, away from the control loop's integer boundaries.


@dataclass(frozen=True)
class ControlFixture:
    """Nested data-dependent loops with alternating inner counts and two branches."""

    base_outer_steps: int = 3
    input_count_scale: int = 4
    vector_start: float = -0.4
    vector_stop: float = 0.6
    dimension: int = 8
    base_inner_steps: int = 2
    inner_count_period: int = 2
    high_decay: float = 0.8
    high_sine_scale: float = 0.01
    low_decay: float = 0.7
    low_cosine_scale: float = 0.02


@dataclass(frozen=True)
class BacksolveFixture:
    """Small upper-triangular system with positive diagonal and nonzero coupling."""

    dimension: int = 8
    upper_triangle_divisor: float = 300.0
    diagonal_start: float = 1.0
    diagonal_stop: float = 2.0


@dataclass(frozen=True)
class ImplicitGradientFixture:
    """Repeated coupled linear solves testing both primal and implicit derivatives.

    The restart budget and precision tolerances belong to this diagnostic only.
    They preserve its original acceptance workload, rather than claiming to be
    minimal or optimal settings for a physical simulation.
    """

    dimension: int = 8
    diagonal_start: float = 2.0
    diagonal_stop: float = 3.0
    dense_coupling: float = 0.04
    variation_start: float = 0.1
    variation_stop: float = 0.3
    superdiagonal_variation: float = 0.02
    recurrent_weight: float = 0.1
    recurrence_steps: int = 3
    krylov_restarts: int = 4
    linear_rtol64: float = 1e-12
    linear_rtol32: float = 1e-6


CONTROL_PROBE = ControlFixture()
BACKSOLVE_PROBE = BacksolveFixture()
IMPLICIT_GRADIENT_PROBE = ImplicitGradientFixture()
