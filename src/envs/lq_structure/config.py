"""Per-setup configuration for the LQ structure co-design environment.

Three setups are provided, mirroring the README:
  - ``lqr``: continuous design parameters, full-state feedback, no measurement noise.
  - ``lqg``: actuator/sensor placement design, partial noisy observation.
  - ``qp`` : actuator saturation under a mass/authority budget.

A config is a plain dataclass describing the plant size, integration, cost, noise,
design bounds, and budget/saturation limits. The environment reads it to build spaces,
dynamics, rewards, and validation rules. Nothing here depends on the codesign base API;
it is deliberately self-contained so the env runs on plain ``gymnasium``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

ControllerType = Literal["lqr", "lqg", "qp"]


@dataclass
class LQConfig:
    """Configuration for a single LQ-structure setup.

    Attributes are grouped as: plant size, integration, cost weights, noise,
    design bounds, and budget/saturation. All matrices are stored densely and are
    sized against ``n`` (number of masses); the state dimension is ``2 * n``.
    """

    # --- identity ---
    controller: ControllerType = "lqr"

    # --- plant size / integration ---
    n: int = 6
    dt: float = 0.05
    horizon: int = 200

    # --- observation ---
    # "full": clean full state is measured (LQR). "partial": only sensed masses are
    # measured, corrupted by measurement noise (LQG). Placement still lives in the
    # embodiment; "full" simply ignores the sensor mask when forming the measurement.
    observation_mode: Literal["full", "partial"] = "full"

    # --- cost weights (per-state / per-input); scalars are broadcast to diagonals ---
    q_pos: float = 1.0  # weight on positions
    q_vel: float = 1.0  # weight on velocities
    r_act: float = 1.0  # weight on control effort

    # --- noise (diagonal covariances) ---
    process_noise_std: float = 0.0  # std of w added to velocities each step
    measurement_noise_std: float = 0.0  # std of measurement noise (partial obs)

    # --- initial condition ---
    init_pos_std: float = 0.5
    init_vel_std: float = 0.0
    # "random": gaussian x0. "impulse": zero state, unit velocity kick on mass 0.
    reset_mode: Literal["random", "impulse"] = "random"

    # --- design bounds (per element, applied to every mass/link) ---
    mass_low: float = 0.5
    mass_high: float = 2.0
    stiffness_low: float = 0.5
    stiffness_high: float = 5.0
    damping_low: float = 0.05
    damping_high: float = 1.0

    # --- placement design ---
    # If False, the corresponding mask is frozen to all-ones (fully actuated / sensed)
    # and excluded from the optimizable embodiment (used by the LQR setup).
    design_actuators: bool = False
    design_sensors: bool = False

    # --- budget / saturation (used by the qp setup; None disables) ---
    mass_budget: float | None = None       # sum(masses) <= mass_budget
    actuator_budget: int | None = None     # sum(actuators) <= actuator_budget
    u_max: float | None = None             # action saturated to [-u_max, u_max]

    # --- base embodiment overrides (defaults derived from bounds midpoints) ---
    base_masses: np.ndarray | None = field(default=None, repr=False)
    base_stiffness: np.ndarray | None = field(default=None, repr=False)
    base_damping: np.ndarray | None = field(default=None, repr=False)
    base_actuators: np.ndarray | None = field(default=None, repr=False)
    base_sensors: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.mass_low <= 0:
            raise ValueError("mass_low must be > 0 (masses are inverted in dynamics)")

    @property
    def state_dim(self) -> int:
        return 2 * self.n

    def Q(self) -> np.ndarray:
        """State cost matrix, ordered [positions (n), velocities (n)]."""
        return np.diag(np.concatenate([np.full(self.n, self.q_pos), np.full(self.n, self.q_vel)]))

    def R(self, n_act: int) -> np.ndarray:
        """Input cost matrix for ``n_act`` active actuators."""
        return self.r_act * np.eye(max(n_act, 1))


def _mid(lo: float, hi: float) -> float:
    return 0.5 * (lo + hi)


def make_config(controller: ControllerType, n: int = 6, **overrides) -> LQConfig:
    """Build a default config for a setup, applying any keyword overrides.

    The three setups differ only in which design variables are live, the observation
    mode, the noise levels, and the budget/saturation constraints. Everything else
    (chain size, integration, cost) shares sensible defaults.
    """
    if controller == "lqr":
        cfg = LQConfig(
            controller="lqr",
            n=n,
            observation_mode="full",
            process_noise_std=0.05,
            measurement_noise_std=0.0,
            design_actuators=False,
            design_sensors=False,
        )
    elif controller == "lqg":
        cfg = LQConfig(
            controller="lqg",
            n=n,
            observation_mode="partial",
            process_noise_std=0.05,
            measurement_noise_std=0.05,
            design_actuators=True,
            design_sensors=True,
        )
    elif controller == "qp":
        cfg = LQConfig(
            controller="qp",
            n=n,
            observation_mode="full",
            process_noise_std=0.05,
            measurement_noise_std=0.0,
            design_actuators=True,
            design_sensors=False,
            mass_budget=_mid(0.5, 2.0) * n * 1.1,
            actuator_budget=max(1, n // 2),
            u_max=1.0,
        )
    else:
        raise ValueError(f"unknown controller type: {controller!r}")

    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise TypeError(f"LQConfig has no field {key!r}")
        setattr(cfg, key, value)
    cfg.__post_init__()
    return cfg
