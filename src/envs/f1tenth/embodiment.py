"""Embodiment design layer for the F1TENTH vehicle co-design environment.

The embodiment is the vehicle *setup*, a 5-vector of design variables the optimizer controls:

    [mu0, C_Sf, C_Sr, F_drive, lr_frac]
      mu0     : baseline tyre friction (grip upgrade)      -- costs mass
      C_Sf    : front cornering-stiffness coefficient      -- costs mass
      C_Sr    : rear cornering-stiffness coefficient       -- costs mass
      F_drive : drive force in N (power upgrade)            -- costs mass
      lr_frac : lr / L, the CG split                       -- mass-neutral (setup)

Everything physical is *derived* from these via literature-anchored couplings (see
``design_to_param_vector``): mass, tyre-load-sensitised friction, yaw inertia, and the
acceleration limit. The result is an 8-field numeric vector matching ``PARAM_FIELDS`` that
overrides those fields on f1tenth's ``Param`` (all other Param fields stay at the preset).

Validation is **structural only** -- every in-range design is feasible (mass is derived, not
constrained), so ``validate_embodiment`` is just a bounds check.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium import spaces

from .config import F1TenthConfig

# Optimizer-facing design variables (order fixed).
DESIGN_KEYS = ("mu0", "C_Sf", "C_Sr", "F_drive", "lr_frac")
# f1tenth Param fields the design overrides (order fixed; consumed by the vmapped step).
PARAM_FIELDS = ("m", "mu", "I", "lf", "lr", "a_max", "C_Sf", "C_Sr")

EmbType = np.ndarray  # shape (5,) float


def make_embodiment_space(cfg: F1TenthConfig) -> spaces.Box:
    low = np.array([cfg.mu_min, cfg.csf_min, cfg.csr_min, cfg.fdrive_min, cfg.lr_frac_min], dtype=np.float32)
    high = np.array([cfg.mu_max, cfg.csf_max, cfg.csr_max, cfg.fdrive_max, cfg.lr_frac_max], dtype=np.float32)
    return spaces.Box(low=low, high=high, shape=(5,), dtype=np.float32)


def max_accel(cfg: F1TenthConfig) -> float:
    """Largest ``a_max = F_drive / m`` any design in the space reaches. ``m`` is affine
    in the design, so ``a_max`` is monotone in each variable and peaks at a corner."""
    space = make_embodiment_space(cfg)
    corners = np.array(np.meshgrid(*zip(space.low, space.high), indexing="ij")).reshape(5, -1).T
    return float(design_to_param_vector(corners, cfg)[:, 5].max())


def base_embodiment(cfg: F1TenthConfig) -> EmbType:
    """The stock F1TENTH car (baseline spec)."""
    return np.array([cfg.mu_base, cfg.csf_base, cfg.csr_base, cfg.fdrive_base, cfg.lr_frac_base], dtype=np.float32)


def validate_embodiment(embodiment: Any, cfg: F1TenthConfig) -> bool:
    """Structural check only: correct shape and within the design bounds."""
    e = np.asarray(embodiment, dtype=np.float32)
    if e.shape != (5,) or not np.all(np.isfinite(e)):
        return False
    return bool(make_embodiment_space(cfg).contains(e))


def design_to_param_vector(design: np.ndarray, cfg: F1TenthConfig) -> np.ndarray:
    """Map a batch of designs ``(N, 5)`` to Param-override vectors ``(N, 8)`` via the couplings.

    - mass (secondary mass): m = m_base + kappa_mu*d_mu + kappa_c*d_CS + kappa_p*d_F
    - tyre load sensitivity : mu_eff = mu0 * (1 - k_mu * (m - m_base)/m_base)
    - yaw inertia (Genta)   : I = DI * m * lf * lr
    - power / mass          : a_max = F_drive / m
    - geometry              : lr = lr_frac * L, lf = L - lr
    """
    d = np.atleast_2d(np.asarray(design, dtype=np.float64))
    mu0, C_Sf, C_Sr, F_drive, lr_frac = d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4]

    m = (
        cfg.m_base
        + cfg.kappa_mu * (mu0 - cfg.mu_base)
        + cfg.kappa_c * ((C_Sf + C_Sr) - (cfg.csf_base + cfg.csr_base))
        + cfg.kappa_p * (F_drive - cfg.fdrive_base)
    )
    L = cfg.wheelbase
    lr = lr_frac * L
    lf = L - lr
    dfz = (m - cfg.m_base) / cfg.m_base
    mu_eff = mu0 * (1.0 - cfg.k_mu * dfz)
    I = cfg.dynamic_index * m * lf * lr
    a_max = F_drive / m

    out = np.stack([m, mu_eff, I, lf, lr, a_max, C_Sf, C_Sr], axis=1)
    return out.astype(np.float32)
