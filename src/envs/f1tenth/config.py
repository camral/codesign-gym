"""Preset definitions for the F1TENTH vehicle co-design environment.

The control policy drives a scaled racing car (steering + longitudinal command) around a
track to maximize progress; the embodiment is the vehicle *setup* -- tyre grip, cornering
stiffness, drive power, and weight distribution. Mass is **not** a design variable: it is
*derived* from the chosen upgrades (grippier/stiffer tyres and more power weigh more), and
tyre load-sensitivity then taxes that added mass. So an optimizer cannot maximize every
capability -- the tradeoffs are intrinsic physics, and every in-range design is feasible.

Couplings & constants are literature-anchored (Milliken & Milliken *RCVD* load sensitivity;
Pacejka mu(Fz); Genta dynamic index Iz=DI*m*lf*lr; CommonRoad single-track model). The
`kappa_*` upgrade->mass slopes and `k_mu` are calibratable modeling choices (documented).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict


@dataclass(frozen=True)
class F1TenthConfig:
    """Static configuration for one F1TENTH co-design preset."""

    name: str
    map_name: str
    num_beams: int          # LiDAR beams (observation)
    time_limit: int         # training control steps per episode; evaluation uses 10000

    # fixed geometry
    wheelbase: float = 0.3302   # L = lf + lr (m), fixed
    cg_height: float = 0.074    # h (m)

    # --- design variable ranges (what the optimizer controls) ---
    mu_min: float = 0.5
    mu_max: float = 1.1
    csf_min: float = 4.0
    csf_max: float = 6.0
    csr_min: float = 4.0
    csr_max: float = 6.0
    fdrive_min: float = 25.0    # drive force (N)
    fdrive_max: float = 55.0
    lr_frac_min: float = 0.45   # lr / L (CG split)
    lr_frac_max: float = 0.60

    # --- baseline (stock F1TENTH car) ---
    mu_base: float = 1.0489
    csf_base: float = 4.718
    csr_base: float = 5.4562
    fdrive_base: float = 35.57  # = a_max_base(9.51) * m_base(3.74)
    lr_frac_base: float = 0.5192  # 0.17145 / 0.3302

    # --- couplings / constants (modeling choices [CHOICE]) ---
    m_base: float = 3.74        # mass at the baseline spec (kg)
    kappa_mu: float = 1.0       # kg per unit mu upgrade
    kappa_c: float = 0.3        # kg per unit (C_Sf + C_Sr) upgrade
    kappa_p: float = 0.03       # kg per N of drive-force upgrade
    k_mu: float = 0.10          # tyre load-sensitivity slope (Milliken)
    dynamic_index: float = 0.4629  # DI in Iz = DI * m * lf * lr (Genta); gives the stock car's Iz = 0.04712 kg m^2
                                  # (race cars sit at ~0.24-0.48, road cars at ~1)


PRESETS: Dict[str, F1TenthConfig] = {
    "spielberg": F1TenthConfig(name="spielberg", map_name="Spielberg", num_beams=32, time_limit=1500),
    "levine": F1TenthConfig(name="levine", map_name="levine", num_beams=32, time_limit=1000),
}


def make_config(setup: str = "spielberg", **overrides: Any) -> F1TenthConfig:
    """Return the :class:`F1TenthConfig` for ``setup``, with optional field overrides."""
    if setup not in PRESETS:
        raise ValueError(f"unknown f1tenth setup {setup!r}; choose from {sorted(PRESETS)}")
    cfg = PRESETS[setup]
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg
