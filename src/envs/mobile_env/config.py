"""Preset definitions for the mobile-env co-design environment.

Each preset is a realistic cellular-deployment archetype: a map, a user
population with a mobility regime, radio parameters, a **base-station budget**
(``min_bs``/``max_bs``), a per-BS transmit-power range, and a **total
radiated-power budget** (``tx_budget_fraction``) capping the summed power of the
active fleet. The budgets are the design pressure -- QoE reward is left untouched
(mobile-env's central handler returns mean UE utility), so the co-design problem
is to place and power a *scarce* set of towers well, not to add unbounded
infrastructure or crank every tower to max power.

The physically-anchored radio/mobility values (tx power in dBm, carrier freq in
MHz, bandwidth in Hz, UE velocity in m/s, antenna heights in m) are set to
standard real-world ranges; the map extent and seed positions stay in
mobile-env's abstract coordinate units (its Okumura-Hata scale is compressed, so
these are not literal metres).

Seed positions come from mobile-env's own ``small`` / ``medium`` scenarios where
applicable, so ``base_embodiment`` is a sensible reference deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class MobileEnvConfig:
    """Static configuration for one mobile-env co-design preset."""

    name: str
    # map (abstract units)
    width: int
    height: int
    ep_max_time: int
    # user population
    num_ues: int
    velocity: float          # m/s
    # radio (shared across base stations)
    freq: float              # MHz
    bw: float                # Hz
    bs_height: float         # m
    ue_height: float         # m
    snr_tr: float
    noise: float
    # base-station budget (design bounds)
    min_bs: int
    max_bs: int
    tx_min: float            # dBm
    tx_max: float            # dBm
    # total radiated-power budget: sum of active-station power (in W) is capped at
    # ``max_bs * W(tx_max) * tx_budget_fraction``. < 1 means the whole fleet cannot
    # run at tx_max at once, so power trades off against station count.
    tx_budget_fraction: float = 0.8
    # reference deployment (seed positions + seed tx) for base_embodiment
    seed_positions: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)
    seed_tx: float = 30.0
    # optional minimum separation between active base stations (0 = disabled)
    min_separation: float = 0.0


# Reference station layouts (from mobile-env's small/medium scenarios).
_SMALL_POS = ((110, 130), (65, 80), (120, 30))
_MEDIUM_POS = ((95, 240), (100, 140), (105, 60), (35, 200), (40, 95), (165, 220), (160, 110))
_RURAL_POS = ((100, 100), (300, 300), (100, 300))


PRESETS: Dict[str, MobileEnvConfig] = {
    # Dense city centre: pedestrians, capacity-limited, low-power small cells @ 3.5 GHz.
    "dense_urban": MobileEnvConfig(
        name="dense_urban",
        width=150, height=150, ep_max_time=100,
        num_ues=15, velocity=1.5,
        freq=3500, bw=20e6, bs_height=25, ue_height=1.5,
        snr_tr=2e-8, noise=1e-9,
        min_bs=3, max_bs=8, tx_min=23, tx_max=33,
        seed_positions=_SMALL_POS, seed_tx=30.0,
    ),
    # Suburban residential + roads: mixed slow-vehicular mobility, micro->macro @ 2.1 GHz.
    "suburban": MobileEnvConfig(
        name="suburban",
        width=200, height=300, ep_max_time=100,
        num_ues=20, velocity=8.0,
        freq=2100, bw=20e6, bs_height=35, ue_height=1.5,
        snr_tr=2e-8, noise=1e-9,
        min_bs=3, max_bs=7, tx_min=33, tx_max=43,
        seed_positions=_MEDIUM_POS, seed_tx=38.0,
    ),
    # Rural / highway: sparse coverage, high mobility, few high-power macros @ 700 MHz.
    "rural_highway": MobileEnvConfig(
        name="rural_highway",
        width=400, height=400, ep_max_time=100,
        num_ues=10, velocity=33.0,
        freq=700, bw=10e6, bs_height=50, ue_height=1.5,
        snr_tr=2e-8, noise=1e-9,
        min_bs=1, max_bs=4, tx_min=40, tx_max=46,
        seed_positions=_RURAL_POS, seed_tx=43.0,
    ),
}


def make_config(setup: str = "dense_urban", **overrides: Any) -> MobileEnvConfig:
    """Return the :class:`MobileEnvConfig` for ``setup``, with optional field overrides."""
    if setup not in PRESETS:
        raise ValueError(f"unknown mobile-env setup {setup!r}; choose from {sorted(PRESETS)}")
    cfg = PRESETS[setup]
    if overrides:
        cfg = replace(cfg, **overrides)
    return cfg
