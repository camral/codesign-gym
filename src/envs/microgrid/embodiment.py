"""Embodiment design layer for the microgrid co-design environment.

The embodiment is the *sizing* of the installation, exposed as a dict of three
continuous parameters (see the Codesign Gym API contract):

    - ``pv_capacity_kw``       : installed solar capacity (kW)
    - ``battery_capacity_kwh`` : usable battery energy capacity (kWh)
    - ``genset_rating_kw``     : diesel generator rated power (kW)

Battery power limits (0.25C) and round-trip efficiency are *fixed constants*,
not part of the design (see PLAN.md, D2).
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
from gymnasium import spaces

# Ordered so a dict can be flattened to / from a vector deterministically.
EMBODIMENT_KEYS = ("pv_capacity_kw", "battery_capacity_kwh", "genset_rating_kw")

# The load is scaled to a nominal peak, and every design range is a multiple of
# that peak -- following python-microgrid's MicrogridGenerator._size_mg, which
# sizes all components as functions of the load:
#   PV      : peak_load * U(0.30, 1.50)      -> up to 1.5x peak
#   battery : U(3,6) h * mean_load (~0.5 peak) -> up to ~3x peak (kWh)
#   genset  : peak_load / 0.9                 -> ~1.1x peak
#   grid    : 2 * peak_load
DEFAULT_DEMAND_PEAK_KW = 300.0
RANGE_MULTIPLE_OF_PEAK = {
    "pv_capacity_kw": 1.5,
    "battery_capacity_kwh": 3.0,   # kWh, ~6 h of mean load
    "genset_rating_kw": 1.2,
}
GRID_LIMIT_MULTIPLE_OF_PEAK = 2.0

# Fixed (non-designed) battery physics.
BATTERY_C_RATE = 0.25           # max charge/discharge power = capacity/4 (4 h duration, per pymgrid)
BATTERY_EFFICIENCY = 0.9        # round-trip efficiency
BATTERY_INIT_SOC = 0.5          # state of charge at episode start (fraction)


def default_bounds(demand_peak_kw: float = DEFAULT_DEMAND_PEAK_KW):
    """(low, high) design bounds as library-derived multiples of the load peak."""
    low = {k: 0.0 for k in EMBODIMENT_KEYS}
    high = {k: RANGE_MULTIPLE_OF_PEAK[k] * float(demand_peak_kw) for k in EMBODIMENT_KEYS}
    return low, high

# Default capex ($ per unit of capacity) and amortization (PLAN.md D5).
DEFAULT_CAPEX_COEFFS = {
    "pv_capacity_kw": 1000.0,        # $/kW installed solar
    "battery_capacity_kwh": 300.0,   # $/kWh battery
    "genset_rating_kw": 500.0,       # $/kW genset
}
DEFAULT_LIFETIME_YEARS = 10.0
HOURS_PER_YEAR = 8760.0

# Type alias for a single embodiment (values may be python floats or 0-d arrays).
EmbType = Dict[str, float]


def make_embodiment_space(
    low: Mapping[str, float] | None = None,
    high: Mapping[str, float] | None = None,
) -> spaces.Dict:
    """Build the ``embodiment_space`` (a ``gymnasium.spaces.Dict`` of scalars)."""
    if low is None or high is None:
        low, high = default_bounds()
    return spaces.Dict({
        k: spaces.Box(low=np.float32(low[k]), high=np.float32(high[k]), shape=(), dtype=np.float32)
        for k in EMBODIMENT_KEYS
    })


def base_embodiment(space: spaces.Dict) -> EmbType:
    """The default embodiment used when the environment is created.

    Chosen as the midpoint of each design range: a reasonable, non-degenerate
    starting installation that the outer optimizer can move away from.
    """
    emb: EmbType = {}
    for k in EMBODIMENT_KEYS:
        box = space[k]
        emb[k] = float(0.5 * (float(box.low) + float(box.high)))
    return emb


def validate_embodiment(embodiment: Mapping[str, float], space: spaces.Dict) -> bool:
    """Return True iff ``embodiment`` has exactly the design keys, all finite and
    within ``space`` bounds."""
    if set(embodiment.keys()) != set(EMBODIMENT_KEYS):
        return False
    for k in EMBODIMENT_KEYS:
        v = float(embodiment[k])
        if not np.isfinite(v):
            return False
        box = space[k]
        if v < float(box.low) - 1e-9 or v > float(box.high) + 1e-9:
            return False
    return True


def capex(
    embodiment: Mapping[str, float],
    coeffs: Mapping[str, float] | None = None,
) -> float:
    """Total (undiscounted) capital cost of an embodiment, in dollars."""
    coeffs = coeffs or DEFAULT_CAPEX_COEFFS
    return float(sum(coeffs[k] * float(embodiment[k]) for k in EMBODIMENT_KEYS))


def amortized_capex(
    embodiment: Mapping[str, float],
    horizon: int,
    coeffs: Mapping[str, float] | None = None,
    lifetime_years: float = DEFAULT_LIFETIME_YEARS,
) -> float:
    """Capex amortized over the fraction of asset lifetime one episode spans.

    An episode covers ``horizon`` hours; the asset lives ``lifetime_years``.
    The episode is charged the corresponding slice of total capital cost. This
    lump is paid on the first step of the episode (see PLAN.md D5).
    """
    fraction = horizon / (lifetime_years * HOURS_PER_YEAR)
    return capex(embodiment, coeffs) * fraction
