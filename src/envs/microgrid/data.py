"""Profile data layer.

Reuses pymgrid's bundled hourly datasets (real TMY solar and reference-building
electrical load) rather than reinventing profiles. Each dataset is 8760 hourly
samples (one year). We expose:

  - solar *capacity factor* in [0, 1] (normalized irradiance proxy), later
    multiplied by the designed ``pv_capacity_kw``;
  - electrical *demand* in kW.

Per episode a random contiguous window of length ``horizon`` is sampled (with
wrap-around), optionally from a randomly chosen city/building pair.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from ._compat import pymgrid

_DATA_ROOT = os.path.join(str(pymgrid.PROJECT_PATH), "data")

# pymgrid ships these matched city/building files. We pair solar and load by
# rough geography so a sampled scenario is internally plausible.
PV_FILES = {
    "houston": "Houston_722430TYA.csv",
    "raleigh": "Raleigh_723060TYA.csv",
    "sanfrancisco": "SanFrancisco_724940TYA.csv",
    "newyork": "NewYork_744860TYA.csv",
    "minneapolis": "Minneapolis_726580TYA.csv",
}
LOAD_FILES = {
    "hotel": "RefBldgLargeHotelNew2004_v1.3_7.1_4A_USA_MD_BALTIMORE.csv",
    "hospital": "RefBldgHospitalNew2004_7.1_5.0_3C_USA_CA_SAN_FRANCISCO.csv",
    "restaurant": "RefBldgFullServiceRestaurantNew2004_v1.3_7.1_6A_USA_MN_MINNEAPOLIS.csv",
    "school": "RefBldgPrimarySchoolNew2004_v1.3_7.1_2A_USA_TX_HOUSTON.csv",
    "office": "RefBldgLargeOfficeNew2004_v1.3_7.1_5A_USA_IL_CHICAGO-OHARE.csv",
}

HOURS_PER_YEAR = 8760


def _read_series(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    return df.iloc[:, 0].to_numpy(dtype=np.float64)


def load_solar_capacity_factor(city: str) -> np.ndarray:
    """Normalized solar generation profile in [0, 1] (8760,)."""
    raw = _read_series(os.path.join(_DATA_ROOT, "pv", PV_FILES[city]))
    peak = raw.max()
    return raw / peak if peak > 0 else raw


def load_demand_kw(building: str, peak_kw: float | None = None) -> np.ndarray:
    """Electrical demand in kW (8760,).

    If ``peak_kw`` is given, the whole-year profile is scaled so its annual peak
    equals ``peak_kw`` (following pymgrid's ``_scale_ts`` load normalization).
    This puts every building on a common scale so the design ranges -- defined
    as multiples of the peak -- are meaningful regardless of which building is
    sampled.
    """
    raw = _read_series(os.path.join(_DATA_ROOT, "load", LOAD_FILES[building]))
    if peak_kw is not None:
        m = raw.max()
        if m > 0:
            raw = raw * (float(peak_kw) / m)
    return raw


@dataclass
class Window:
    """A sampled episode's exogenous profiles (length ``horizon``)."""

    demand_kw: np.ndarray          # (horizon,) aggregate load
    demand_by_building: np.ndarray  # (horizon, n_buildings) per-building load
    solar_cf: np.ndarray           # (horizon,) capacity factor in [0, 1]
    hour_of_day: np.ndarray        # (horizon,) integer 0..23
    city: str
    building: str                  # "+"-joined names, aligned with columns above
    start: int


class ProfileSampler:
    """Samples episode windows from the pymgrid datasets.

    Parameters
    ----------
    horizon:
        Number of hourly steps per episode.
    cities / buildings:
        Restrict sampling to these datasets. A single-element list fixes the
        scenario; multiple elements randomize it per reset.
    randomize_start:
        If True, a random start hour is chosen each reset; otherwise start=0.
    """

    def __init__(
        self,
        horizon: int,
        cities: List[str] | None = None,
        buildings: List[str] | None = None,
        randomize_start: bool = True,
        n_buildings: int = 1,
        per_building_peak_kw: float | None = None,
    ):
        self.horizon = int(horizon)
        self.cities = cities or list(PV_FILES.keys())
        self.buildings = buildings or list(LOAD_FILES.keys())
        self.randomize_start = randomize_start
        self.n_buildings = int(n_buildings)
        self.per_building_peak_kw = per_building_peak_kw
        # Cache each full-year building profile scaled to a common per-building
        # peak; a campus load is the sum of several of these.
        self._solar = {c: load_solar_capacity_factor(c) for c in self.cities}
        self._demand = {b: load_demand_kw(b, per_building_peak_kw) for b in self.buildings}

    def sample(self, rng: np.random.Generator) -> Window:
        city = self.cities[rng.integers(len(self.cities))]
        replace = self.n_buildings > len(self.buildings)
        chosen = list(rng.choice(self.buildings, size=self.n_buildings, replace=replace))
        start = int(rng.integers(HOURS_PER_YEAR)) if self.randomize_start else 0
        idx = (start + np.arange(self.horizon)) % HOURS_PER_YEAR
        # Per-building loads (horizon, n), and their diversified aggregate.
        per = np.stack([self._demand[b][idx] for b in chosen], axis=1).astype(float)
        return Window(
            demand_kw=per.sum(axis=1),
            demand_by_building=per,
            solar_cf=self._solar[city][idx].copy(),
            hour_of_day=(idx % 24).astype(np.int64),
            city=city,
            building="+".join(chosen),
            start=start,
        )
