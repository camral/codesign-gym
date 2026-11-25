"""Native vectorized NumPy microgrid co-design environment.

``MicrogridVectorEnv`` steps all ``num_envs`` microgrids simultaneously in
batched NumPy arrays, reproducing ``pymgrid`` 1.4.1's exact dispatch, battery
efficiency, genset fuel, grid import/export, and balancing equations to machine
precision without per-env Python object overhead.

  - ``reset(embodiment=...)`` and ``step(actions, embodiment=...)`` accept a batch
    of designs -- ``None`` (all base), a single dict (broadcast), a stacked dict
    whose values have a leading ``num_envs`` axis, or a list of ``num_envs`` dicts
    (entries may be ``None``).
  - ``reset`` returns ``embodiment_rejected`` as a boolean array of shape
    ``(num_envs,)``; a per-env rejection leaves that lane invalid until reset.
  - Transition-level embodiment changes project stored kWh to the new battery
    capacity in O(N) vectorized NumPy with zero object rebuilds.

Autoreset follows Gymnasium >=1.0 NEXT_STEP semantics: a lane that finished on
step ``t`` is reset on step ``t+1`` (action ignored, reward 0, done flags False).
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence as AbcSequence
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

from core.codesign_vec_env import CoDesignVecEnv
from core.embodiment_batch import split_dict_designs
from core.seeding import lane_seeds

from . import embodiment as emb_mod
from .data import ProfileSampler, HOURS_PER_YEAR
from .embodiment import EmbType, EMBODIMENT_KEYS

_EPS = 1e-6
_CORE_OBS_DIM = 5
_ACTION_DIM = 3

BatchedEmb = Union[None, Mapping[str, float], Sequence[Mapping[str, float]], Mapping[str, Sequence[float]]]


class _EmbodimentListView(AbcSequence):
    """O(1) lazy sequence view exposing per-lane embodiment dicts without
    allocating ``num_envs`` Python dicts every step."""

    __slots__ = ("_pv", "_batt", "_gen")

    def __init__(self, pv: np.ndarray, batt: np.ndarray, gen: np.ndarray):
        self._pv = pv
        self._batt = batt
        self._gen = gen

    def __len__(self) -> int:
        return int(self._pv.shape[0])

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        return {
            "pv_capacity_kw": float(self._pv[idx]),
            "battery_capacity_kwh": float(self._batt[idx]),
            "genset_rating_kw": float(self._gen[idx]),
        }


class MicrogridVectorEnv(CoDesignVecEnv):
    """Batched NumPy microgrid sizing-and-dispatch co-design environment."""

    metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP, "render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_envs: int,
        *,
        horizon: int = 24,
        is_eval_env: bool = False,
        embodiment_bounds: Optional[Tuple[Mapping[str, float], Mapping[str, float]]] = None,
        sampler: Optional[ProfileSampler] = None,
        demand_peak_kw: float = emb_mod.DEFAULT_DEMAND_PEAK_KW,
        n_buildings: int = 1,
        per_building_peak_kw: Optional[float] = None,
        off_grid: bool = False,
        capex_coeffs: Optional[Mapping[str, float]] = None,
        lifetime_years: float = emb_mod.DEFAULT_LIFETIME_YEARS,
        import_price: float = 0.30,
        export_price: float = 0.05,
        genset_fuel_cost: float = 0.40,
        unmet_penalty: float = 10.0,
        overgen_penalty: float = 2.0,
        grid_max_import: Optional[float] = None,
        grid_max_export: Optional[float] = None,
        allow_transition_embodiment: bool = True,
        forecast_horizon: int = 0,
        forecast_noise: float = 0.0,
        render_mode: Optional[str] = None,
        render_pause: float = 0.4,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")

        self.horizon = int(horizon)
        self.render_mode = render_mode
        self.render_pause = float(render_pause)
        self._hfigs: List[Any] = [None] * num_envs

        self.demand_peak_kw = float(demand_peak_kw)
        if embodiment_bounds is None:
            low, high = emb_mod.default_bounds(self.demand_peak_kw)
        else:
            low, high = embodiment_bounds
        single_emb_space = emb_mod.make_embodiment_space(low, high)
        base_emb = emb_mod.base_embodiment(single_emb_space)

        self.n_buildings = int(n_buildings)
        self.off_grid = bool(off_grid)
        pb = per_building_peak_kw if per_building_peak_kw is not None else self.demand_peak_kw / max(self.n_buildings, 1)
        # Single shared ProfileSampler across all num_envs lanes!
        self.sampler = sampler or ProfileSampler(self.horizon, n_buildings=self.n_buildings, per_building_peak_kw=pb)

        self.capex_coeffs = dict(capex_coeffs or emb_mod.DEFAULT_CAPEX_COEFFS)
        self._c_pv = float(self.capex_coeffs["pv_capacity_kw"])
        self._c_batt = float(self.capex_coeffs["battery_capacity_kwh"])
        self._c_gen = float(self.capex_coeffs["genset_rating_kw"])
        self.lifetime_years = float(lifetime_years)
        self._amort_frac = self.horizon / (self.lifetime_years * emb_mod.HOURS_PER_YEAR)

        self.import_price = float(import_price)
        self.export_price = float(export_price)
        self.genset_fuel_cost = float(genset_fuel_cost)
        self.unmet_penalty = float(unmet_penalty)
        self.overgen_penalty = float(overgen_penalty)
        grid_limit = 0.0 if self.off_grid else emb_mod.GRID_LIMIT_MULTIPLE_OF_PEAK * self.demand_peak_kw
        self.grid_max_import = float(grid_max_import) if grid_max_import is not None else grid_limit
        self.grid_max_export = float(grid_max_export) if grid_max_export is not None else grid_limit

        self.allow_transition_embodiment = bool(allow_transition_embodiment) and not is_eval_env
        self._warned_transition = False
        self.forecast_horizon = int(forecast_horizon)
        self.forecast_noise = float(forecast_noise)

        core_obs_dim = _CORE_OBS_DIM + 2 * self.forecast_horizon
        self._core_obs_dim = core_obs_dim
        single_act_space = spaces.Box(low=-1.0, high=1.0, shape=(_ACTION_DIM,), dtype=np.float32)
        single_obs_space = spaces.Box(
            low=np.zeros(core_obs_dim, np.float32),
            high=np.full(core_obs_dim, np.inf, np.float32),
            shape=(core_obs_dim,),
            dtype=np.float32,
        )

        super().__init__(
            num_envs,
            single_obs_space,
            single_act_space,
            single_emb_space,
            base_emb,
            is_eval_env=is_eval_env,
        )

        # Design bounds as scalars for fast vectorized validation
        self._lo_pv = float(single_emb_space["pv_capacity_kw"].low)
        self._hi_pv = float(single_emb_space["pv_capacity_kw"].high)
        self._lo_batt = float(single_emb_space["battery_capacity_kwh"].low)
        self._hi_batt = float(single_emb_space["battery_capacity_kwh"].high)
        self._lo_gen = float(single_emb_space["genset_rating_kw"].low)
        self._hi_gen = float(single_emb_space["genset_rating_kw"].high)

        self._base_pv = float(base_emb["pv_capacity_kw"])
        self._base_batt = float(base_emb["battery_capacity_kwh"])
        self._base_gen = float(base_emb["genset_rating_kw"])

        # Pre-stacked profile tables for fast vectorized window indexing
        self._solar_table = np.stack([self.sampler._solar[c] for c in self.sampler.cities], axis=0)
        self._demand_table = np.stack([self.sampler._demand[b] for b in self.sampler.buildings], axis=0)
        self._h_arange = np.arange(self.horizon, dtype=np.int64)

        # Per-lane RNGs (matches gym.Env.reset seeding for seed=[s] parity with single env)
        self._rngs: List[np.random.Generator] = [seeding.np_random(None)[0] for _ in range(num_envs)]

        # Vectorized simulation state
        n, h = num_envs, self.horizon
        self._pv_cap = np.full(n, self._base_pv, dtype=np.float64)
        self._batt_kwh = np.full(n, self._base_batt, dtype=np.float64)
        self._gen_kw = np.full(n, self._base_gen, dtype=np.float64)
        self._soc = np.full(n, emb_mod.BATTERY_INIT_SOC, dtype=np.float64)
        self._t = np.zeros(n, dtype=np.int64)
        self._first_step = np.ones(n, dtype=bool)
        self._valid = np.zeros(n, dtype=bool)
        self._autoreset = np.zeros(n, dtype=bool)
        self._amort_capex = np.zeros(n, dtype=np.float64)
        self._capex_total = np.zeros(n, dtype=np.float64)

        self._demand_kw = np.zeros((n, h), dtype=np.float64)
        self._solar_cf = np.zeros((n, h), dtype=np.float64)
        self._hour_of_day = np.zeros((n, h), dtype=np.int64)
        self._cities: List[str] = [""] * n
        self._buildings: List[str] = [""] * n
        self._demand_by_bldg: Optional[np.ndarray] = (
            np.zeros((n, h, self.n_buildings), dtype=np.float64) if self.render_mode is not None else None
        )
        self._histories: Optional[List[list]] = [[] for _ in range(n)] if self.render_mode is not None else None

        self._last_obs: Optional[np.ndarray] = None
        self._has_been_reset = np.zeros(n, dtype=bool)

    # ------------------------------------------------------------------ #
    # Fast vectorized embodiment parsing & validation
    # ------------------------------------------------------------------ #
    def _parse_embodiment_arrays(
        self, embodiment: BatchedEmb, *, default_to_base: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Parse any BatchedEmb into (pv, batt, gen, present_mask) arrays of shape (N,).

        Raises TypeError/ValueError if structure/batch size is malformed.
        """
        n = self.num_envs
        if embodiment is None:
            present = np.zeros(n, dtype=bool)
            if default_to_base:
                return (
                    np.full(n, self._base_pv, dtype=np.float64),
                    np.full(n, self._base_batt, dtype=np.float64),
                    np.full(n, self._base_gen, dtype=np.float64),
                    present,
                )
            return self._pv_cap.copy(), self._batt_kwh.copy(), self._gen_kw.copy(), present

        # Fast path: dict of scalars (broadcast) or dict of length-N arrays
        if isinstance(embodiment, Mapping):
            if set(embodiment.keys()) != set(EMBODIMENT_KEYS):
                raise ValueError(f"Expected embodiment keys {EMBODIMENT_KEYS}, got {tuple(embodiment.keys())}")
            vals = []
            for k in EMBODIMENT_KEYS:
                arr = np.asarray(embodiment[k], dtype=np.float64)
                if arr.ndim == 0:
                    arr = np.full(n, float(arr), dtype=np.float64)
                elif arr.ndim == 1 and arr.shape[0] == n:
                    arr = arr.astype(np.float64, copy=False)
                else:
                    raise ValueError(f"Embodiment key {k!r} has shape {arr.shape}, expected () or ({n},)")
                vals.append(arr)
            return vals[0], vals[1], vals[2], np.ones(n, dtype=bool)

        # Fallback: sequence of per-lane dicts (may contain None entries)
        embs = split_dict_designs(embodiment, n, {k: 0 for k in EMBODIMENT_KEYS})
        pv = np.full(n, self._base_pv, dtype=np.float64) if default_to_base else self._pv_cap.copy()
        batt = np.full(n, self._base_batt, dtype=np.float64) if default_to_base else self._batt_kwh.copy()
        gen = np.full(n, self._base_gen, dtype=np.float64) if default_to_base else self._gen_kw.copy()
        present = np.zeros(n, dtype=bool)
        for i, e in enumerate(embs):
            if e is None:
                continue
            if not isinstance(e, Mapping) or set(e.keys()) != set(EMBODIMENT_KEYS):
                raise ValueError(f"Invalid embodiment at lane {i}: {e}")
            pv[i] = float(e["pv_capacity_kw"])
            batt[i] = float(e["battery_capacity_kwh"])
            gen[i] = float(e["genset_rating_kw"])
            present[i] = True
        return pv, batt, gen, present

    def _valid_mask(self, pv: np.ndarray, batt: np.ndarray, gen: np.ndarray) -> np.ndarray:
        return (
            np.isfinite(pv) & np.isfinite(batt) & np.isfinite(gen)
            & (pv >= self._lo_pv - 1e-9) & (pv <= self._hi_pv + 1e-9)
            & (batt >= self._lo_batt - 1e-9) & (batt <= self._hi_batt + 1e-9)
            & (gen >= self._lo_gen - 1e-9) & (gen <= self._hi_gen + 1e-9)
        )

    def _validate_embodiment(self, embodiment: BatchedEmb) -> np.ndarray:
        if embodiment is None:
            return np.ones(self.num_envs, dtype=bool)
        try:
            pv, batt, gen, present = self._parse_embodiment_arrays(embodiment, default_to_base=True)
        except (TypeError, ValueError, KeyError):
            # Per-lane fallback if a sequence has one malformed dict among valid ones
            if isinstance(embodiment, Sequence) and not isinstance(embodiment, (str, bytes)) and len(embodiment) == self.num_envs:
                out = np.zeros(self.num_envs, dtype=bool)
                for i, e in enumerate(embodiment):
                    if e is None:
                        out[i] = True
                    elif isinstance(e, Mapping):
                        try:
                            out[i] = emb_mod.validate_embodiment(e, self.embodiment_space)
                        except (TypeError, ValueError, KeyError):
                            out[i] = False
                return out
            return np.zeros(self.num_envs, dtype=bool)
        return (~present) | self._valid_mask(pv, batt, gen)

    # ------------------------------------------------------------------ #
    # Sampling & Observation
    # ------------------------------------------------------------------ #
    def _sample_lanes(self, idx_list: np.ndarray) -> None:
        """Sample exogenous demand/solar windows for the given lane indices."""
        n_cities = len(self.sampler.cities)
        n_bldgs_avail = len(self.sampler.buildings)
        replace = self.n_buildings > n_bldgs_avail
        rand_start = self.sampler.randomize_start
        h_ar = self._h_arange
        need_bldg = self._demand_by_bldg is not None

        for i in idx_list:
            rng = self._rngs[int(i)]
            c_idx = int(rng.integers(n_cities))
            # Match ProfileSampler.sample RNG draw order identically!
            chosen_idx = rng.choice(n_bldgs_avail, size=self.n_buildings, replace=replace)
            start = int(rng.integers(HOURS_PER_YEAR)) if rand_start else 0
            t_idx = (start + h_ar) % HOURS_PER_YEAR
            per = self._demand_table[chosen_idx][:, t_idx].T  # (H, n_buildings)
            self._demand_kw[i] = per.sum(axis=1)
            self._solar_cf[i] = self._solar_table[c_idx, t_idx]
            self._hour_of_day[i] = t_idx % 24
            self._cities[i] = self.sampler.cities[c_idx]
            self._buildings[i] = "+".join(self.sampler.buildings[int(b)] for b in chosen_idx)
            if need_bldg:
                self._demand_by_bldg[i] = per

    def _observe_lanes(self, lanes: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute (len(lanes), obs_dim) float32 observations."""
        if lanes is None:
            idx = np.arange(self.num_envs, dtype=np.int64)
        else:
            idx = np.asarray(lanes, dtype=np.int64)
        m = idx.shape[0]
        t_cur = np.minimum(self._t[idx], self.horizon - 1)
        dem = self._demand_kw[idx, t_cur]
        pv_avail = self._pv_cap[idx] * self._solar_cf[idx, t_cur]
        hr_frac = self._hour_of_day[idx, t_cur] / 24.0

        obs = np.empty((m, self._core_obs_dim), dtype=np.float32)
        obs[:, 0] = self._soc[idx]
        obs[:, 1] = dem
        obs[:, 2] = pv_avail
        obs[:, 3] = hr_frac
        obs[:, 4] = self.import_price

        fh = self.forecast_horizon
        if fh > 0:
            offs = np.arange(1, fh + 1, dtype=np.int64)[None, :]  # (1, fh)
            f_idx = np.minimum(t_cur[:, None] + offs, self.horizon - 1)  # (m, fh)
            dem_f = self._demand_kw[idx[:, None], f_idx].astype(np.float64)
            pv_f = (self._pv_cap[idx, None] * self._solar_cf[idx[:, None], f_idx]).astype(np.float64)
            if self.forecast_noise > 0.0:
                for k, lane in enumerate(idx):
                    rng = self._rngs[int(lane)]
                    dem_f[k] = np.clip(dem_f[k] + rng.normal(0.0, self.forecast_noise * (np.abs(dem_f[k]) + 1e-6)), 0.0, None)
                    pv_f[k] = np.clip(pv_f[k] + rng.normal(0.0, self.forecast_noise * (np.abs(pv_f[k]) + 1e-6)), 0.0, None)
            obs[:, 5 : 5 + fh] = dem_f
            obs[:, 5 + fh : 5 + 2 * fh] = pv_f
        return obs

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def _reset(
        self,
        *,
        seed: Optional[Union[int, Sequence[int]]] = None,
        options: Optional[dict] = None,
        embodiment: BatchedEmb = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        seeds = lane_seeds(seed, self.num_envs)
        pv, batt, gen, _ = self._parse_embodiment_arrays(embodiment, default_to_base=True)

        options = dict(options or {})
        mask = np.ones(self.num_envs, dtype=bool)
        if "reset_mask" in options:
            mask = np.asarray(options.pop("reset_mask"), dtype=bool)
            if mask.shape != (self.num_envs,):
                raise ValueError("reset_mask must have shape (num_envs,)")
            if not np.all(self._has_been_reset[~mask]):
                raise ValueError("a full reset is required before a partial reset")

        valid_emb = self._valid_mask(pv, batt, gen)
        rejected = mask & (~valid_emb)
        accept = mask & valid_emb

        # Seed RNGs on masked lanes
        reset_idx = np.flatnonzero(mask)
        for i in reset_idx:
            s = seeds[int(i)]
            if s is not None:
                self._rngs[int(i)] = seeding.np_random(int(s))[0]

        # Reject invalid lanes
        if np.any(rejected):
            self._valid[rejected] = False
            self._autoreset[rejected] = False

        # Accept valid lanes
        accept_idx = np.flatnonzero(accept)
        if accept_idx.size > 0:
            self._pv_cap[accept] = pv[accept]
            self._batt_kwh[accept] = batt[accept]
            self._gen_kw[accept] = gen[accept]
            self._soc[accept] = emb_mod.BATTERY_INIT_SOC
            self._t[accept] = 0
            self._first_step[accept] = True
            self._valid[accept] = True
            self._autoreset[accept] = False
            self._has_been_reset[accept] = True
            self._capex_total[accept] = (
                self._c_pv * pv[accept] + self._c_batt * batt[accept] + self._c_gen * gen[accept]
            )
            self._amort_capex[accept] = self._capex_total[accept] * self._amort_frac
            self._sample_lanes(accept_idx)
            if self._histories is not None:
                for i in accept_idx:
                    self._histories[int(i)] = []

        if self._last_obs is None:
            obs = np.zeros((self.num_envs, self._core_obs_dim), dtype=np.float32)
        else:
            obs = self._last_obs.copy()

        if accept_idx.size > 0:
            obs[accept_idx] = self._observe_lanes(accept_idx)
        if np.any(rejected):
            obs[rejected] = 0.0
        self._last_obs = obs.copy()

        info = {
            "embodiment": _EmbodimentListView(self._pv_cap.copy(), self._batt_kwh.copy(), self._gen_kw.copy()),
            "capex": self._amort_capex.copy(),
            "capex_total": self._capex_total.copy(),
            "city": list(self._cities),
            "building": list(self._buildings),
        }
        return obs, rejected, info

    # ------------------------------------------------------------------ #
    # Vectorized Step (exact pymgrid 1.4.1 physics parity)
    # ------------------------------------------------------------------ #
    def _step(
        self,
        actions: np.ndarray,
        embodiment: BatchedEmb = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        n = self.num_envs
        act = np.asarray(actions, dtype=np.float64)
        if act.ndim == 1:
            act = np.broadcast_to(act, (n, _ACTION_DIM))
        act = np.clip(act[:, :_ACTION_DIM], -1.0, 1.0)

        auto_mask = self._autoreset.copy()
        active_mask = ~auto_mask

        if np.any(active_mask & (~self._valid)):
            raise RuntimeError("step() called on an invalid environment (embodiment was rejected at reset, or reset was never called).")

        # Parse optional embodiment batch
        new_pv, new_batt, new_gen, present = self._parse_embodiment_arrays(embodiment, default_to_base=False)

        capex_delta = np.zeros(n, dtype=np.float64)
        emb_rejected = np.zeros(n, dtype=bool)

        # --- Transition-level embodiment on active lanes ---
        trans_mask = active_mask & present
        if np.any(trans_mask):
            if not self.allow_transition_embodiment:
                if not self._warned_transition:
                    warnings.warn(
                        "This environment does not support transition-level "
                        "embodiment changes; the provided embodiment is ignored."
                    )
                    self._warned_transition = True
            else:
                v_mask = self._valid_mask(new_pv, new_batt, new_gen)
                if not np.all(v_mask[trans_mask]):
                    bad_i = int(np.flatnonzero(trans_mask & (~v_mask))[0])
                    raise ValueError(
                        f"Invalid transition embodiment at lane {bad_i}: "
                        f"pv={new_pv[bad_i]}, batt={new_batt[bad_i]}, gen={new_gen[bad_i]}"
                    )
                # Vectorized SoC projection: preserve stored kWh clamped to new capacity
                stored = self._soc[trans_mask] * self._batt_kwh[trans_mask]
                nb = new_batt[trans_mask]
                proj_kwh = np.minimum(stored, nb)
                self._soc[trans_mask] = np.clip(proj_kwh / np.maximum(nb, _EPS), 0.0, 1.0)

                old_capex_tot = self._capex_total[trans_mask]
                new_capex_tot = (
                    self._c_pv * new_pv[trans_mask]
                    + self._c_batt * nb
                    + self._c_gen * new_gen[trans_mask]
                )
                rem_frac = (self.horizon - self._t[trans_mask]) / (self.lifetime_years * emb_mod.HOURS_PER_YEAR)
                capex_delta[trans_mask] = np.maximum(new_capex_tot - old_capex_tot, 0.0) * rem_frac

                self._pv_cap[trans_mask] = new_pv[trans_mask]
                self._batt_kwh[trans_mask] = nb
                self._gen_kw[trans_mask] = new_gen[trans_mask]
                self._capex_total[trans_mask] = new_capex_tot
                self._amort_capex[trans_mask] = new_capex_tot * self._amort_frac

        # --- Vectorized physics step on active lanes ---
        rewards = np.zeros(n, dtype=np.float32)
        terms = np.zeros(n, dtype=bool)
        truncs = np.zeros(n, dtype=bool)
        op_cost_arr = np.zeros(n, dtype=np.float64)
        capex_charged = np.zeros(n, dtype=np.float64)

        act_idx = np.flatnonzero(active_mask)
        if act_idx.size > 0:
            t_cur = self._t[act_idx]
            dem = self._demand_kw[act_idx, t_cur]
            pv_avail = self._pv_cap[act_idx] * self._solar_cf[act_idx, t_cur]

            # First-step amortized capex lump
            first = self._first_step[act_idx]
            c_chg = np.where(first, self._amort_capex[act_idx], 0.0)
            capex_charged[act_idx] = c_chg
            self._first_step[act_idx] = False

            # Battery physics (pymgrid BatteryModule linear transition, efficiency=0.9)
            eff = emb_mod.BATTERY_EFFICIENCY
            b_cap = np.maximum(self._batt_kwh[act_idx], _EPS)
            b_pwr = np.maximum(emb_mod.BATTERY_C_RATE * self._batt_kwh[act_idx], _EPS)
            charge = self._soc[act_idx] * b_cap
            a_batt = act[act_idx, 0]
            batt_cmd = a_batt * (emb_mod.BATTERY_C_RATE * self._batt_kwh[act_idx])

            # Discharge (batt_cmd >= 0) vs Charge (batt_cmd < 0)
            max_ext_dis = np.minimum(b_pwr, charge) * eff
            max_ext_chg = np.minimum(b_pwr, b_cap - charge) / eff
            is_dis = batt_cmd >= 0.0
            ext_dis = np.minimum(np.maximum(batt_cmd, 0.0), max_ext_dis)
            ext_chg = np.minimum(np.maximum(-batt_cmd, 0.0), max_ext_chg)
            batt_net = np.where(is_dis, ext_dis, -ext_chg)
            new_charge = np.where(is_dis, charge - ext_dis / eff, charge + ext_chg * eff)
            new_soc = np.clip(new_charge / b_cap, 0.0, 1.0)
            self._soc[act_idx] = new_soc

            # Genset physics
            a_gen = act[act_idx, 1]
            gen_raw = np.maximum(a_gen, 0.0) * self._gen_kw[act_idx]
            gen_prod = np.where(gen_raw > _EPS, np.minimum(gen_raw, np.maximum(self._gen_kw[act_idx], _EPS)), 0.0)
            gen_cost = self.genset_fuel_cost * gen_prod

            # Grid physics
            a_grid = act[act_idx, 2]
            grid_net = np.where(a_grid >= 0.0, a_grid * self.grid_max_import, a_grid * self.grid_max_export)
            grid_cost = np.where(grid_net >= 0.0, self.import_price * grid_net, self.export_price * grid_net)

            # Fixed Load + Flex PV + Flex Balancing
            diff = batt_net + gen_prod + grid_net - dem
            need = np.maximum(-diff, 0.0)
            pv_used = np.minimum(pv_avail, need)
            pv_curtail = pv_avail - pv_used
            unmet = need - pv_used
            overgen = np.maximum(diff, 0.0)

            bal_cost = self.unmet_penalty * unmet + self.overgen_penalty * overgen
            op_cost = gen_cost + grid_cost + bal_cost
            op_cost_arr[act_idx] = op_cost
            rew = -op_cost - c_chg
            rewards[act_idx] = rew.astype(np.float32)

            # Record history only when rendering is enabled
            if self._histories is not None:
                for k, i in enumerate(act_idx):
                    ii = int(i)
                    self._histories[ii].append({
                        "hour": len(self._histories[ii]),
                        "demand": float(dem[k]),
                        "pv": float(pv_used[k]),
                        "battery": float(batt_net[k]),
                        "genset": float(gen_prod[k]),
                        "grid": float(grid_net[k]),
                        "unmet": float(unmet[k]),
                        "overgen": float(overgen[k]),
                        "curtail": float(pv_curtail[k]),
                        "soc": float(new_soc[k]),
                        "reward": float(rew[k]),
                        "op_cost": float(op_cost[k]),
                        "capex_charged": float(c_chg[k]),
                    })

            self._t[act_idx] += 1
            done_active = self._t[act_idx] >= self.horizon
            truncs[act_idx] = done_active
            self._autoreset[act_idx] = done_active

        # --- Autoreset lanes (NEXT_STEP semantics) ---
        auto_idx = np.flatnonzero(auto_mask)
        if auto_idx.size > 0:
            # Install step's embodiment if provided, else keep lane's current design
            pv_a = np.where(present[auto_idx], new_pv[auto_idx], self._pv_cap[auto_idx])
            bt_a = np.where(present[auto_idx], new_batt[auto_idx], self._batt_kwh[auto_idx])
            gn_a = np.where(present[auto_idx], new_gen[auto_idx], self._gen_kw[auto_idx])
            v_a = self._valid_mask(pv_a, bt_a, gn_a)
            rej_a = auto_idx[~v_a]
            acc_a = auto_idx[v_a]
            if rej_a.size > 0:
                emb_rejected[rej_a] = True
                self._valid[rej_a] = False
                self._autoreset[rej_a] = False
            if acc_a.size > 0:
                self._pv_cap[acc_a] = pv_a[v_a]
                self._batt_kwh[acc_a] = bt_a[v_a]
                self._gen_kw[acc_a] = gn_a[v_a]
                self._soc[acc_a] = emb_mod.BATTERY_INIT_SOC
                self._t[acc_a] = 0
                self._first_step[acc_a] = True
                self._valid[acc_a] = True
                self._autoreset[acc_a] = False
                self._capex_total[acc_a] = (
                    self._c_pv * pv_a[v_a] + self._c_batt * bt_a[v_a] + self._c_gen * gn_a[v_a]
                )
                self._amort_capex[acc_a] = self._capex_total[acc_a] * self._amort_frac
                self._sample_lanes(acc_a)
                if self._histories is not None:
                    for ii in acc_a:
                        self._histories[int(ii)] = []

        obs = self._observe_lanes()
        if np.any(emb_rejected):
            obs[emb_rejected] = 0.0
        self._last_obs = obs.copy()

        info = {
            "embodiment_rejected": emb_rejected,
            "op_cost": op_cost_arr,
            "capex": self._amort_capex.copy(),
            "capex_total": self._capex_total.copy(),
            "capex_charged": capex_charged,
            "capex_delta": capex_delta,
            "embodiment": _EmbodimentListView(self._pv_cap.copy(), self._batt_kwh.copy(), self._gen_kw.copy()),
            "soc": self._soc.copy(),
        }
        return obs, rewards, terms, truncs, info

    # ------------------------------------------------------------------ #
    # Rendering (delegates to Coastal Solarpunk Eco-Island schematic)
    # ------------------------------------------------------------------ #
    def _render_lane(self, i: int):
        if self.render_mode not in ("human", "rgb_array") or self._histories is None or not self._histories[i]:
            return None
        from .schematic import render_env_figure

        hist = self._histories[i]
        keys = ("hour", "demand", "pv", "battery", "genset", "grid", "unmet",
                "overgen", "curtail", "soc", "reward", "op_cost", "capex_charged")
        rec = {k: np.array([h[k] for h in hist], dtype=float) for k in keys}
        emb = {
            "pv_capacity_kw": float(self._pv_cap[i]),
            "battery_capacity_kwh": float(self._batt_kwh[i]),
            "genset_rating_kw": float(self._gen_kw[i]),
        }
        rec["embodiment"] = emb
        last = hist[-1]
        buildings = self._buildings[i].split("+")
        nb = len(buildings)
        t = len(hist) - 1
        per_bldg = self._demand_by_bldg[i, t]
        served_frac = (last["demand"] - last["unmet"]) / max(last["demand"], 1e-6)
        by_building = [
            {"name": nm, "demand": float(dem), "served": float(dem * served_frac)}
            for nm, dem in zip(buildings, per_bldg)
        ]
        conn = "off-grid" if self.off_grid else "grid-connected"
        bounds = {
            k: (float(sp.low.flat[0]), float(sp.high.flat[0]))
            for k, sp in self.embodiment_space.spaces.items()
        }
        state = {
            "embodiment": emb,
            "embodiment_bounds": bounds,
            "off_grid": self.off_grid,
            "buildings": buildings,
            "demand_peak": float(max(rec["demand"].max(), 1.0)),
            "grid_max": self.grid_max_import,
            "dispatch": {
                "demand": last["demand"], "pv": last["pv"], "battery": last["battery"],
                "genset": last["genset"], "grid": last["grid"], "unmet": last["unmet"],
                "pv_curtailment": last["curtail"], "by_building": by_building,
            },
            "soc": last["soc"],
            "step_idx": len(hist),
            "horizon": self.horizon,
            "city": self._cities[i],
            "title": "Microgrid Sizing & Dispatch",
            "subtitle": (
                f"{conn} · {nb} building{'s' if nb > 1 else ''} "
                f"({', '.join(buildings)}) · {self._cities[i]} · "
                f"hour {len(hist)}/{self.horizon}"
            ),
        }

        if self.render_mode == "human":
            import matplotlib.pyplot as plt
            if self._hfigs[i] is None:
                plt.ion()
                self._hfigs[i] = render_env_figure(state, rec)
                self._hfigs[i].show()
            else:
                render_env_figure(state, rec, fig=self._hfigs[i])
                self._hfigs[i].canvas.draw_idle()
            plt.pause(max(self.render_pause, 1e-3))
            return None

        from matplotlib.backends.backend_agg import FigureCanvasAgg
        import matplotlib.pyplot as plt
        fig = render_env_figure(state, rec)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgb = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return rgb

    def render(self):
        """Render every sub-env; returns a tuple of per-env frames (Gymnasium API)."""
        return tuple(self._render_lane(i) for i in range(self.num_envs))

    def close_extras(self, **kwargs):
        if any(f is not None for f in self._hfigs):
            import matplotlib.pyplot as plt
            for i, f in enumerate(self._hfigs):
                if f is not None:
                    plt.close(f)
                    self._hfigs[i] = None
