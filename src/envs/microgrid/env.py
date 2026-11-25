"""Core single-instance microgrid co-design environment.

A ``core.codesign_env.CoDesignEnv`` that wraps a pymgrid ``Microgrid`` (rebuilt
from the current embodiment) and adds the Codesign Gym extensions: an
``embodiment_space``, embodiment-aware ``_reset`` / ``_step`` hooks, capex
accounting, action/observation padding, and (optional) transition-level
embodiment changes with SoC projection.

The base class owns the public ``reset`` / ``step`` and the ``is_eval_env``
freeze (an ``embodiment`` passed to ``step`` is nulled by the base when
``is_eval_env`` is True). This subclass implements the ``_reset`` / ``_step``
hooks, returning ``(obs, embodiment_rejected, info)`` and the standard 5-tuple
respectively.

Conventions (verified against python-microgrid 1.4.1):
  - battery control  : + = discharge, - = charge (absolute kW)
  - grid control     : + = import,    - = export  (absolute kW)
  - genset control   : [status, production_kw]     (absolute kW)
  - ``Microgrid.step`` reward is ``-(operating cost)`` and already includes grid
    cost/credit, genset fuel, and the unmet/over-generation penalty via the
    auto-added balancing module.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.codesign_env import CoDesignEnv

from . import _compat  # noqa: F401 (load pymgrid with compatibility handling first)
from pymgrid import Microgrid
from pymgrid.modules import (
    BatteryModule,
    GensetModule,
    GridModule,
    LoadModule,
    RenewableModule,
)

from . import embodiment as emb_mod
from .data import ProfileSampler, Window

_EPS = 1e-6  # floor for zero-capacity modules so pymgrid does not divide by zero

# Core (pre-padding) observation layout:
#   [soc, demand_kw, pv_avail_kw, hour_frac, grid_import_price]
_CORE_OBS_DIM = 5
_ACTION_DIM = 3  # [battery, genset, grid], each in [-1, 1]


class MicrogridCodesignEnv(CoDesignEnv):
    """Single microgrid sizing-and-dispatch co-design environment."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        horizon: int = 24,
        *,
        is_eval_env: bool = False,
        embodiment_bounds: Optional[Tuple[Mapping[str, float], Mapping[str, float]]] = None,
        sampler: Optional[ProfileSampler] = None,
        demand_peak_kw: float = emb_mod.DEFAULT_DEMAND_PEAK_KW,
        n_buildings: int = 1,
        per_building_peak_kw: Optional[float] = None,
        off_grid: bool = False,
        # economics
        capex_coeffs: Optional[Mapping[str, float]] = None,
        lifetime_years: float = emb_mod.DEFAULT_LIFETIME_YEARS,
        import_price: float = 0.30,
        export_price: float = 0.05,
        genset_fuel_cost: float = 0.40,
        unmet_penalty: float = 10.0,
        overgen_penalty: float = 2.0,
        grid_max_import: Optional[float] = None,
        grid_max_export: Optional[float] = None,
        # transition-level embodiment
        allow_transition_embodiment: bool = True,
        # forecast augmentation
        forecast_horizon: int = 0,
        forecast_noise: float = 0.0,
        render_mode: Optional[str] = None,
        render_pause: float = 0.4,   # seconds per hour in human (live) mode
    ):
        self.horizon = int(horizon)
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.render_pause = float(render_pause)
        self._hfig = None  # persistent figure reused for live human rendering

        self.demand_peak_kw = float(demand_peak_kw)
        if embodiment_bounds is None:
            low, high = emb_mod.default_bounds(self.demand_peak_kw)
        else:
            low, high = embodiment_bounds
        embodiment_space: spaces.Dict = emb_mod.make_embodiment_space(low, high)
        # Base sets self.embodiment_space and self.is_eval_env (and gym.Env init).
        super().__init__(embodiment_space=embodiment_space, is_eval_env=is_eval_env)
        self._base_embodiment = emb_mod.base_embodiment(self.embodiment_space)

        self.n_buildings = int(n_buildings)
        self.off_grid = bool(off_grid)
        pb = per_building_peak_kw if per_building_peak_kw is not None else self.demand_peak_kw / max(self.n_buildings, 1)
        self.sampler = sampler or ProfileSampler(self.horizon, n_buildings=self.n_buildings, per_building_peak_kw=pb)

        # economics
        self.capex_coeffs = dict(capex_coeffs or emb_mod.DEFAULT_CAPEX_COEFFS)
        self.lifetime_years = float(lifetime_years)
        self.import_price = float(import_price)
        self.export_price = float(export_price)
        self.genset_fuel_cost = float(genset_fuel_cost)
        self.unmet_penalty = float(unmet_penalty)
        self.overgen_penalty = float(overgen_penalty)
        if self.off_grid:
            grid_limit = 0.0  # no utility connection: must self-supply
        else:
            grid_limit = emb_mod.GRID_LIMIT_MULTIPLE_OF_PEAK * self.demand_peak_kw
        self.grid_max_import = float(grid_max_import) if grid_max_import is not None else grid_limit
        self.grid_max_export = float(grid_max_export) if grid_max_export is not None else grid_limit

        # transition-level embodiment: never allowed in an eval env (API).
        self.allow_transition_embodiment = bool(allow_transition_embodiment) and not self.is_eval_env

        # forecast
        self.forecast_horizon = int(forecast_horizon)
        self.forecast_noise = float(forecast_noise)

        # --- spaces -------------------------------------------------------
        core_obs_dim = _CORE_OBS_DIM + 2 * self.forecast_horizon
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(_ACTION_DIM,), dtype=np.float32)
        self.observation_space = self._build_observation_space(core_obs_dim)
        self._core_obs_dim = core_obs_dim

        # --- runtime state ------------------------------------------------
        self._mg: Optional[Microgrid] = None
        self._batt_mod = None
        self._mg_embodiment: Optional[emb_mod.EmbType] = None  # design the cached mg was built for
        self._window: Optional[Window] = None
        self._embodiment: emb_mod.EmbType = dict(self._base_embodiment)
        self._t = 0
        self._first_step = True
        self._amort_capex = 0.0
        self._valid = False
        self._rejected = False
        self._warned_transition = False
        self._history: list = []  # per-step records for render()

    # ------------------------------------------------------------------ #
    # Space construction
    # ------------------------------------------------------------------ #
    def _build_observation_space(self, core_dim: int) -> spaces.Box:
        return spaces.Box(low=np.zeros(core_dim, np.float32), high=np.full(core_dim, np.inf, np.float32), shape=(core_dim,), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Codesign API helpers
    # ------------------------------------------------------------------ #
    @property
    def base_embodiment(self) -> emb_mod.EmbType:
        return dict(self._base_embodiment)

    def validate_embodiment(self, embodiment: Mapping[str, float]) -> bool:
        """Base space check, then the microgrid constraints. Plain numbers become arrays of the space dtype first, so
        Gymnasium's ``contains`` does not warn about casting them."""
        if isinstance(embodiment, Mapping) and set(embodiment) == set(self.embodiment_space.spaces):
            embodiment = {k: np.asarray(v, dtype=self.embodiment_space[k].dtype) for k, v in embodiment.items()}
        return super().validate_embodiment(embodiment)

    def _validate_embodiment(self, embodiment: Mapping[str, float]) -> bool:
        """Apply microgrid-specific constraints after the base space check."""
        return emb_mod.validate_embodiment(embodiment, self.embodiment_space)

    # ------------------------------------------------------------------ #
    # Microgrid construction
    # ------------------------------------------------------------------ #
    def _grid_price_series(self, length: int) -> np.ndarray:
        return np.column_stack([np.full(length, self.import_price), np.full(length, self.export_price), np.zeros(length)])

    def _build_microgrid(
        self,
        embodiment: Mapping[str, float],
        demand: np.ndarray,
        pv_avail: np.ndarray,
        init_charge_kwh: Optional[float],
        init_soc: Optional[float],
    ) -> Microgrid:
        # python-microgrid 1.4.1 dispatches from index 0 and runs exactly
        # len(series) steps (no off-by-one, no read past the end), so the
        # exogenous series length equals the horizon with no buffer padding.
        kwh = float(embodiment["battery_capacity_kwh"])
        gen_kw = float(embodiment["genset_rating_kw"])
        power = emb_mod.BATTERY_C_RATE * kwh

        battery = BatteryModule(
            min_capacity=0.0,
            max_capacity=max(kwh, _EPS),
            max_charge=max(power, _EPS),
            max_discharge=max(power, _EPS),
            efficiency=emb_mod.BATTERY_EFFICIENCY,
            init_charge=init_charge_kwh,
            init_soc=(init_soc if init_charge_kwh is None else None),
        )
        genset = GensetModule(running_min_production=0.0, running_max_production=max(gen_kw, _EPS), genset_cost=self.genset_fuel_cost)
        length = len(demand)
        grid = GridModule(max_import=self.grid_max_import, max_export=self.grid_max_export, time_series=self._grid_price_series(length))
        mg = Microgrid([battery, genset, ("pv", RenewableModule(pv_avail)), LoadModule(demand), grid])
        # Configure the auto-added balancing (unmet/over-generation) penalties.
        bal = mg.modules["balancing"][0]
        bal.loss_load_cost = self.unmet_penalty
        bal.overgeneration_cost = self.overgen_penalty
        mg.reset()
        self._batt_mod = mg.modules["battery"][0]
        return mg

    def _dispatch_breakdown(self, run_info: dict, t: int) -> Dict[str, float]:
        """Signed per-source energy for step ``t`` (kW over the 1-hour step).

        Sign convention (positive = supplied toward the load):
        battery + discharge / - charge; grid + import / - export.
        """
        def net(module: str) -> float:
            if module not in run_info:
                return 0.0
            d = run_info[module][0]
            return float(d.get("provided_energy", 0.0)) - float(d.get("absorbed_energy", 0.0))

        pv_entry = run_info.get("pv", [{}])[0]
        bal = net("balancing")
        return {"demand": float(self._window.demand_kw[t]), "pv": net("pv"), "pv_curtailment": float(pv_entry.get("curtailment", 0.0)),
                "battery": net("battery"), "genset": net("genset"), "grid": net("grid"), "unmet": max(bal, 0.0), "overgen": max(-bal, 0.0)}

    def _current_soc(self) -> float:
        return float(self._batt_mod.soc)

    def _current_stored_kwh(self) -> float:
        return self._current_soc() * float(self._embodiment["battery_capacity_kwh"])

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def _forecast(self, series: np.ndarray, t: int) -> np.ndarray:
        """Next ``forecast_horizon`` values of a window series with additive
        Gaussian noise (noise scaled by the local magnitude)."""
        h = self.forecast_horizon
        idx = np.minimum(t + 1 + np.arange(h), len(series) - 1)
        vals = series[idx].astype(np.float64)
        if self.forecast_noise > 0:
            vals = vals + self.np_random.normal(0.0, self.forecast_noise * (np.abs(vals) + 1e-6))
        return np.clip(vals, 0.0, None)

    def _observe(self) -> np.ndarray:
        w = self._window
        t = min(self._t, self.horizon - 1)
        pv_avail = float(self._embodiment["pv_capacity_kw"]) * float(w.solar_cf[t])
        core = [self._current_soc(), float(w.demand_kw[t]), pv_avail, float(w.hour_of_day[t]) / 24.0, self.import_price]
        if self.forecast_horizon > 0:
            core.extend(self._forecast(w.demand_kw, t).tolist())
            pv_series = float(self._embodiment["pv_capacity_kw"]) * w.solar_cf
            core.extend(self._forecast(pv_series, t).tolist())
        return np.asarray(core, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Action -> pymgrid control
    # ------------------------------------------------------------------ #
    def _control_from_action(self, action: np.ndarray) -> Dict[str, list]:
        a = np.asarray(action, dtype=np.float64).ravel()[:_ACTION_DIM]
        a = np.clip(a, -1.0, 1.0)
        a_batt, a_gen, a_grid = a[0], a[1], a[2]

        power = emb_mod.BATTERY_C_RATE * float(self._embodiment["battery_capacity_kwh"])
        batt_cmd = a_batt * power  # + discharge / - charge

        gen_kw = max(a_gen, 0.0) * float(self._embodiment["genset_rating_kw"])
        gen_status = 1.0 if gen_kw > _EPS else 0.0

        grid_cmd = a_grid * self.grid_max_import if a_grid >= 0 else a_grid * self.grid_max_export
        return {"genset": [np.array([gen_status, gen_kw])], "battery": [batt_cmd], "grid": [grid_cmd]}

    # ------------------------------------------------------------------ #
    # CoDesignEnv hook: reset
    # ------------------------------------------------------------------ #
    def _reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        embodiment: Optional[Mapping[str, float]] = None,
    ) -> Tuple[np.ndarray, bool, dict]:
        """Reset the episode, optionally installing a new embodiment.

        Returns ``(obs, embodiment_rejected, info)``. A rejected embodiment leaves
        the environment invalid and makes a subsequent ``step`` raise.
        """
        # Seed gym.Env's RNG without recursing through CoDesignEnv.reset.
        gym.Env.reset(self, seed=seed)

        emb = dict(embodiment) if embodiment is not None else dict(self._base_embodiment)
        if not self._validate_embodiment(emb):
            self._rejected = True
            self._valid = False
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, True, {}

        self._embodiment = emb
        self._window = self.sampler.sample(self.np_random)
        pv_avail = float(emb["pv_capacity_kw"]) * self._window.solar_cf
        # Fast path: pymgrid bakes capacities/bounds at construction, but re-reads
        # the exogenous time-series on reset. So when the embodiment is unchanged
        # we only swap the profiles and reset -- no costly Microgrid rebuild.
        # (Rebuild only when the design changes; the cache is invalidated after a
        # transition-level change, which leaves a shorter, sliced microgrid.)
        if self._mg is not None and self._mg_embodiment == emb:
            self._mg.modules["load"][0].time_series = self._window.demand_kw
            self._mg.modules["pv"][0].time_series = pv_avail
            # pymgrid's reset rewinds the step but keeps the battery where the last
            # episode left it; every episode starts at BATTERY_INIT_SOC.
            self._mg.modules["battery"][0].soc = emb_mod.BATTERY_INIT_SOC
            self._mg.reset()
        else:
            self._mg = self._build_microgrid(emb, self._window.demand_kw, pv_avail, init_charge_kwh=None, init_soc=emb_mod.BATTERY_INIT_SOC)
            self._mg_embodiment = dict(emb)
        self._t = 0
        self._first_step = True
        self._rejected = False
        self._valid = True
        self._history = []
        self._amort_capex = emb_mod.amortized_capex(emb, self.horizon, self.capex_coeffs, self.lifetime_years)

        info = {
            "embodiment": dict(emb),
            "capex": self._amort_capex,
            "capex_total": emb_mod.capex(emb, self.capex_coeffs),
            "city": self._window.city,
            "building": self._window.building,
        }
        return self._observe(), False, info

    # ------------------------------------------------------------------ #
    # transition-level embodiment change
    # ------------------------------------------------------------------ #
    def _apply_transition_embodiment(self, new_emb: Mapping[str, float]) -> float:
        """Rebuild the microgrid for a mid-episode capacity change, projecting
        SoC by preserving stored kWh (clamped to the new capacity).

        Retrofit capex is deliberately *not* charged to the reward (user
        decision): only the initial embodiment's capex is paid (first step).
        The incremental amortized capex of the change is returned so the caller
        can surface it in ``info`` -- users shape their own reward with it.
        """
        old_emb = self._embodiment
        stored = self._current_stored_kwh()
        new_cap = float(new_emb["battery_capacity_kwh"])
        projected = min(stored, new_cap)

        remaining = self.horizon - self._t
        demand = self._window.demand_kw[self._t :]
        pv_avail = float(new_emb["pv_capacity_kw"]) * self._window.solar_cf[self._t :]
        self._mg = self._build_microgrid(new_emb, demand, pv_avail, init_charge_kwh=projected, init_soc=None)
        self._mg_embodiment = None  # sliced/remaining series -> don't reuse next reset
        self._embodiment = dict(new_emb)

        # Incremental capital added, amortized over the remaining horizon --
        # reported in info only, NOT charged to reward.
        delta_total = emb_mod.capex(new_emb, self.capex_coeffs) - emb_mod.capex(old_emb, self.capex_coeffs)
        frac = remaining / (self.lifetime_years * emb_mod.HOURS_PER_YEAR)
        incremental = max(delta_total, 0.0) * frac
        # Refresh the reported per-step amortized capex to the new embodiment.
        self._amort_capex = emb_mod.amortized_capex(new_emb, self.horizon, self.capex_coeffs, self.lifetime_years)
        return incremental

    # ------------------------------------------------------------------ #
    # CoDesignEnv hook: step
    # ------------------------------------------------------------------ #
    def _step(
        self,
        action: np.ndarray,
        embodiment: Optional[Mapping[str, float]] = None,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Advance one step, optionally applying a transition-level design change.

        The base class nulls ``embodiment`` here when ``is_eval_env`` is True, so
        in an evaluation env the design is frozen.
        """
        if not self._valid:
            raise RuntimeError("step() called on an invalid environment (embodiment was rejected " "at reset, or reset was never called).")

        capex_charged = 0.0
        capex_delta = 0.0

        # --- transition-level embodiment change --------------------------
        if embodiment is not None:
            if not self.allow_transition_embodiment:
                if not self._warned_transition:
                    warnings.warn(
                        "This environment does not support transition-level "
                        "embodiment changes; the provided embodiment is ignored."
                    )
                    self._warned_transition = True
            else:
                if not self.validate_embodiment(embodiment):
                    raise ValueError(f"Invalid transition embodiment: {embodiment}")
                new_emb = {k: float(embodiment[k]) for k in emb_mod.EMBODIMENT_KEYS}
                if new_emb != self._embodiment:
                    # Reported in info only; NOT added to the reward (user decision).
                    capex_delta = self._apply_transition_embodiment(new_emb)

        # --- capex lump on first step ------------------------------------
        if self._first_step:
            capex_charged += self._amort_capex
            self._first_step = False

        # --- dispatch ----------------------------------------------------
        control = self._control_from_action(action)
        _, op_reward, _, run_info = self._mg.step(control, normalized=False)
        op_reward = float(op_reward)
        dispatch = self._dispatch_breakdown(run_info, self._t)
        soc_after = self._current_soc()

        reward = op_reward - capex_charged
        self._t += 1
        truncated = self._t >= self.horizon
        terminated = False

        info = {
            "embodiment_rejected": False,
            "op_cost": -op_reward,
            "capex": self._amort_capex,
            "capex_total": emb_mod.capex(self._embodiment, self.capex_coeffs),
            "capex_charged": capex_charged,
            "capex_delta": capex_delta,  # retrofit incremental capex (info only)
            "embodiment": dict(self._embodiment),
            "soc": soc_after,
            "dispatch": dispatch,
        }
        if self.render_mode is not None:
            self._history.append({
                "hour": len(self._history),
                "demand": dispatch["demand"], "pv": dispatch["pv"],
                "battery": dispatch["battery"], "genset": dispatch["genset"],
                "grid": dispatch["grid"], "unmet": dispatch["unmet"],
                "overgen": dispatch["overgen"], "curtail": dispatch["pv_curtailment"],
                "soc": soc_after, "reward": reward, "op_cost": -op_reward,
                "capex_charged": capex_charged,
            })
        return self._observe(), reward, terminated, truncated, info

    def _records_from_history(self) -> Dict[str, np.ndarray]:
        keys = ("hour", "demand", "pv", "battery", "genset", "grid", "unmet",
                "overgen", "curtail", "soc", "reward", "op_cost", "capex_charged")
        rec = {k: np.array([h[k] for h in self._history], dtype=float) for k in keys}
        rec["embodiment"] = dict(self._embodiment)
        return rec

    def _render_state(self, rec: Dict[str, np.ndarray]) -> Dict:
        last = self._history[-1]
        buildings = self._window.building.split("+")
        n = len(buildings)
        t = len(self._history) - 1
        per_bldg = self._window.demand_by_building[t]           # (n,) kW this step
        served_frac = (last["demand"] - last["unmet"]) / max(last["demand"], 1e-6)
        by_building = [
            {"name": nm, "demand": float(dem), "served": float(dem * served_frac)}
            for nm, dem in zip(buildings, per_bldg)
        ]
        dispatch = {
            "demand": last["demand"], "pv": last["pv"], "battery": last["battery"],
            "genset": last["genset"], "grid": last["grid"], "unmet": last["unmet"],
            "pv_curtailment": last["curtail"], "by_building": by_building,
        }
        conn = "off-grid" if self.off_grid else "grid-connected"
        bounds = {
            k: (float(sp.low.flat[0]), float(sp.high.flat[0]))
            for k, sp in self.embodiment_space.spaces.items()
        }
        return {
            "embodiment": dict(self._embodiment),
            "embodiment_bounds": bounds,
            "off_grid": self.off_grid,
            "buildings": buildings,
            "demand_peak": float(max(rec["demand"].max(), 1.0)),
            "grid_max": self.grid_max_import,
            "dispatch": dispatch,
            "soc": last["soc"],
            "step_idx": len(self._history),
            "horizon": self.horizon,
            "city": self._window.city,
            "title": "Microgrid Sizing & Dispatch",
            "subtitle": (
                f"{conn} · {n} building{'s' if n > 1 else ''} "
                f"({', '.join(buildings)}) · {self._window.city} · "
                f"hour {len(self._history)}/{self.horizon}"
            ),
        }

    def render(self):
        """Render the microgrid as a single-line schematic (design + live power
        flow) beside the episode trajectory, via
        ``envs.microgrid.schematic.render_env_figure``.

        ``render_mode="rgb_array"`` returns an (H, W, 3) uint8 array;
        ``render_mode="human"`` shows the figure and returns None. Returns None
        if no step has been taken yet.
        """
        if self.render_mode not in ("human", "rgb_array") or not self._history:
            return None
        from .schematic import render_env_figure

        rec = self._records_from_history()
        state = self._render_state(rec)

        if self.render_mode == "human":
            # live update: redraw the SAME figure each call and pause, so the
            # policy's decisions are visible hour by hour as the horizon advances
            import matplotlib.pyplot as plt

            if self._hfig is None:
                plt.ion()
                self._hfig = render_env_figure(state, rec)
                self._hfig.show()
            else:
                render_env_figure(state, rec, fig=self._hfig)
                self._hfig.canvas.draw_idle()
            plt.pause(max(self.render_pause, 1e-3))
            return None

        # rgb_array: render off-screen via an Agg canvas (no display needed)
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = render_env_figure(state, rec)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        rgb = np.asarray(canvas.buffer_rgba())[..., :3].copy()
        import matplotlib.pyplot as plt

        plt.close(fig)
        return rgb

    def close(self):  # pragma: no cover
        if self._hfig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._hfig)
            self._hfig = None
        return None
