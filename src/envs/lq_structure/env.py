"""The Linear-Quadratic Structure Co-Design environment.

``LQStructureEnv`` subclasses ``core.codesign_env.CoDesignEnv``: it implements the
``_reset``/``_step`` hooks while the base class owns ``reset``/``step`` and the
``is_eval_env`` freeze (an ``embodiment`` passed to ``step`` is nulled by the base when
``is_eval_env`` is True, so a fixed design cannot be reshaped mid-episode).

Beyond the base contract this env exposes:
  - ``embodiment_space`` (a gymnasium ``Dict`` space) and ``base_embodiment``.
  - ``validate_embodiment(embodiment) -> bool``.
  - ``_reset(embodiment=...)`` installing a design and returning an
    ``embodiment_rejected`` flag (a rejected design makes a subsequent ``step`` raise).
  - ``_step(action, embodiment=...)`` supporting transition-level design changes
    (full, including placement) with a fixed padded action / observation layout.
  - ``render()`` chain animation and a rich ``info`` dict for user-side visualization.

Task: disturbance regulation. reset draws a random (or impulse) initial state; each step
adds process noise; reward is the negative quadratic stage cost times ``dt``; the episode
truncates at the horizon with no early termination.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from core.codesign_env import CoDesignEnv

from .config import LQConfig
from . import designs
from .dynamics import Plant
from .rendering import ChainRenderer


class LQStructureEnv(CoDesignEnv):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(
        self,
        cfg: LQConfig,
        is_eval_env: bool = False,
        render_mode: str | None = None,
    ) -> None:
        self.cfg = cfg
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self._renderer = None
        embodiment_space: spaces.Dict = designs.build_embodiment_space(cfg)
        super().__init__(embodiment_space=embodiment_space, is_eval_env=is_eval_env)

        self.base_embodiment: dict[str, np.ndarray] = designs.base_embodiment(cfg)

        # Fixed-size spaces, independent of placement: one action slot per possible actuator (the live
        # ones take the first slots), and the natural 4N observation (see designs.observation_dim).
        u_bound = cfg.u_max if cfg.u_max is not None else np.inf
        self.action_space = spaces.Box(low=-u_bound, high=u_bound, shape=(cfg.n,), dtype=np.float64)
        obs_dim = designs.observation_dim(cfg)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64)

        # Runtime state (set on reset).
        self._emb: dict[str, np.ndarray] = self.base_embodiment
        self._plant: Plant | None = None
        self._x: np.ndarray | None = None
        self._t: int = 0
        self._rejected: bool = False

    # --- codesign extension: validation -------------------------------------------

    def _validate_embodiment(self, embodiment: dict) -> bool:
        """Apply setup-specific constraints after the base space check."""
        ok, _ = designs.validate_embodiment(embodiment, self.cfg)
        return ok

    def _build_plant(self, emb: dict[str, np.ndarray]) -> None:
        self._plant = Plant(
            masses=emb["masses"], stiffness=emb["stiffness"], damping=emb["damping"],
            actuators=emb["actuators"], sensors=emb["sensors"], dt=self.cfg.dt,
        )

    def _plant_info(self) -> dict[str, np.ndarray | int]:
        """Plant matrices for the design in effect (for user-side ground truth)."""
        p = self._plant
        return {
            "A_cont": p.A_cont.copy(), "Ad": p.Ad.copy(), "Bd": p.Bd.copy(),
            "Gd": p.Gd.copy(), "Cs": p.Cs.copy(),
            "n_actuators": p.n_actuators, "n_sensors": p.n_sensors,
        }

    def _emb_copy(self) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._emb.items()}

    # --- CoDesignEnv hooks --------------------------------------------------------

    def _reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None,
               embodiment: dict | None = None):
        """Reset the episode, optionally installing a new embodiment.

        Returns ``(obs, embodiment_rejected, info)``. A rejected embodiment leaves
        the plant unset and makes any subsequent ``step`` raise.
        """
        # Seed gym.Env's RNG without recursing through CoDesignEnv.reset.
        gym.Env.reset(self, seed=seed)

        if embodiment is not None:
            ok, reason = designs.validate_embodiment(embodiment, self.cfg)
            if not ok:
                self._rejected = True
                self._plant = None
                self._x = None
                return np.zeros(self.observation_space.shape, dtype=np.float64), True, {"reason": reason}
            self._emb = designs.coerce_embodiment(embodiment, self.cfg)
        else:
            self._emb = designs.coerce_embodiment(self.base_embodiment, self.cfg)

        self._rejected = False
        self._build_plant(self._emb)
        self._x = self._sample_initial_state()
        self._t = 0

        obs = self._observe()
        info = {"state": self._x.copy(), "embodiment": self._emb_copy(), "plant": self._plant_info()}
        return obs, False, info

    def _step(self, action: np.ndarray, embodiment: dict | None = None):
        """Advance one step, optionally applying a transition-level design change.

        The base class nulls ``embodiment`` here when ``is_eval_env`` is True, so in an
        evaluation env the design is frozen. Otherwise the full embodiment (including
        placement) may change each step; the plant matrices are rebuilt while the state
        ``x`` is carried over (positions and velocities stay valid across a design change).
        """
        if self._rejected or self._plant is None or self._x is None:
            raise RuntimeError(
                "step called before a successful reset (embodiment was rejected or "
                "reset not called)"
            )

        if embodiment is not None:
            ok, reason = designs.validate_embodiment(embodiment, self.cfg)
            if not ok:
                raise ValueError(f"invalid embodiment passed to step: {reason}")
            self._emb = designs.coerce_embodiment(embodiment, self.cfg)
            self._build_plant(self._emb)

        plant = self._plant
        n_act = plant.n_actuators

        # Map/saturate the padded action to the active actuators.
        u = np.asarray(action, dtype=np.float64).reshape(-1)[:n_act]
        if u.size < n_act:
            u = np.pad(u, (0, n_act - u.size))
        if self.cfg.u_max is not None:
            u = np.clip(u, -self.cfg.u_max, self.cfg.u_max)

        # Quadratic stage cost on the pre-transition state and applied control.
        Q = self.cfg.Q()
        x = self._x
        stage_cost = float(x @ Q @ x)
        if n_act > 0:
            R = self.cfg.R(n_act)
            stage_cost += float(u @ R @ u)
        reward = -stage_cost * self.cfg.dt

        # Process noise on velocities, advance the state.
        w = self._sample_process_noise()
        self._x = plant.step_state(x, u, w)
        self._t += 1

        truncated = self._t >= self.cfg.horizon
        terminated = False
        obs = self._observe()
        # Everything a user needs to reconstruct trajectory / control-effort / cost plots
        # without re-deriving them (and independent of observation_mode).
        info = {
            "embodiment_rejected": False,
            "n_actuators": n_act,
            "state": x.copy(),
            "applied_control": u.copy(),
            "stage_cost": stage_cost,
            "embodiment": self._emb_copy(),
            "plant": self._plant_info(),
        }
        return obs, reward, terminated, truncated, info

    # --- helpers ------------------------------------------------------------------

    def _sample_initial_state(self) -> np.ndarray:
        n = self.cfg.n
        if self.cfg.reset_mode == "impulse":
            x = np.zeros(2 * n)
            x[n] = 1.0  # unit velocity kick on mass 0
            return x
        pos = self.np_random.normal(0.0, self.cfg.init_pos_std, size=n)
        vel = self.np_random.normal(0.0, self.cfg.init_vel_std, size=n)
        return np.concatenate([pos, vel])

    def _sample_process_noise(self) -> np.ndarray:
        n = self.cfg.n
        if self.cfg.process_noise_std <= 0:
            return np.zeros(n)
        return self.np_random.normal(0.0, self.cfg.process_noise_std, size=n)

    def _observe(self) -> np.ndarray:
        meas_noise = None
        if self.cfg.observation_mode == "partial" and self.cfg.measurement_noise_std > 0:
            n_sen = int(self._emb["sensors"].sum())
            meas_noise = self.np_random.normal(0.0, self.cfg.measurement_noise_std, size=2 * n_sen)
        return designs.build_observation(self.cfg, self._x, self._emb, meas_noise)

    def render(self):
        """Render the chain in the configured ``render_mode``.

        ``rgb_array`` returns an ``(H, W, 3)`` uint8 frame; ``human`` draws to a window
        and returns None. Returns None if no ``render_mode`` was set or before the first
        successful reset.
        """
        if self.render_mode is None or self._plant is None or self._x is None:
            return None
        if self._renderer is None:
            self._renderer = ChainRenderer(self.cfg.n, self.render_mode, fps=self.metadata["render_fps"])
        return self._renderer.render(self._x, self._emb)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
