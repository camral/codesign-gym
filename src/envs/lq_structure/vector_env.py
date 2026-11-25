"""Codesign-aware synchronous vectorized environment.

``LQStructureVectorEnv`` holds ``num_envs`` independent
:class:`~envs.lq_structure.env.LQStructureEnv` instances (each a ``CoDesignEnv``) and
plumbs the embodiment through both ``reset`` and ``step`` — Gymnasium's generic vector
wrappers forward only ``actions`` and cannot carry a batch of designs. It is the native
vectorization for this env; ``codesign_gym.make_vec`` returns it directly rather than
wrapping the env in a Sync/Async vector env.

  - ``reset(embodiment=...)`` and ``step(actions, embodiment=...)`` accept a batch of
    designs — a list of ``num_envs`` embodiment dicts (entries may be ``None``) or a
    single "stacked" dict whose values have a leading ``num_envs`` axis.
  - ``reset`` returns ``embodiment_rejected`` as a boolean array of shape
    ``(num_envs,)``; a per-env rejection leaves that env without a plant.

Autoreset follows Gymnasium's >=1.0 semantics: an env that finished on step ``t`` is
reset on step ``t+1`` (its action is ignored that step). During an autoreset the design
supplied for that env in the ``step`` call is installed for the new episode; during a
normal step a supplied design is a transition-level change to the running episode. Per-env
``is_eval_env`` freezing is handled by each sub-env's ``CoDesignEnv`` base.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym

from core.codesign_vec_env import CoDesignVecEnv
from core.embodiment_batch import split_dict_designs
from core.seeding import lane_seeds

from .config import LQConfig
from .env import LQStructureEnv
from .designs import EMB_KEYS


def _split_embodiment(embodiment, num_envs: int) -> list[dict | None]:
    """Normalize a batch embodiment into a per-env list of dicts (or ``None``); see :func:`split_dict_designs`."""
    return split_dict_designs(embodiment, num_envs, {k: 1 for k in EMB_KEYS})


class LQStructureVectorEnv(CoDesignVecEnv):
    """Synchronous batch of LQ-structure environments with embodiment plumbing."""

    metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP,
                "render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        cfg: LQConfig,
        num_envs: int,
        is_eval_env: bool = False,
        render_mode: str | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.render_mode = render_mode
        self.cfg = cfg
        self.envs = [
            LQStructureEnv(
                cfg,
                is_eval_env=is_eval_env,
                render_mode=render_mode,
            )
            for _ in range(num_envs)
        ]
        super().__init__(
            num_envs,
            self.envs[0].observation_space,
            self.envs[0].action_space,
            self.envs[0].embodiment_space,
            self.envs[0].base_embodiment,
            is_eval_env=is_eval_env,
        )

        self._autoreset = np.zeros(num_envs, dtype=bool)
        # Each lane's latest (obs, state, embodiment, plant), what a partial reset returns for the lanes it leaves alone.
        self._last: list = [None] * num_envs

    # --- codesign extension -------------------------------------------------------

    def _validate_embodiment(self, embodiment) -> np.ndarray:
        """Validate a batch of designs; returns a boolean array of shape (num_envs,)."""
        try:
            split = _split_embodiment(embodiment, self.num_envs)
        except (TypeError, ValueError):     # not a design batch at all: every lane is invalid, never raise
            return np.zeros(self.num_envs, dtype=bool)
        valid = []
        for env, emb in zip(self.envs, split):
            if emb is None:
                valid.append(True)
                continue
            try:
                valid.append(bool(env.validate_embodiment(emb)))
            except (TypeError, ValueError, KeyError):
                valid.append(False)
        return np.asarray(valid, dtype=bool)

    # --- vector API (extended) ----------------------------------------------------

    def _reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
        embodiment=None,
    ):
        """Reset all sub-envs, optionally installing a batch of designs.

        Returns ``(obs, embodiment_rejected, info)`` with batched observations and
        a boolean rejection array of shape ``(num_envs,)``.
        """
        seeds = lane_seeds(seed, self.num_envs)

        mask = np.ones(self.num_envs, dtype=bool)
        if options is not None and "reset_mask" in options:
            mask = np.asarray(options["reset_mask"], dtype=bool)
            if mask.shape != (self.num_envs,):
                raise ValueError("reset_mask must have shape (num_envs,)")
            if any(last is None for last, keep in zip(self._last, ~mask) if keep):
                raise ValueError("a full reset is required before a partial reset")

        split = _split_embodiment(embodiment, self.num_envs)
        obs_list, rejected = [], np.zeros(self.num_envs, dtype=bool)
        states = np.zeros((self.num_envs, self.cfg.state_dim), dtype=np.float64)
        embodiments: list[dict | None] = [None] * self.num_envs
        plants: list[dict | None] = [None] * self.num_envs
        for i, env in enumerate(self.envs):
            if not mask[i]:                 # left alone: current obs, pending autoreset kept
                obs_i, states[i], embodiments[i], plants[i] = self._last[i]
                obs_list.append(obs_i)
                continue
            obs_i, rejected_i, info_i = env.reset(seed=seeds[i], embodiment=split[i])
            obs_list.append(obs_i)
            rejected[i] = bool(rejected_i)
            if not rejected[i]:
                states[i] = info_i["state"]
                embodiments[i] = info_i["embodiment"]
                plants[i] = info_i["plant"]
            self._autoreset[i] = False
            self._last[i] = (obs_i, states[i].copy(), embodiments[i], plants[i])
        obs = np.stack(obs_list, axis=0)
        info = {"state": states, "embodiment": embodiments, "plant": plants}
        return obs, rejected, info

    def _step(self, actions: np.ndarray, embodiment=None):
        """Step all sub-envs; autoreset finished ones (Gymnasium >=1.0 semantics).

        ``actions`` has shape ``(num_envs, action_dim)``. Returns stacked
        ``(obs, rewards, terminateds, truncateds, info)``.
        """
        actions = np.asarray(actions)
        split = _split_embodiment(embodiment, self.num_envs)

        obs_list = []
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        terminateds = np.zeros(self.num_envs, dtype=bool)
        truncateds = np.zeros(self.num_envs, dtype=bool)
        rejected = np.zeros(self.num_envs, dtype=bool)
        n_actuators = np.zeros(self.num_envs, dtype=np.int64)
        stage_cost = np.zeros(self.num_envs, dtype=np.float64)
        states = np.zeros((self.num_envs, self.cfg.state_dim), dtype=np.float64)
        # actuator count varies per env, so applied control is a per-env list.
        applied_control: list[np.ndarray | None] = [None] * self.num_envs
        embodiments: list[dict | None] = [None] * self.num_envs
        plants: list[dict | None] = [None] * self.num_envs

        for i, env in enumerate(self.envs):
            if self._autoreset[i]:
                # New episode: ignore the action; install this step's design, or keep
                # the lane's current one (a sub-env reset without a design would
                # fall back to the base design).
                design = split[i] if split[i] is not None else env._emb_copy()
                obs_i, rejected_i, info_i = env.reset(embodiment=design)
                obs_list.append(obs_i)
                rejected[i] = bool(rejected_i)
                if not rejected[i]:
                    states[i] = info_i["state"]
                    embodiments[i] = info_i["embodiment"]
                    plants[i] = info_i["plant"]
                self._autoreset[i] = False
            else:
                obs_i, r_i, term_i, trunc_i, info_i = env.step(actions[i], embodiment=split[i])
                obs_list.append(obs_i)
                rewards[i] = r_i
                terminateds[i] = term_i
                truncateds[i] = trunc_i
                n_actuators[i] = info_i.get("n_actuators", 0)
                stage_cost[i] = info_i.get("stage_cost", 0.0)
                states[i] = info_i["state"]
                applied_control[i] = info_i["applied_control"]
                embodiments[i] = info_i["embodiment"]
                plants[i] = info_i["plant"]
                self._autoreset[i] = term_i or trunc_i
            self._last[i] = (obs_list[-1], states[i].copy(), embodiments[i], plants[i])

        obs = np.stack(obs_list, axis=0)
        info = {
            "embodiment_rejected": rejected,
            "n_actuators": n_actuators,
            "state": states,
            "applied_control": applied_control,
            "stage_cost": stage_cost,
            "embodiment": embodiments,
            "plant": plants,
        }
        return obs, rewards, terminateds, truncateds, info

    def render(self):
        """Render every sub-env; returns a tuple of per-env frames (Gymnasium API)."""
        return tuple(env.render() for env in self.envs)

    def close_extras(self, **kwargs):
        for env in self.envs:
            env.close()
