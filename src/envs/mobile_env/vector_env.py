"""Codesign-aware synchronous vectorized mobile-env environment.

``MobileEnvVectorEnv`` holds ``num_envs`` independent
:class:`~envs.mobile_env.env.MobileEnvCoDesign` instances (each a ``CoDesignEnv``)
and plumbs a batch of embodiments (base-station deployments) through both
``reset`` and ``step`` -- Gymnasium's generic vector wrappers forward only
``actions`` and cannot carry a batch of designs. It is the native vectorization
for this env; ``codesign_gym.make_vec`` returns it directly. mobile-env has no
native vectorization, so per-step compute loops over the sub-envs.

  - ``reset(embodiment=...)`` and ``step(actions, embodiment=...)`` accept a batch
    of designs: ``None`` (all base), a single embodiment dict (broadcast), or a
    list of ``num_envs`` dicts (entries may be ``None``).
  - ``reset`` returns ``embodiment_rejected`` as a boolean array of shape
    ``(num_envs,)``; a per-env rejection leaves that env invalid (a later step on
    it raises).

Base stations are fixed per episode, so an ``embodiment`` supplied to ``step`` is
ignored by each sub-env (with a warning on every such call) rather than applied
mid-episode. Autoreset follows Gymnasium's >=1.0 semantics: an env that finished
on step ``t`` is reset on ``t+1`` (its action ignored that step), reinstalling the
embodiment it was running.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import gymnasium as gym

from core.codesign_vec_env import CoDesignVecEnv
from core.embodiment_batch import split_dict_designs
from core.seeding import lane_seeds

from .config import MobileEnvConfig
from .env import MobileEnvCoDesign
from . import embodiment as emb_mod

# A batched embodiment: None, a single embodiment dict, or a sequence of per-env
# dicts (entries may be None).
BatchedEmb = Union[None, Mapping[str, Any], Sequence[Optional[Mapping[str, Any]]]]  # a Mapping may also be stacked per env


class MobileEnvVectorEnv(CoDesignVecEnv):
    """Synchronous batch of mobile-env co-design environments."""

    # Base stations are fixed for an episode -> no transition-level embodiment changes.
    metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP, "supports_transition_embodiment": False,
                "render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_envs: int,
        *,
        cfg: MobileEnvConfig,
        is_eval_env: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.render_mode = render_mode
        self.cfg = cfg
        self.envs: List[MobileEnvCoDesign] = [
            MobileEnvCoDesign(
                cfg,
                is_eval_env=is_eval_env,
                render_mode=render_mode,
            )
            for _ in range(num_envs)
        ]
        # Codesign extension: spaces/embodiment are shared across sub-envs.
        super().__init__(
            num_envs,
            self.envs[0].observation_space,
            self.envs[0].action_space,
            self.envs[0].embodiment_space,
            self.envs[0].base_embodiment,
            is_eval_env=is_eval_env,
        )

        self._autoreset = np.zeros(self.num_envs, dtype=bool)
        # Each lane's latest obs/info, what a partial reset returns for the lanes it leaves alone.
        self._last_obs: list = [None] * self.num_envs
        self._last_info: list = [{}] * self.num_envs
        self._current_embodiments: List[Optional[emb_mod.EmbType]] = [None] * self.num_envs

    # ------------------------------------------------------------------ #
    # Codesign API helpers
    # ------------------------------------------------------------------ #
    def _validate_embodiment(self, embodiment: BatchedEmb) -> np.ndarray:
        """Validate a batch of designs; returns a boolean array of shape (num_envs,).

        A ``None`` entry (no design supplied for that env) validates as True.
        """
        try:
            embs = self._split_embodiment(embodiment)
        except (TypeError, ValueError):     # not a design batch at all: every lane is invalid, never raise
            return np.zeros(self.num_envs, dtype=bool)
        valid = []
        for env, emb in zip(self.envs, embs):
            if emb is None:
                valid.append(True)
                continue
            try:
                valid.append(bool(env.validate_embodiment(emb)))
            except (TypeError, ValueError, KeyError):
                valid.append(False)
        return np.asarray(valid, dtype=bool)

    def _split_embodiment(self, embodiment: BatchedEmb) -> List[Optional[emb_mod.EmbType]]:
        """Normalize any accepted batched-embodiment form to a length-N list of per-env dicts (or ``None``);
        see :func:`core.embodiment_batch.split_dict_designs`."""
        return split_dict_designs(embodiment, self.num_envs, {"num_bs": 0, "stations": 2})

    # ------------------------------------------------------------------ #
    # reset
    # ------------------------------------------------------------------ #
    def _reset(
        self,
        *,
        seed: Optional[Union[int, Sequence[int]]] = None,
        options: Optional[dict] = None,
        embodiment: BatchedEmb = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Reset all sub-envs, optionally installing a batch of designs.

        Returns ``(obs, embodiment_rejected, info)`` with batched observations and
        a boolean rejection array of shape ``(num_envs,)``.
        """
        seeds = self._seed_list(seed)
        embs = self._split_embodiment(embodiment)

        options = dict(options or {})
        mask = np.ones(self.num_envs, dtype=bool)
        if "reset_mask" in options:
            mask = np.asarray(options.pop("reset_mask"), dtype=bool)
            if mask.shape != (self.num_envs,):
                raise ValueError("reset_mask must have shape (num_envs,)")
            if any(o is None for o, keep in zip(self._last_obs, ~mask) if keep):
                raise ValueError("a full reset is required before a partial reset")

        obs_list, infos = [], []
        rejected = np.zeros(self.num_envs, dtype=bool)
        for i, env in enumerate(self.envs):
            if not mask[i]:                 # left alone: current obs, pending autoreset kept
                obs_list.append(self._last_obs[i])
                infos.append(self._last_info[i])
                continue
            o, rejected_i, info = env.reset(seed=seeds[i], options=options, embodiment=embs[i])
            obs_list.append(o)
            infos.append(info)
            rejected[i] = bool(rejected_i)
            self._current_embodiments[i] = None if rejected_i else env._embodiment
            self._autoreset[i] = False
            self._last_obs[i], self._last_info[i] = o, info

        obs = np.stack(obs_list).astype(np.float32)
        return obs, rejected, self._stack_infos(infos)

    # ------------------------------------------------------------------ #
    # step
    # ------------------------------------------------------------------ #
    def _step(
        self,
        actions: np.ndarray,
        embodiment: BatchedEmb = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Step all sub-envs; autoreset finished ones (Gymnasium >=1.0 semantics).

        ``actions`` has shape ``(num_envs, num_ues)``. Returns stacked
        ``(obs, rewards, terminateds, truncateds, info)``.
        """
        actions = np.asarray(actions)
        embs = self._split_embodiment(embodiment)

        obs_list, infos = [], []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terms = np.zeros(self.num_envs, dtype=bool)
        truncs = np.zeros(self.num_envs, dtype=bool)
        for i, env in enumerate(self.envs):
            if self._autoreset[i]:
                # New episode: ignore the action; reinstall the embodiment this env
                # was running (transition-level changes are unsupported here).
                emb_i = self._current_embodiments[i]
                o, rejected_i, info = env.reset(embodiment=emb_i)
                info["embodiment_rejected"] = bool(rejected_i)
                self._current_embodiments[i] = None if rejected_i else env._embodiment
                self._autoreset[i] = False
            else:
                o, r, term, trunc, info = env.step(actions[i], embodiment=embs[i])
                rewards[i] = r
                terms[i] = term
                truncs[i] = trunc
                self._autoreset[i] = term or trunc
            obs_list.append(o)
            infos.append(info)
            self._last_obs[i], self._last_info[i] = o, info

        obs = np.stack(obs_list).astype(np.float32)
        return obs, rewards, terms, truncs, self._stack_infos(infos)

    def render(self):
        """Render every sub-env; returns a tuple of per-env frames (Gymnasium API)."""
        return tuple(env.render() for env in self.envs)

    # ------------------------------------------------------------------ #
    # utilities
    # ------------------------------------------------------------------ #
    def _seed_list(self, seed) -> List[Optional[int]]:
        return lane_seeds(seed, self.num_envs)

    @staticmethod
    def _stack_infos(infos: List[dict]) -> dict:
        """Aggregate per-env info dicts: scalar keys -> arrays, ``action_mask`` ->
        a stacked ``(num_envs, ...)`` array, everything else -> lists."""
        out: Dict[str, Any] = {}
        keys = set().union(*(d.keys() for d in infos)) if infos else set()
        for k in keys:
            vals = [d.get(k, None) for d in infos]
            if k == "action_mask" and all(isinstance(v, np.ndarray) for v in vals):
                out[k] = np.stack(vals)
            elif all(isinstance(v, (int, float, bool, np.floating, np.integer)) for v in vals):
                out[k] = np.array(vals)
            else:
                out[k] = vals
        return out

    def close_extras(self, **kwargs):
        for env in self.envs:
            env.close()
