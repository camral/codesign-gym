"""
Authors: Aviraj Newatia and Leonard Pleiss

Vectorized base class for co-design environments.

The batched analogue of :class:`core.codesign_env.CoDesignEnv`. gymnasium's
``VectorEnv`` is an abstract batched *interface* -- its ``__init__`` is a no-op and
it spawns nothing (subprocess parallelization lives only in ``AsyncVectorEnv``).
So each co-design env keeps its own native vectorization (a per-sub-env loop, JAX
``vmap``, GPU batching, a process-sharded server, ...), and this base only layers on
the co-design contract:

  - ``embodiment_space`` and ``base_embodiment`` (mirroring ``CoDesignEnv``);
  - ``reset(..., embodiment=...) -> (obs, embodiment_rejected, info)``;
  - ``step(actions, embodiment=...)`` with the ``is_eval_env`` / non-transition
    freeze applied via the shared :func:`core.codesign_env.resolve_step_embodiment`
    (so the warnings match the single-env base);
  - ``validate_embodiment`` over a batch of designs.

Subclasses supply their native vectorization and implement the ``_reset`` /
``_step`` / ``_validate_embodiment`` hooks. A subclass whose design cannot change
mid-episode sets ``metadata = {..., "supports_transition_embodiment": False}`` to
get the freeze/warning for free; an evaluation env (``is_eval_env=True``) is always
frozen.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import gymnasium as gym
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from .codesign_env import resolve_step_embodiment


class CoDesignVecEnv(VectorEnv):

    # Gymnasium's key for when a finished copy resets, see docs/concepts.md
    metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP}
    embodiment_space: spaces.Space

    def __init__(
        self,
        num_envs: int,
        single_observation_space: spaces.Space,
        single_action_space: spaces.Space,
        embodiment_space: spaces.Space,
        base_embodiment: Any,
        *,
        is_eval_env: bool = False,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)
        self.single_observation_space = single_observation_space
        self.single_action_space = single_action_space
        self.observation_space = batch_space(single_observation_space, self.num_envs)
        self.action_space = batch_space(single_action_space, self.num_envs)
        # Codesign extensions, shared across the batch.
        self.embodiment_space = embodiment_space
        self.base_embodiment = base_embodiment
        self.is_eval_env = bool(is_eval_env)

    def reset(self, *, seed: int | Sequence[int] | None = None, options: dict | None = None,
              embodiment: Any = None) -> tuple[Any, Any, dict]:
        result = self._reset(seed=seed, options=options, embodiment=embodiment)
        if getattr(self, "_render_human_automatically", False) and getattr(self, "render_mode", None) == "human":
            self.render()
        return result

    def step(self, actions: Any, embodiment: Any = None):
        embodiment = resolve_step_embodiment(self, embodiment)
        result = self._step(actions, embodiment)
        if getattr(self, "_render_human_automatically", False) and getattr(self, "render_mode", None) == "human":
            self.render()
        return result

    def validate_embodiment(self, embodiment: Any) -> Any:
        return self._validate_embodiment(embodiment)

    # ------------------------------------------------------------------ #
    # Subclass hooks (native vectorization lives here)
    # ------------------------------------------------------------------ #
    def _reset(self, *, seed=None, options: dict | None = None, embodiment: Any = None) -> tuple[Any, Any, dict]:
        raise NotImplementedError

    def _step(self, actions: Any, embodiment: Any = None):
        raise NotImplementedError

    def _validate_embodiment(self, embodiment: Any) -> Any:
        raise NotImplementedError
