"""Codesign-aware vectorized Pokemon Showdown environment.

``PokenvVectorEnv`` is the native vectorization codesign_gym returns for the Pokemon benchmark. It
wraps the ``pokemon_codesign_env`` package's own server-owning, process-sharded vector env
(``ServerOwningVecEnv`` over ``ShardedVecEnv``): worker processes each run a thread-pool of
I/O-bound battles, across a cluster of Showdown servers the env launches and owns. That machinery is
what the package's throughput work was tuned around, so this class is a thin adapter over it rather
than a reimplementation -- it must not add per-step work on the hot path, or the effective step rate
would regress.

The adapter only reconciles the two contracts:

  - ``reset`` returns codesign_gym's ``(obs, embodiment_rejected, info)`` order; the underlying env
    returns ``(obs, info, embodiment_rejected)``.
  - ``step`` accepts an ``embodiment`` for API symmetry but the Pokemon battle locks the team for
    its duration, so a supplied design is ignored with a warning (no transition-level change). The
    action batch is forwarded straight through.
  - ``validate_embodiment`` takes a batch and returns a boolean array, one flag per env.

``embodiment_space`` is shared across the sub-envs; ``base_embodiment`` is a single representative
design (the codesign_gym contract treats it as one), with the full per-env list kept in
``base_embodiments``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.codesign_vec_env import CoDesignVecEnv
from core.seeding import lane_seeds

from pokemon_codesign_env import make_vec as _pokemon_make_vec


class PokenvVectorEnv(CoDesignVecEnv):
    """Thin codesign-contract adapter over the Pokemon package's sharded, server-owning vector env."""

    def __init__(
        self,
        num_envs: int,
        *,
        opponent: str = "random",
        is_eval_env: bool = False,
        render_mode: str | None = None,
        **pokemon_kwargs: Any,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be at least one, got {num_envs}")
        if render_mode not in (None, "html"):
            raise ValueError(f"unsupported render_mode {render_mode!r}; Pokémon supports None or 'html'")
        self.render_mode = render_mode
        self._replay_html = (None,) * int(num_envs)

        # Launches the Showdown cluster and the sharded workers; returns the server-owning vector
        # env plus one base design per env.
        self._venv, base_embodiments = _pokemon_make_vec(
            opponent=opponent,
            num_envs=int(num_envs),
            is_eval_env=is_eval_env,
            render_mode=render_mode,
            **pokemon_kwargs,
        )
        self.base_embodiments = list(base_embodiments)
        # The design space lives in the workers; every sub-env shares the same one.
        super().__init__(
            int(num_envs),
            self._venv.single_observation_space,
            self._venv.single_action_space,
            self._sharded.get_attr("embodiment_space")[0],
            self.base_embodiments[0] if self.base_embodiments else {},
            is_eval_env=is_eval_env,
        )
        # Use the package's own batched spaces + metadata (rather than the base's batch_space);
        # the team is locked for the battle, so no transition-level embodiment change.
        self.observation_space = self._venv.observation_space
        self.action_space = self._venv.action_space
        self.metadata = {**getattr(self._venv, "metadata", {}), "render_modes": ["html"], "supports_transition_embodiment": False,
                         "supports_partial_reset": False}

    @property
    def _sharded(self):
        """The underlying ShardedVecEnv, beneath any monitoring wrapper (for get_attr/env_method)."""
        return self._venv.env.unwrapped

    # ------------------------------------------------------------------ #
    # codesign API
    # ------------------------------------------------------------------ #
    def _validate_embodiment(self, embodiment: Any) -> np.ndarray:
        """Validate a batch of designs; returns a boolean array of shape ``(num_envs,)``.

        Not on the hot path -- called before a reset to screen designs -- so a per-env round-trip
        to the workers is acceptable.
        """
        designs = self._designs(embodiment)
        valid = [
            bool(self._sharded.env_method("validate_embodiment", design, indices=index)[0])
            for index, design in enumerate(designs)
        ]
        return np.asarray(valid, dtype=bool)

    def _designs(self, embodiment: Any) -> list:
        """One team per env: ``None`` (or a ``None`` entry) is that env's base team, as in the other envs. The
        package reads ``None`` as "keep the installed team", which only its own autoreset relies on."""
        designs = [None] * self.num_envs if embodiment is None else self._venv._embodiments_for_each_environment(embodiment)
        return [self.base_embodiments[i] if d is None else d for i, d in enumerate(designs)]

    # ------------------------------------------------------------------ #
    # gym vector API
    # ------------------------------------------------------------------ #
    def _reset(self, *, seed=None, options=None, embodiment=None):
        # Every lane resets together: a new team for some lanes means a new team (or the same one) for all. The
        # package's lanes autoreset on their own (SAME_STEP), so a mask is only ever an early reset, which it lacks.
        options = dict(options or {})
        if "reset_mask" in options and not np.asarray(options.pop("reset_mask"), dtype=bool).all():
            raise NotImplementedError("Pokenv resets every lane together; pass no reset_mask (or an all-True one)")
        # Each team goes in through its env's reset options, as the package's own embodiment argument does.
        per_env = [{**options, "embodiment": design} for design in self._designs(embodiment)]
        obs, infos, rejected = self._venv.reset(seed=lane_seeds(seed, self.num_envs), options=per_env)
        self._replay_html = (None,) * self.num_envs
        return obs, np.asarray(rejected, dtype=bool), infos

    def _step(self, actions, embodiment=None):
        # Base has already frozen/warned any transition embodiment (metadata: unsupported).
        # Straight passthrough: ServerOwningVecEnv.step forwards the action batch to the shards.
        result = self._venv.step(actions)
        infos = result[4]
        if self.render_mode == "html":
            values = infos.get("replay_html")
            present = infos.get("_replay_html")
            self._replay_html = tuple(
                values[i] if values is not None and (present is None or present[i]) else self._replay_html[i]
                for i in range(self.num_envs)
            )
        return result

    def render(self):
        return self._replay_html if self.render_mode == "html" else None

    def close(self, **kwargs):
        self._venv.close(**kwargs)
