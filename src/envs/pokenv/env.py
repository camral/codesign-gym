"""Single-environment Pokemon Showdown co-design environment.

``PokenvCodesignEnv`` is the :class:`~core.codesign_env.CoDesignEnv` conformer for the Pokemon
Showdown benchmark: one in-process ``PokEnv`` from the ``pokemon_codesign_env`` package, which owns
the Showdown server it battles on. The embodiment is a team design drawn from ``TeamDesignSpace``;
the policy plays the battle the team is locked into.

This is the single-env building block, mirroring how the microgrid env ships both
``MicrogridCodesignEnv`` and ``MicrogridVectorEnv``. The vectorized form
(:class:`~envs.pokenv.vector_env.PokenvVectorEnv`) does not build its sub-envs from this class:
the Pokemon package parallelizes with its own process-sharded, server-owning vector env whose
workers are coupled to ``PokEnv``'s exact return convention, so reusing that machinery keeps the
throughput it was tuned for. This class is the in-process CoDesignEnv for callers who want one
environment without the vector/server-owning wrapper.

The Pokemon battle locks the team at the first step of an episode, so there is no transition-level
embodiment change: an ``embodiment`` passed to ``step`` is ignored (with a warning raised by the
underlying ``PokEnv``), consistent with the CoDesignEnv contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.codesign_env import CoDesignEnv
from pokemon_codesign_env import make as _pokemon_make


class PokenvCodesignEnv(CoDesignEnv):
    """One in-process Pokemon Showdown co-design environment, owning its Showdown server."""

    metadata = {"render_modes": ["html"]}

    def __init__(
        self,
        *,
        opponent: str = "random",
        is_eval_env: bool = False,
        render_mode: str | None = None,
        **pokemon_kwargs: Any,
    ) -> None:
        # pokemon_codesign_env.make launches and owns one Showdown server for this env.
        self._env, base_embodiment = _pokemon_make(
            opponent=opponent,
            is_eval_env=is_eval_env,
            render_mode=render_mode,
            **pokemon_kwargs,
        )
        super().__init__(embodiment_space=self._env.embodiment_space, is_eval_env=is_eval_env)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.base_embodiment = base_embodiment
        self.render_mode = render_mode

    # ------------------------------------------------------------------ #
    # CoDesignEnv hooks
    # ------------------------------------------------------------------ #
    def _reset(self, *, seed: int | None, options: dict | None, embodiment: Any) -> tuple[Any, bool, dict]:
        # PokEnv.reset returns (obs, info, embodiment_rejected) with the flag as a length-1 array;
        # the CoDesignEnv contract is (obs, rejected, info) with a scalar flag.
        obs, info, rejected = self._env.reset(seed=seed, options=options, embodiment=embodiment)
        return obs, bool(np.asarray(rejected).ravel()[0]), info

    def _step(self, action: Any, embodiment: Any):
        # PokEnv itself warns and ignores a transition-level embodiment (the team is locked for the
        # battle), so delegating keeps a single source of that behavior.
        return self._env.step(action, embodiment=embodiment)

    def _validate_embodiment(self, embodiment: Any) -> bool:
        return bool(self._env.validate_embodiment(embodiment))

    # ------------------------------------------------------------------ #
    # passthrough
    # ------------------------------------------------------------------ #
    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()
