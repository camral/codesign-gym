"""
Authors: Aviraj Newatia

Single-environment squeeze adapter over a num_envs=1 vector env.

``codesign_gym.make`` is ``make_vec(env_id, num_envs=1)`` wrapped in :class:`SingleEnv`,
which removes the leading batch axis so single-env consumers need no ``[0]`` indexing:

  - ``reset`` accepts a single embodiment and returns an unbatched observation,
    scalar rejection flag, and unwrapped ``info``; ``step`` returns Gymnasium's
    usual five values.
  - ``action`` may be passed unbatched (shape ``(A,)``); it is unsqueezed to ``(1, A)``.

The adapter is env-agnostic: it only manipulates the batch dimension, so it works over any
``build_vec`` result that implements the co-design contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def extract_single(value: Any) -> Any:
    """Return the single-env slice of a batched info value (leaves scalars alone)."""
    if isinstance(value, dict):
        # A Dict observation (or a nested info dict) is batched per key, so squeeze each value
        # rather than returning the mapping untouched -- otherwise a Dict-obs env's single-env
        # observation keeps its leading (1,) axis.
        return {k: extract_single(x) for k, x in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, np.ndarray):
        return value[0] if value.ndim else value
    if isinstance(value, (list, tuple)):
        return value[0] if len(value) else value
    if hasattr(value, "__len__"):  # e.g. torch tensor
        try:
            return value[0]
        except Exception:
            return value
    return value


def squeeze_info(info: dict) -> dict:
    return {k: extract_single(v) for k, v in info.items()}


class SingleEnv:
    """Squeeze wrapper presenting a num_envs=1 vector env as a single env."""

    def __init__(self, vector_env):
        self.vector_env = vector_env
        self.observation_space = getattr(vector_env, "single_observation_space", None)
        self.action_space = getattr(vector_env, "single_action_space", None)
        self.embodiment_space = getattr(vector_env, "embodiment_space", None)
        self.base_embodiment = getattr(vector_env, "base_embodiment", None)
        if getattr(vector_env, "base_embodiment_is_batched", False):
            self.base_embodiment = extract_single(self.base_embodiment)
        self.metadata = getattr(vector_env, "metadata", {})
        self.render_mode = getattr(vector_env, "render_mode", None)

    def __getattr__(self, name):
        # Passthrough for env-specific attributes (e.g. cfg) not surfaced above.
        # __getattr__ runs only when normal lookup fails, so it never shadows the
        # wrapper's own attributes/methods.
        return getattr(self.__dict__["vector_env"], name)

    # --- codesign extension -------------------------------------------------------

    def validate_embodiment(self, embodiment) -> bool:
        result = self.vector_env.validate_embodiment([embodiment])
        return bool(extract_single(result))

    # --- gym-style API (unbatched) ------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None, embodiment=None):
        embodiment = None if embodiment is None else [embodiment]
        obs, rejected, info = self.vector_env.reset(seed=seed, options=options, embodiment=embodiment)
        return extract_single(obs), bool(extract_single(rejected)), squeeze_info(info)

    def step(self, action, embodiment=None):
        a = action if hasattr(action, "detach") else np.asarray(action)
        if a.ndim == 0:
            # A scalar Discrete action stands for the length-1 batch the vector env expects.
            a = a.reshape(1)
        elif a.ndim == 1:
            a = a.reshape(1, -1)
        embodiment = None if embodiment is None else [embodiment]
        obs, reward, term, trunc, info = self.vector_env.step(a, embodiment=embodiment)
        return extract_single(obs), float(extract_single(reward)), bool(extract_single(term)), bool(extract_single(trunc)), squeeze_info(info)

    def render(self):
        out = self.vector_env.render()
        if isinstance(out, (list, tuple)):
            return out[0] if len(out) else None
        return out

    def close(self):
        self.vector_env.close()
