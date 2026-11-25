"""
Authors: Aviraj Newatia and Leonard Pleiss

Gymnasium base class for co-design environments.

A co-design environment layers an embodiment (design) axis on top of the
usual Gymnasium contract: alongside the policy action, both ``reset()`` and
``step()`` accept an optional ``embodiment`` that sets or updates the shape of
the agent. ``embodiment_space`` declares its bounds, mirroring
``action_space``.

Set ``is_eval_env=True`` to freeze the shape ``reset()`` establishes: an
``embodiment`` passed to ``step()`` is then ignored, so evaluating one fixed
design cannot be reshaped mid-episode by an exploratory or accidental
embodiment action. A caller that never supplies ``embodiment`` sees the plain
Gymnasium contract.
"""

from __future__ import annotations

from typing import Any
import warnings

import gymnasium as gym
from gymnasium import spaces


def resolve_step_embodiment(env: Any, embodiment: Any) -> Any:
    """Freeze a step-time embodiment for eval / non-transition envs, warning once.

    Shared by :class:`CoDesignEnv` and :class:`core.codesign_vec_env.CoDesignVecEnv`
    so the two messages live in one place. Returns the embodiment to actually apply
    (``None`` when it is frozen out).
    """
    if embodiment is not None and (env.is_eval_env or not env.metadata.get("supports_transition_embodiment", True)):
        warnings.warn("Transition-level embodiment changes are not permitted", UserWarning, stacklevel=3)
        return None
    return embodiment


class CoDesignEnv(gym.Env):

    embodiment_space: spaces.Space

    def __init__(self, embodiment_space: spaces.Space, is_eval_env: bool = False):
        super().__init__()
        self.embodiment_space = embodiment_space
        self.is_eval_env = is_eval_env

    def reset(self, *, seed: int | None = None, options: dict | None = None,
              embodiment: Any = None) -> tuple[Any, Any, dict]:
        result = self._reset(seed=seed, options=options, embodiment=embodiment)
        if getattr(self, "render_mode", None) == "human" and not getattr(self, "_reset_failed", False):
            self.render()
        return result

    def validate_embodiment(self, embodiment: Any) -> Any:
        if not self.embodiment_space.contains(embodiment):
            return False
        return self._validate_embodiment(embodiment=embodiment)

    def step(self, action: Any, embodiment: Any = None):
        embodiment = resolve_step_embodiment(self, embodiment)
        result = self._step(action, embodiment)
        if getattr(self, "render_mode", None) == "human":
            self.render()
        return result

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    def _reset(self, *, seed: int | None, options: dict | None, embodiment: Any) -> tuple[Any, Any, dict]:
        raise NotImplementedError

    def _validate_embodiment(self, embodiment: Any) -> Any:
        raise NotImplementedError

    def _step(self, action: Any, embodiment: Any):
        raise NotImplementedError
