"""Single-instance Gymnasium adapter for the ball-catcher env.

Wraps a one-world ``BallCatcherParallelEnv`` as an unbatched ``CoDesignEnv``,
for runners that expect a single ``gymnasium.Env``. ``make_ball_catcher`` is
the factory hook for ``ppo_ngopt_rlgo``'s ``environment_factory``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from gymnasium import spaces

from core.codesign_env import CoDesignEnv
from core.mujoco_rendering import show_human

from .ball_catcher_env import BallCatcherParallelEnv


class BallCatcherGymEnv(CoDesignEnv):
    """Unbatched co-design view of a one-world ``BallCatcherParallelEnv``.

    Holds the wrapped env. The spaces are its single-world spaces, and the
    embodiment space is its ``offset_bounds``.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, device: str = "cpu", render_mode: Optional[str] = None,
                 is_eval_env: bool = False, **env_kwargs: Any):
        env_kwargs.pop("num_envs", None)
        self._env = BallCatcherParallelEnv(
            num_envs=1, device=device, render_mode=render_mode, **env_kwargs)
        lo, hi = self._env.offset_bounds
        embodiment_space = spaces.Box(
            low=lo, high=hi, dtype=np.float32, seed=self._env.seed)
        super().__init__(embodiment_space=embodiment_space, is_eval_env=is_eval_env)
        self.observation_space = self._env.single_observation_space
        self.action_space = self._env.single_action_space
        self.render_mode = render_mode

    def _to_batch(self, arr: Any) -> torch.Tensor:
        return torch.as_tensor(np.asarray(arr), dtype=torch.float32, device=self._env.device).reshape(1, -1)

    def _validate_embodiment(self, embodiment: Any) -> bool:
        return True   # the embodiment_space check is all there is

    def _reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None,
               embodiment: Any = None):
        obs = self._env.reset()
        if embodiment is not None:
            # A design only takes effect through a respawn, so applying one costs a
            # zero-control step: mark the world as ended, and the step respawns it
            # with the design (NEXT_STEP autoreset) instead of advancing it
            self._env.reset_mask.fill_(True)
            self._env._reset_pending = True
            zero_ctrl = torch.zeros((1, self._env.act_dim()), dtype=torch.float32, device=self._env.device)
            obs, _, _, _, _ = self._env.step(zero_ctrl, embodiment=self._to_batch(embodiment))
        return obs[0].detach().cpu().numpy(), False, {}

    def _step(self, action, embodiment: Any = None):
        actions = self._to_batch(action)
        emb = None if embodiment is None else self._to_batch(embodiment)
        obs, reward, terminated, truncated, info = self._env.step(actions, embodiment=emb)
        step_info = {
            key: (value[0].detach().cpu().numpy() if torch.is_tensor(value) else value)
            for key, value in info.items()
        }
        return obs[0].detach().cpu().numpy(), float(reward[0].item()), bool(terminated[0].item()), bool(truncated[0].item()), step_info

    def render(self):
        if self.render_mode is None:
            return None
        if self.render_mode == "human":
            show_human(self._env)
            return None
        return self._env.render()

    def close(self):
        self._env.close()


def make_ball_catcher(device: str = "cpu", **kwargs: Any) -> BallCatcherGymEnv:
    """Factory hook for ``ppo_ngopt_rlgo``'s ``environment_factory``."""
    return BallCatcherGymEnv(device=device, **kwargs)
