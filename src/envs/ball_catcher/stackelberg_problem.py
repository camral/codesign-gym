"""StackelbergPPO ``CodesignProblem`` backend for the ball catcher.

StackelbergPPO works on graphs: N nodes with (N, D) designs and (N, S)
observations. The ball catcher is one fixed-topology robot, so it is a single
node whose design is the env's embodiment action and whose observation is the
env's observation. ``make_problem`` is the hook for ``--problem``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch

from .ball_catcher_env import BallCatcherParallelEnv

_NON_ENV_KEYS = {"device"}


class BallCatcherProblem:
    """A one-world ball catcher as a single-node ``CodesignProblem``.

    Holds the wrapped env, the design and its bounds as (1, D) float64 arrays,
    and placeholder graph features.
    """

    design: np.ndarray
    design_low: np.ndarray
    design_high: np.ndarray
    fixed_features: np.ndarray
    edges: np.ndarray
    control_action_dim: int

    def __init__(self, config: Mapping[str, Any]):
        config = dict(config)
        device = config.pop("device", "cpu")
        env_kwargs = {k: v for k, v in config.items() if k not in _NON_ENV_KEYS}

        # StackelbergPPO sets torch's default dtype to float64, but the env's
        # buffers use the default dtype and its kernels need float32
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            self.env = BallCatcherParallelEnv(num_envs=1, device=device, **env_kwargs)
        finally:
            torch.set_default_dtype(previous_dtype)
        self.control_action_dim = self.env.act_dim()

        design_low, design_high = self.env.offset_bounds
        self.design_low = design_low.reshape(1, -1).astype(np.float64)
        self.design_high = design_high.reshape(1, -1).astype(np.float64)
        self.design = np.zeros_like(self.design_low)

        # The (N, F) contract needs at least one fixed feature, and there is none
        self.fixed_features = np.zeros((1, 1), dtype=np.float64)
        self.edges = np.empty((2, 0), dtype=np.int64)

    def reset(self, design: np.ndarray, seed: int | None = None) -> np.ndarray:
        del seed  # the env's own RNG drives episode variation
        embodiment = torch.as_tensor(np.asarray(design)[0], dtype=torch.float32, device=self.env.device).reshape(1, -1)
        zero_ctrl = torch.zeros((1, self.control_action_dim), dtype=torch.float32, device=self.env.device)
        self.env.reset()
        # The zero-control step stages the design for the next respawn. The
        # episode this reset started still runs the one staged before
        obs, _, _, _, _ = self.env.step(zero_ctrl, embodiment=embodiment)
        return obs[0].detach().cpu().numpy().astype(np.float64).reshape(1, -1)

    def step(self, control: np.ndarray):
        ctrl = torch.as_tensor(np.asarray(control)[0], dtype=torch.float32, device=self.env.device).reshape(1, -1)
        obs, reward, terminated, truncated, _ = self.env.step(ctrl)
        observation = obs[0].detach().cpu().numpy().astype(np.float64).reshape(1, -1)
        return observation, float(reward[0].item()), bool(terminated[0].item()), bool(truncated[0].item()), {}


def make_problem(config: Mapping[str, Any]) -> BallCatcherProblem:
    """Factory hook for ``stackelberg_ppo``'s ``--problem`` flag."""
    return BallCatcherProblem(config)
