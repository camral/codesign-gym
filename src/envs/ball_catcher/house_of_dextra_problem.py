"""House of Dextra ``CodesignProblem`` / ``ControllerBatch`` backend for the
ball catcher.

House of Dextra trains one design-conditioned controller per design family,
then searches that family with the controller frozen. The ball catcher has a
single family. A design is the env's embodiment action, encoded as a single-node
graph. ``make_problem`` is the hook for ``--problem``.

- BallCatcherControllerBatch: a population of designs, one per world of one
``BallCatcherParallelEnv``;
- BallCatcherHouseOfDextraProblem: the design-space interface (sampling,
mutation, encoding, controller batches).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .ball_catcher_env import BallCatcherParallelEnv

_FAMILY = "ball_catcher"
_NON_ENV_KEYS = {"device"}


class BallCatcherControllerBatch:
    """Runs a population of designs, one per world of one batched env.

    Holds the env, the stacked designs and a zero control action.
    """

    def __init__(self, designs: Sequence[np.ndarray], env_kwargs: Mapping[str, Any],
                 device: str, seed: int):
        self.env = BallCatcherParallelEnv(
            num_envs=len(designs), device=device, seed=seed, **env_kwargs)
        self.observation_size = self.env.obs_dim()
        self.action_size = self.env.act_dim()
        self._designs = torch.as_tensor(
            np.stack([np.asarray(d, dtype=np.float32) for d in designs]),
            dtype=torch.float32, device=self.env.device,
        )
        self._zero_ctrl = torch.zeros(
            (len(designs), self.action_size), dtype=torch.float32, device=self.env.device)

    def reset(self) -> torch.Tensor:
        self.env.reset()
        # The zero-control step stages the designs in the env's buffer, which
        # every later respawn reads. The episode this reset started still runs
        # the designs staged before
        obs, _, _, _, _ = self.env.step(self._zero_ctrl, embodiment=self._designs)
        return obs

    def step(self, actions: Any):
        actions_t = torch.as_tensor(
            actions, dtype=torch.float32, device=self.env.device
        ).reshape(len(self._designs), self.action_size)
        obs, reward, terminated, truncated, _ = self.env.step(actions_t)
        return obs, reward, (terminated | truncated), {}

    def close(self) -> None:
        self.env.close()


class BallCatcherHouseOfDextraProblem:
    """Design-space interface for House of Dextra.

    Holds the env kwargs, and the design bounds and sizes read from a probe env.
    """

    def __init__(self, config: Mapping[str, Any]):
        config = dict(config)
        self.device = str(config.pop("device", "cpu"))
        self.env_kwargs = {k: v for k, v in config.items() if k not in _NON_ENV_KEYS}

        probe = BallCatcherParallelEnv(num_envs=1, device=self.device, **self.env_kwargs)
        design_low, design_high = probe.offset_bounds
        self._n_offsets = probe.n_offsets
        self._design_low = design_low.astype(np.float32)
        self._design_high = design_high.astype(np.float32)
        self._action_size = probe.act_dim()
        probe.close()

    def families(self) -> Sequence[str]:
        return [_FAMILY]

    def sample_design(self, family: str, rng: np.random.Generator) -> np.ndarray:
        self._check_family(family)
        return rng.uniform(self._design_low, self._design_high).astype(np.float32)

    def initial_design(self, family: str, rng: np.random.Generator) -> np.ndarray:
        del rng
        self._check_family(family)
        return np.zeros(self._n_offsets, dtype=np.float32)

    def mutate(self, design: Any, family: str, rng: np.random.Generator) -> np.ndarray:
        self._check_family(family)
        mutated = np.asarray(design, dtype=np.float32).copy()
        dim = int(rng.integers(self._n_offsets))
        mutated[dim] = rng.uniform(self._design_low[dim], self._design_high[dim])
        return mutated

    def family(self, design: Any) -> str:
        del design
        return _FAMILY

    def encode(self, design: Any):
        node_features = np.asarray(design, dtype=np.float32).reshape(1, -1)
        edges = np.empty((2, 0), dtype=np.int64)
        return node_features, edges

    def morphology(self, designs: Sequence[Any]) -> np.ndarray:
        return np.stack([np.asarray(d, dtype=np.float32) for d in designs])

    def action_mask(self, designs: Sequence[Any]) -> np.ndarray:
        return np.ones((len(designs), self._action_size), dtype=np.float32)

    def make_controller_batch(
        self, designs: Sequence[Any], *, training: bool, seed: int
    ) -> BallCatcherControllerBatch:
        del training
        return BallCatcherControllerBatch(designs, self.env_kwargs, self.device, seed)

    def serialize(self, design: Any) -> Any:
        return np.asarray(design, dtype=np.float32).tolist()

    def deserialize(self, payload: Any) -> np.ndarray:
        return np.asarray(payload, dtype=np.float32)

    @staticmethod
    def _check_family(family: str) -> None:
        if family != _FAMILY:
            raise ValueError(f"unknown family {family!r}")


def make_problem(config: Mapping[str, Any]) -> BallCatcherHouseOfDextraProblem:
    """Factory hook for house_of_dextra's ``--problem`` flag."""
    return BallCatcherHouseOfDextraProblem(config)
