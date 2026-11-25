"""Co-design adapter over the native batched ball-catcher env.

Implements ``core.codesign_vec_env.CoDesignVecEnv`` with reset-only designs: reset()
applies a design to the selected worlds without a physics step and returns
(obs, rejected, info). Designs passed to step() are ignored.

- EmbodimentSpace: the embodiment Box, accepting one design or one row per world.
- BallCatcherCoDesignEnv: the adapter, returned by ``registration.build_vec``.
"""

import numpy as np
import mujoco_warp as mjw
import torch
import warp as wp
from gymnasium import spaces

from core.codesign_vec_env import CoDesignVecEnv
from core.embodiment_batch import design_rows, lane_valid
from core.seeding import parse_seed
from core.mujoco_rendering import render_batch
from .ball_catcher_env import BallCatcherParallelEnv


def _tensor(value, device):
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
        return torch.stack([item.to(device=device, dtype=torch.float32) for item in value])
    if isinstance(value, (list, tuple)):
        value = np.asarray(value)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


class EmbodimentSpace(spaces.Box):
    """Normalised embodiment space, accepting one design or one row per native world."""

    def __init__(self, low, high, num_envs):
        super().__init__(low=low, high=high, dtype=np.float32)
        self.num_envs = num_envs

    def contains(self, value):
        try:
            value = _tensor(value, "cpu").numpy()
        except (TypeError, ValueError, RuntimeError):
            return False
        return value.shape in (self.shape, (1, *self.shape), (self.num_envs, *self.shape)) and bool(np.isfinite(value).all() and (value >= self.low).all() and (value <= self.high).all())


class BallCatcherCoDesignEnv(BallCatcherParallelEnv, CoDesignVecEnv):
    """``BallCatcherParallelEnv`` with the ``CoDesignVecEnv`` reset/step contract.

    Adds the embodiment space, the base (all-zero) design, and the copies that
    still need a successful reset, which block step().
    """

    metadata = {**CoDesignVecEnv.metadata, "supports_transition_embodiment": False,
                "supports_partial_reset": True, "render_modes": ["human", "rgb_array"]}
    _render_human_automatically = True
    # ParallelMujocoEnv comes first in the MRO and has its own reset and step
    reset = CoDesignVecEnv.reset
    step = CoDesignVecEnv.step

    def __init__(self, *, is_eval_env=False, **kwargs):
        # CoDesignVecEnv.__init__ is skipped: it would rebuild the spaces
        BallCatcherParallelEnv.__init__(self, **kwargs)
        self.embodiment_space = EmbodimentSpace(*self.offset_bounds, self.num_envs)
        self.is_eval_env = is_eval_env
        self.base_embodiment = np.zeros(self.n_offsets, dtype=np.float32)
        # Copies never reset, or whose last reset was rejected: step refuses to run while any is set
        self._needs_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._any_needs_reset = True

    def _validate_embodiment(self, embodiment):
        """Per world: whether its design is inside embodiment_space. A ``None`` lane is valid."""
        try:
            rows, _ = design_rows(embodiment, self.base_embodiment, self.num_envs)
        except (TypeError, ValueError):
            return np.zeros(self.num_envs, dtype=bool)
        return lane_valid(rows, self.embodiment_space.low, self.embodiment_space.high)

    def _reset(self, *, seed=None, options=None, embodiment=None):
        """Apply designs to the worlds in ``options['reset_mask']`` (default all)
        and respawn them without a physics step.

        Runs the reset kernel and a forward pass, as an autoreset would,
        and initialises the reward's episode state. Worlds that come out
        non-finite are returned as rejected.
        """
        options = {} if options is None else options
        if set(options) - {"reset_mask"}:
            raise ValueError("only reset_mask is supported in reset options")
        mask = torch.as_tensor(options.get("reset_mask", [True] * self.num_envs), device=self.device)
        if mask.dtype != torch.bool or mask.shape != (self.num_envs,):
            raise ValueError("reset_mask must be boolean with shape (num_envs,)")
        if not self.validate_embodiment(embodiment).all():
            raise ValueError(f"embodiment {embodiment} is not in embodiment_space {self.embodiment_space}")
        self._needs_reset |= mask              # until this reset succeeds
        self._any_needs_reset = True
        # None, or a None lane, resets to the base design
        design = torch.as_tensor(design_rows(embodiment, self.base_embodiment, self.num_envs)[0], device=self.device)
        self._ensure_graph()
        pending = self.reset_mask.clone()   # worlds the next step would respawn
        self.reset_mask.copy_(mask)
        self._embodiment_buf[mask] = design[mask]
        self._act_buf[mask] = 0
        self._apply_seed(seed, mask)
        with self._wp_scope():
            self._launch_reset()
            wp.to_torch(self.gpu_data.ctrl)[mask] = 0
        self._refresh_constants()
        with self._wp_scope():
            mjw.forward(self.gpu_model, self.gpu_data)
        self.world_steps[mask] = 0
        self.ep_return[mask] = 0
        obs = self._compute_obs(only_reset=True)   # other worlds' episode state does not change
        self._rebase_episode(mask)
        # Worlds reset here must not respawn again on the next step
        self.reset_mask.copy_(pending & ~mask)
        self._reset_pending = bool(self.reset_mask.any())
        rejected = self._diverged & mask
        self._needs_reset = (self._needs_reset & ~mask) | rejected
        self._any_needs_reset = bool(self._needs_reset.any())
        return obs, rejected, {"embodiment": self._offsets_t.clone()}

    def _apply_seed(self, seed, mask) -> None:
        """Seed the reset worlds. An integer is every reset world's seed (worlds still differ: each draws at its own
        index); a per-lane sequence gives world i its own. The step counter restarts, so a seeded reset replays, and
        streams are keyed by (seed, step) hashed together, so runs seeded s and s + 1 share none."""
        parsed = parse_seed(seed, self.num_envs)
        if parsed is None:
            return
        lanes = torch.full((self.num_envs,), parsed, dtype=torch.int32) if isinstance(parsed, int) else torch.as_tensor(parsed, dtype=torch.int32)
        self._lane_seed[mask] = lanes.to(self.device)[mask]
        self._tick.fill_(-1)                               # tick_kernel adds 1 before the respawn draws

    def _step(self, action, embodiment=None):
        if self._any_needs_reset:
            copies = torch.nonzero(self._needs_reset).flatten().tolist()
            raise RuntimeError(f"call reset() before step(): copies {copies} have no successful reset")
        action = _tensor(action, self.device)
        if action.shape != self._act_buf.shape or not bool(torch.isfinite(action).all()):
            raise ValueError("action must be finite with shape (num_envs, action_dim)")
        return BallCatcherParallelEnv.step(self, action)

    def render(self, env_idx=None):
        return render_batch(self, BallCatcherParallelEnv.render, env_idx)

    def sample_observations(self, n=1):
        """``n`` episode-start observations from full resets with the base design."""
        rows = [self.reset()[0] for _ in range((int(n) + self.num_envs - 1) // self.num_envs)]
        return torch.cat(rows)[:int(n)]
