"""Reset-only co-design API over the batched Swimmer simulator."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import mujoco_warp as mjw
import torch
import warp as wp

from core.codesign_env import CoDesignEnv
from core.embodiment_batch import design_rows, lane_valid
from core.seeding import parse_seed
from .swimmer_env import SwimmerParallelEnv
from .embodiment import EmbodimentSpace, _tensor


class SwimmerCoDesignEnv(CoDesignEnv, SwimmerParallelEnv):
    metadata = {"supports_transition_embodiment": False, "autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP, "render_modes": ["human", "rgb_array"]}

    def __init__(self, *, is_eval_env: bool = False, **kwargs):
        SwimmerParallelEnv.__init__(self, **kwargs)
        self.embodiment_space = EmbodimentSpace(*self.offset_bounds, self.num_envs)
        self.is_eval_env = is_eval_env
        self.base_embodiment = np.zeros(self.n_offsets, dtype=np.float32)
        self._reset_failed = True
        self._const_offsets: torch.Tensor | None = None

    def validate_embodiment(self, embodiment):
        """Per lane: True where the design is valid -- the right shape, finite and
        inside embodiment_space. A ``None`` lane has no design to check and is valid."""
        try:
            rows, _ = design_rows(embodiment, self.base_embodiment, self.num_envs)
        except (TypeError, ValueError):
            return np.zeros(self.num_envs, dtype=bool)
        return lane_valid(rows, self.embodiment_space.low, self.embodiment_space.high)

    def _validate_embodiment(self, embodiment):
        return True  # structure is checked by EmbodimentSpace.contains

    def _reset(self, *, seed=None, options=None, embodiment=None):
        options = {} if options is None else options
        if set(options) - {"reset_mask"}:
            raise ValueError("only reset_mask is supported in reset options")
        mask = torch.as_tensor(options.get("reset_mask", [True] * self.num_envs), device=self.device)
        if mask.dtype != torch.bool or mask.shape != (self.num_envs,):
            raise ValueError("reset_mask must be boolean with shape (num_envs,)")
        if self._reset_failed and not bool(mask.all()):
            raise ValueError("a full reset is required before a partial reset")
        self._reset_failed = True
        if not self.validate_embodiment(embodiment).all():
            raise ValueError(f"embodiment {embodiment} is not in embodiment_space {self.embodiment_space}")
        # None, or a None lane, resets to the base design.
        design = torch.as_tensor(design_rows(embodiment, self.base_embodiment, self.num_envs)[0], device=self.device)
        self._ensure_graph()
        pending = self.reset_mask.clone()
        self.reset_mask.copy_(mask)
        self._embodiment_buf[mask] = design[mask]
        self._act_buf[mask] = 0
        self._apply_seed(seed, mask)
        with self._wp_scope():
            self._launch_reset()
            wp.to_torch(self.gpu_data.ctrl)[mask] = 0
        if self._const_offsets is None or not bool(torch.equal(self._offsets_t, self._const_offsets)):
            self._refresh_constants()
            self._const_offsets = self._offsets_t.clone()
        with self._wp_scope():
            mjw.forward(self.gpu_model, self.gpu_data)
        self.world_steps[mask] = 0
        self.ep_return[mask] = 0
        obs = self._compute_obs()
        self.reset_mask.copy_(pending & ~mask)
        self._reset_pending = bool(self.reset_mask.any())
        rejected = self._diverged & mask
        self._reset_failed = bool(rejected.any())
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
        if self._reset_failed:
            raise RuntimeError("step requires a successful reset")
        action = _tensor(action, self.device)
        if action.shape != self._act_buf.shape or not bool(torch.isfinite(action).all()):
            raise ValueError("action must be finite with shape (num_envs, action_dim)")
        return SwimmerParallelEnv.step(self, action)

    def render(self, env_idx=None):
        from core.mujoco_rendering import render_batch
        return render_batch(self, SwimmerParallelEnv.render, env_idx)

    def close(self, **kwargs):
        # gym.Env.close comes first in the MRO and does nothing; free the renderers.
        return SwimmerParallelEnv.close(self)
