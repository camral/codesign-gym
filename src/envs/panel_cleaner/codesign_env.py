"""Co-design API over the native batched panel-cleaner env.

- PanelCleanerCoDesignEnv: ``PanelCleanParallelEnv`` under the ``CoDesignVecEnv`` contract,
returned by ``registration.build_vec``.
"""

from __future__ import annotations

import mujoco_warp as mjw
import numpy as np
import torch
from gymnasium import spaces

from core.codesign_vec_env import CoDesignVecEnv
from core.embodiment_batch import design_rows, lane_valid
from core.seeding import parse_seed
from core.mujoco_rendering import render_batch

from .panel_cleaner_env import PanelCleanParallelEnv
from .parallel_env import copy_into


class PanelCleanerCoDesignEnv(PanelCleanParallelEnv, CoDesignVecEnv):
    """``PanelCleanParallelEnv`` under the ``CoDesignVecEnv`` contract.

    The embodiment is fixed per episode: reset() applies it immediately, and
    autoresets reapply each world's current one. Designs passed to step() are
    ignored. Holds the embodiment space, the base embodiment and each world's
    current embodiment.
    """

    metadata = {**CoDesignVecEnv.metadata, "supports_transition_embodiment": False,
                "supports_partial_reset": True, "render_modes": ["human", "rgb_array"]}
    _render_human_automatically = True
    # ParallelMujocoEnv comes first in the MRO and has its own reset and step
    reset = CoDesignVecEnv.reset
    step = CoDesignVecEnv.step

    def render(self, env_idx=None):
        return render_batch(self, PanelCleanParallelEnv.render, env_idx)

    def __init__(self, num_envs: int, *, is_eval_env: bool = False, **cfg) -> None:
        # CoDesignVecEnv.__init__ is skipped: it would rebuild the spaces
        PanelCleanParallelEnv.__init__(self, num_envs, **cfg)
        lo, hi = self.offset_bounds
        self.embodiment_space = spaces.Box(low=lo.astype(np.float32),
                                           high=hi.astype(np.float32),
                                           dtype=np.float32)
        base = np.zeros(self.n_offsets, dtype=np.float32)
        if self.disable_legs:
            base[self.n_link_slots:] = 1.0      # the base embodiment keeps its legs
        self.base_embodiment = base
        self.is_eval_env = bool(is_eval_env)
        self._current_embodiment = self._as_batched_embodiment(None)
        # Copies never reset, or whose last reset was rejected: step refuses to run while any is set
        self._needs_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._any_needs_reset = True

    # ------------------------------------------------------------------ #
    # Co-design contract
    # ------------------------------------------------------------------ #
    def _as_batched_embodiment(self, embodiment) -> torch.Tensor:
        # None, or a None lane, means the base embodiment
        return torch.as_tensor(design_rows(embodiment, self.base_embodiment, self.num_envs)[0], device=self.device)

    def _validate_embodiment(self, embodiment):
        try:
            rows, _ = design_rows(embodiment, self.base_embodiment, self.num_envs)
        except (TypeError, ValueError):
            return np.zeros(self.num_envs, dtype=bool)
        return lane_valid(rows, *self.offset_bounds, tol=1e-6)

    def _reset(self, *, seed=None, options=None, embodiment=None):
        if not self._validate_embodiment(embodiment if embodiment is not None else self.base_embodiment).all():
            raise ValueError("embodiment is invalid: wrong shape, non-finite, or outside embodiment_space (per-slot bounds from the link suffixes)")
        emb = self._as_batched_embodiment(embodiment)
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if options is not None and "reset_mask" in options:
            mask = torch.as_tensor(options["reset_mask"], dtype=torch.bool,
                                   device=self.device)
            if mask.shape != (self.num_envs,):
                raise ValueError("reset_mask must have shape (num_envs,)")
        self._needs_reset |= mask              # until this reset succeeds
        self._any_needs_reset = True
        self._current_embodiment = torch.where(
            mask[:, None], emb, self._current_embodiment)
        # Stage the embodiment and respawn the masked worlds now, on new terrain,
        # so the returned obs is where the first action starts
        self._ensure_graph()
        self._apply_seed(seed, mask)
        pending = self.reset_mask.clone()        # worlds the next step would respawn
        self.reset_mask.copy_(mask)
        copy_into(self._reset_act_buf, self._current_embodiment)
        with self._wp_scope():
            self._launch_reset()
            if self.terrain:
                self._launch_terrain()
        self._refresh_constants()
        with self._wp_scope():
            mjw.forward(self.gpu_model, self.gpu_data)
        self.world_steps.masked_fill_(mask, 0)
        self.ep_return.masked_fill_(mask, 0.0)
        obs = self._compute_obs(only_reset=True)   # other worlds' episode state does not change
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
        if self.terrain:
            self._ep_count[mask] = -1                      # the terrain launch adds 1: the seeded run's first episode

    def _step(self, actions, embodiment=None):
        if self._any_needs_reset:
            copies = torch.nonzero(self._needs_reset).flatten().tolist()
            raise RuntimeError(f"call reset() before step(): copies {copies} have no successful reset")
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        if actions.shape != self._act_buf.shape or not bool(torch.isfinite(actions).all()):
            raise ValueError("action must be finite with shape (num_envs, action_dim)")
        # Stage the current embodiment, so worlds autoresetting this step rebuild it
        obs, reward, terminated, truncated, info = PanelCleanParallelEnv.step(
            self, actions, reset_actions=self._current_embodiment)
        # The obs was built before the reward advanced worlds that finished a goal,
        # so show them their new goal (guard avoids unnecessary clone/kernel on ~99% of steps)
        fresh = self.last_cleaned & ~(terminated | truncated)
        if bool(fresh.any()):
            self._launch_goal(only_reset=True, mask=fresh)
            updated = self._with_current_goal(obs)
            obs = torch.where(fresh[:, None], updated, obs)
        return obs, reward, terminated, truncated, info

    def _with_current_goal(self, rows: torch.Tensor) -> torch.Tensor:
        """``rows`` with the goal block of the obs recomputed for the current goal."""
        g = int(self.m.nq + self.m.nv) + 3 + 4 + 3      # offset of the goal block
        rows = rows.clone()
        rows[:, g:g + 3] = (self._goal_pt - self._fpos) / self._fsize
        rows[:, g + 3:g + 6] = self._goal_normal
        rows[:, g + 6:g + 9] = self._roller_rel_local / self.max_rel_dist
        if self.time_in_obs:
            rows[:, g + 9:g + 10] = self._goal_time_left()
        return rows

