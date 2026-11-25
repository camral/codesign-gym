"""Co-design API over RotateInHand's native batched simulator, with transition-level
embodiment changes.

The embodiment is the native deform vector: every heightfield pad cell in
``[0, 1]``, then every link-offset slot in its own clip range (see
``RotateInHandParallelEnv.offset_bounds``). ``reset()`` writes it and respawns
the selected worlds without a physics step, as the other MuJoCo Warp envs do.
``step(action, embodiment)`` replaces the design of every world from that
step on: both native ``deform_*_during_rollout_s`` windows are held open for
the whole episode, so the deform kernel rewrites each world's shape from its
current design on every control step (exact, since it rebuilds from the
undeformed model), and a world that autoresets respawns with its latest design.
An eval env drops step-time designs with a warning (``core.codesign_env``).

The design is not observed: the native ``can_deform`` flags and the trailing
offsets in force are cut from the observation.
"""

from __future__ import annotations

import math

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from core.codesign_vec_env import CoDesignVecEnv
from core.embodiment_batch import design_rows, lane_valid
from core.seeding import parse_seed
from core.mujoco_rendering import render_batch
from .rotate_in_hand_env import RotateInHandParallelEnv

_WINDOWS = ("deform_hfield_during_rollout_s", "deform_links_during_rollout_s")


def _tensor(value, device):
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
        return torch.stack([item.to(device=device, dtype=torch.float32) for item in value])
    if isinstance(value, (list, tuple)):
        value = np.asarray(value)
    return torch.as_tensor(value, device=device, dtype=torch.float32)


class EmbodimentSpace(spaces.Box):
    """Deform vector bounds, accepting one design or one row per native world."""

    def __init__(self, low, high, num_envs):
        super().__init__(low=low, high=high, dtype=np.float32)
        self.num_envs = num_envs

    def __repr__(self):
        low = np.array2string(self.low, threshold=12, edgeitems=3, max_line_width=120)
        high = np.array2string(self.high, threshold=12, edgeitems=3, max_line_width=120)
        return f"EmbodimentSpace(low={low}, high={high}, shape={self.shape}, dtype={self.dtype})"

    def contains(self, value):
        try:
            value = _tensor(value, "cpu").numpy()
        except (TypeError, ValueError, RuntimeError):
            return False
        return value.shape in (self.shape, (1, *self.shape), (self.num_envs, *self.shape)) and bool(
            np.isfinite(value).all() and (value >= self.low).all() and (value <= self.high).all())


class RotateInHandCoDesignEnv(RotateInHandParallelEnv, CoDesignVecEnv):
    metadata = {**CoDesignVecEnv.metadata, "supports_transition_embodiment": True,
                "supports_partial_reset": True, "render_modes": ["human", "rgb_array"]}
    _render_human_automatically = True
    # ParallelMujocoEnv comes first in the MRO and has its own reset and step
    reset = CoDesignVecEnv.reset
    step = CoDesignVecEnv.step

    def __init__(self, *, is_eval_env=False, **kwargs):
        windows = {name: float(kwargs.get(name, 0.0)) for name in _WINDOWS}
        if any(windows.values()):
            raise ValueError(f"the co-design API holds both deform windows open and changes the "
                             f"shape through step(action, embodiment), so they must be 0, got {windows}")
        kwargs.update({name: math.inf for name in _WINDOWS})
        # CoDesignVecEnv.__init__ is skipped: it would rebuild the spaces
        RotateInHandParallelEnv.__init__(self, **kwargs)
        if not self.n_offsets:
            raise ValueError("hfield_offset_fraction and max_link_offset are both 0, "
                             "so the hand has no embodiment to design")
        self.embodiment_space = EmbodimentSpace(*self.offset_bounds, self.num_envs)
        self.is_eval_env = is_eval_env
        # Undeformed: every pad at its base plane, every link at its MJCF length.
        self.base_embodiment = np.zeros(self.n_offsets, dtype=np.float32)
        self.last_scored = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_diverged = torch.zeros_like(self.last_scored)
        # Copies never reset, or whose last reset was rejected: step refuses to run while any is set
        self._needs_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._any_needs_reset = True

    @property
    def last_success(self):
        """Worlds that met a goal this step; an episode with one is a success."""
        return self.last_scored

    def obs_dim(self):
        # The native layout less the can_deform flags and the offsets in force.
        return RotateInHandParallelEnv.obs_dim(self) - len(self._can_deform_t) - self.n_offsets

    def _make_spaces(self):
        # True bounds, not the native [-1, 1] hull: hand qpos (soft joint limits), hand qvel and the
        # object's position and velocities are unbounded; the object and goal quats lie in [-1, 1];
        # the angle to the goal (over pi) and the trailing contact flags lie in [0, 1].
        low, high = np.full(self.obs_dim(), -np.inf, dtype=np.float32), np.full(self.obs_dim(), np.inf, dtype=np.float32)
        obj = len(self._hand_qadr) + len(self._hand_dadr)
        for a, b in ((obj + 3, obj + 7), (obj + 13, obj + 17)):
            low[a:b], high[a:b] = -1.0, 1.0
        low[obj + 17:], high[obj + 17:] = 0.0, 1.0
        self.single_observation_space = spaces.Box(low=low, high=high, dtype=np.float32, seed=self.seed)
        self.single_action_space = spaces.Box(low=np.full(self.act_dim(), -1.0, dtype=np.float32),
                                              high=np.full(self.act_dim(), 1.0, dtype=np.float32), dtype=np.float32, seed=self.seed)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def _compute_obs(self, only_reset=False, from_step=False):
        obs = RotateInHandParallelEnv._compute_obs(self, only_reset, from_step=from_step)
        head = len(self._hand_qadr) + len(self._hand_dadr) + 18
        return torch.cat([obs[:, :head], obs[:, head + len(self._can_deform_t):obs.shape[1] - self.n_offsets]], dim=1)

    def _validate_embodiment(self, embodiment):
        """Per lane: True where the design is valid -- the right shape, finite and
        inside embodiment_space. A ``None`` lane has no design to check and is valid."""
        try:
            rows, _ = design_rows(embodiment, self.base_embodiment, self.num_envs)
        except (TypeError, ValueError):
            return np.zeros(self.num_envs, dtype=bool)
        return lane_valid(rows, self.embodiment_space.low, self.embodiment_space.high)

    def _reset(self, *, seed=None, options=None, embodiment=None):
        """Apply a design without physics; options['reset_mask'] selects worlds."""
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
        # None, or a None lane, resets to the base design.
        design = torch.as_tensor(design_rows(embodiment, self.base_embodiment, self.num_envs)[0], device=self.device)
        self._ensure_graph()
        pending = self.reset_mask.clone()
        self.reset_mask.copy_(mask)
        self._reset_act_buf[mask] = design[mask]
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
        # Only the reset worlds: the others' goal, dwell and contact history must not tick.
        obs = self._compute_obs(only_reset=True)
        self._rebase_episode(mask)
        # Cleared worlds must not respawn again on the next physical step.
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
        if embodiment is None:
            return RotateInHandParallelEnv.step(self, action)
        if not self.validate_embodiment(embodiment).all():
            raise ValueError(f"embodiment {embodiment} is not in embodiment_space {self.embodiment_space}")
        # The new design is the deform action from this step on, for every world it
        # names, and the design a world that autoresets respawns with; a None lane
        # keeps the design it has.
        rows, given = design_rows(embodiment, self.base_embodiment, self.num_envs)
        given = torch.as_tensor(given, device=self.device)
        design = torch.where(given[:, None], torch.as_tensor(rows, device=self.device), self._reset_act_buf)
        return RotateInHandParallelEnv.step(self, action, design)

    def render(self, env_idx=None):
        return render_batch(self, RotateInHandParallelEnv.render, env_idx)

    def sample_observations(self, n=1):
        """Episode-start observations. This *resets the batch*: use a throwaway env."""
        rows = [self.reset()[0] for _ in range(math.ceil(int(n) / self.num_envs))]
        return torch.cat(rows)[:int(n)]
