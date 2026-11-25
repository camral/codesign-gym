"""Codesign-aware vectorized RobotWarehouse environment (JAX).

``RobotWarehouseVectorEnv`` wraps Jumanji's functional RobotWarehouse and exposes
the codesign-gym API over a batch. Unlike the suite's other vector envs (which loop
over per-sub-env instances), this one is natively batched with ``jax.vmap`` -- the
grid is fixed per preset, so a batch of *different* shelf placements has homogeneous
array shapes and vmaps directly. Design and control:

  - Embodiment = shelf positions on the fixed grid (see ``embodiment.py``); installed
    at ``reset`` by rebuilding the batch of states. Shelves are fixed for an episode,
    so ``step(embodiment=...)`` is ignored with a one-time warning.
  - Control policy = central joint action ``MultiDiscrete([5] * num_agents)``; Jumanji's
    per-agent ``action_mask`` is surfaced in ``info``.
  - Reward = Jumanji's global deliveries (unchanged).

Autoreset follows Gymnasium's >=1.0 NEXT_STEP semantics: an env that finished on step
``t`` (collision or horizon) is reset on ``t+1``, reinstalling the same shelf layout.
Observation/action shapes are design-invariant.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # allocate VRAM on demand, not JAX's default ~75% pool
import numpy as np
import jax
import jax.numpy as jnp
import gymnasium as gym
from gymnasium import spaces

from core.codesign_vec_env import CoDesignVecEnv
from core.seeding import parse_seed

from jumanji.environments.routing.robot_warehouse.viewer import RobotWarehouseViewer
from jumanji.environments.routing.robot_warehouse.constants import _SHELVES

from .config import RobotWarehouseConfig
from .generator import CoDesignWarehouseGenerator, CoDesignRobotWarehouse
from . import embodiment as emb_mod

BatchedEmb = Union[None, np.ndarray, Sequence[Optional[np.ndarray]]]


def _bcast(mask: jnp.ndarray, ndim: int) -> jnp.ndarray:
    """Reshape a length-N boolean mask to broadcast over a leaf of rank ``ndim``."""
    return mask.reshape((mask.shape[0],) + (1,) * (ndim - 1))


class RobotWarehouseVectorEnv(CoDesignVecEnv):
    """Synchronous JAX-vmapped batch of RobotWarehouse co-design environments."""

    _render_human_automatically = True

    # Shelves are fixed for an episode -> no transition-level embodiment changes.
    metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP, "supports_transition_embodiment": False,
                "supports_partial_reset": True, "render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_envs: int,
        *,
        cfg: RobotWarehouseConfig,
        is_eval_env: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self.render_mode = render_mode

        self._gen = CoDesignWarehouseGenerator(cfg)
        # Jumanji's default viewer renders in "human" mode (returns no frame); build one in
        # the requested mode so rgb_array actually yields frames.
        viewer = None
        if render_mode is not None:
            viewer = RobotWarehouseViewer(self._gen.grid_size, self._gen.goals, "RobotWarehouse", render_mode=render_mode)
        self.env = CoDesignRobotWarehouse(generator=self._gen, time_limit=cfg.time_limit, viewer=viewer)
        self.num_agents = int(self.env.num_agents)
        self._feat = int(self.env.num_obs_features)
        self._time_limit = int(cfg.time_limit)

        single_action_space = self._build_action_space()
        single_observation_space = self._build_observation_space()
        super().__init__(
            num_envs,
            single_observation_space,
            single_action_space,
            emb_mod.make_embodiment_space(cfg),
            emb_mod.base_embodiment(cfg, self._gen),
            is_eval_env=is_eval_env,
        )

        # vmapped/jitted JAX kernels
        self._vbuild = jax.vmap(self._gen.build_state)
        self._vobs = jax.vmap(lambda s: self.env._make_observations(s.grid, s.agents, s.shelves))
        self._vstep = jax.vmap(self.env.step)
        self._build_batch = jax.jit(self._make_build_batch())
        self._advance = jax.jit(self._make_advance())

        # runtime state
        self._key = jax.random.PRNGKey(0)
        self._states = None
        self._design = None
        self._autoreset = np.zeros(self.num_envs, dtype=bool)
        self._rejected = np.zeros(self.num_envs, dtype=bool)      # lanes whose last reset rejected their design
        self._returns = np.zeros(self.num_envs, dtype=np.float32)  # per-episode deliveries (for render)

    # ------------------------------------------------------------------ #
    # Space construction
    # ------------------------------------------------------------------ #
    def _build_action_space(self) -> spaces.MultiDiscrete:
        return spaces.MultiDiscrete([5] * self.num_agents)

    def _build_observation_space(self) -> spaces.Box:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_agents * self._feat,), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # JAX kernels
    # ------------------------------------------------------------------ #
    def _make_build_batch(self):
        def build_batch(keys, masks):
            states = self._vbuild(keys, masks)
            return states, self._vobs(states), states.action_mask
        return build_batch

    def _make_advance(self):
        def advance(states, actions, masks, newkeys, auto):
            step_states, ts = self._vstep(states, actions)
            last = ts.last()
            trunc = last & (step_states.step_count >= self._time_limit)
            term = last & (~trunc)
            reset_states = self._vbuild(newkeys, masks)
            reset_view = self._vobs(reset_states)

            def sel(a, b):
                return jax.tree_util.tree_map(lambda x, y: jnp.where(_bcast(auto, x.ndim), x, y), a, b)

            out_states = sel(reset_states, step_states)
            view = jnp.where(_bcast(auto, reset_view.ndim), reset_view, ts.observation.agents_view)
            mask = jnp.where(_bcast(auto, out_states.action_mask.ndim),
                             reset_states.action_mask, step_states.action_mask)
            reward = jnp.where(auto, 0.0, ts.reward)
            term_out = jnp.where(auto, False, term)
            trunc_out = jnp.where(auto, False, trunc)
            return out_states, view, reward, term_out, trunc_out, mask
        return advance

    # ------------------------------------------------------------------ #
    # Codesign API helpers
    # ------------------------------------------------------------------ #
    def _validate_embodiment(self, embodiment: BatchedEmb) -> np.ndarray:
        """Validate a batch of designs; returns a boolean array of shape (num_envs,).

        A ``None`` entry (no design supplied for that env) validates as True.
        """
        return np.array([
            True if c is None else emb_mod.validate_embodiment(c, self.cfg)
            for c in self._split_embodiment(embodiment)
        ], dtype=bool)

    def _split_embodiment(self, embodiment: BatchedEmb) -> List[Optional[np.ndarray]]:
        """Normalize the batched embodiment to a length-N list of ``(n_eligible,)`` masks / None."""
        if embodiment is None:
            return [None] * self.num_envs
        if isinstance(embodiment, np.ndarray):
            if embodiment.ndim == 1:  # single occupancy vector -> broadcast
                return [embodiment] * self.num_envs
            if embodiment.ndim == 2:
                if embodiment.shape[0] != self.num_envs:
                    raise ValueError(f"batched embodiment length {embodiment.shape[0]} != num_envs {self.num_envs}")
                return [embodiment[i] for i in range(self.num_envs)]
            raise ValueError(f"embodiment array must be (n_eligible,) or (num_envs, n_eligible), got shape {embodiment.shape}")
        seq = list(embodiment)
        if len(seq) != self.num_envs:
            raise ValueError(f"embodiment sequence length {len(seq)} != num_envs {self.num_envs}")
        return [None if e is None else np.asarray(e) for e in seq]

    def _prepare(self, embodiment: BatchedEmb) -> Tuple[jnp.ndarray, np.ndarray]:
        """Return (masks (N, n_eligible) int32, rejected (N,)). Rejected envs fall back to base."""
        mask_list = self._split_embodiment(embodiment)
        rejected = np.zeros(self.num_envs, dtype=bool)
        base = np.asarray(self.base_embodiment, dtype=np.int32)
        out = np.empty((self.num_envs, base.shape[0]), dtype=np.int32)
        for i, m in enumerate(mask_list):
            if m is None:
                out[i] = base
            elif emb_mod.validate_embodiment(m, self.cfg):
                out[i] = np.asarray(m, dtype=np.int32)
            else:
                out[i] = base
                rejected[i] = True
        return jnp.asarray(out), rejected

    def _next_keys(self) -> jnp.ndarray:
        """Advance the RNG and return a fresh batch of num_envs keys."""
        keys = jax.random.split(self._key, self.num_envs + 1)
        self._key = keys[0]
        return keys[1:]

    def _to_obs(self, view: jnp.ndarray) -> np.ndarray:
        obs = np.asarray(view, dtype=np.float32).reshape(self.num_envs, -1)
        return obs

    # ------------------------------------------------------------------ #
    # reset / step
    # ------------------------------------------------------------------ #
    def _reset(self, *, seed=None, options=None, embodiment: BatchedEmb = None):
        reset_mask = np.asarray((options or {}).get("reset_mask", np.ones(self.num_envs, bool)), dtype=bool)
        if reset_mask.shape != (self.num_envs,):
            raise ValueError("reset_mask must have shape (num_envs,)")
        partial = not reset_mask.all()
        if partial and self._states is None:
            raise RuntimeError("partial reset requires an earlier full reset")
        if not reset_mask.any():
            return self._to_obs(self._vobs(self._states)), np.zeros(self.num_envs, bool), {
                "action_mask": np.asarray(self._states.action_mask),
                "step_count": np.asarray(self._states.step_count),
            }
        # One key for the batch from an integer; one per lane from a sequence.
        parsed, lane_keys = parse_seed(seed, self.num_envs), None
        if isinstance(parsed, int):
            self._key = jax.random.PRNGKey(parsed)
        elif parsed is not None:
            self._key = jax.random.PRNGKey(parsed[0])
            lane_keys = jnp.stack([jax.random.PRNGKey(s) for s in parsed])
        masks, rejected = self._prepare(embodiment)
        keys = lane_keys if lane_keys is not None else self._next_keys()
        states, view, mask = self._build_batch(keys, masks)
        if partial:
            choose = jnp.asarray(reset_mask)
            states = jax.tree_util.tree_map(
                lambda new, old: jnp.where(_bcast(choose, new.ndim), new, old),
                states, self._states,
            )
            masks = jnp.where(_bcast(choose, masks.ndim), masks, self._design)
            view, mask = self._vobs(states), states.action_mask
        self._states, self._design = states, masks
        self._autoreset[reset_mask] = False
        self._returns[reset_mask] = 0.0
        self._rejected[reset_mask] = rejected[reset_mask]   # a rejected lane runs nothing until a reset accepts it

        info = {"action_mask": np.asarray(mask), "step_count": np.asarray(states.step_count)}
        return self._to_obs(view), rejected & reset_mask, info

    def _step(self, actions: np.ndarray, embodiment: BatchedEmb = None):
        if self._states is None:
            raise RuntimeError("call reset() before step()")
        if self._rejected.any():
            raise RuntimeError(f"step called before a successful reset of lanes {np.flatnonzero(self._rejected).tolist()} "
                               "(their embodiment was rejected); reset them with a valid design")
        # Base has already frozen/warned any transition embodiment (metadata: unsupported).
        a = jnp.asarray(np.asarray(actions), dtype=jnp.int32).reshape(self.num_envs, self.num_agents)
        auto = self._autoreset.copy()
        states, view, reward, term, trunc, mask = self._advance(
            self._states, a, self._design, self._next_keys(), jnp.asarray(auto)
        )
        self._states = states
        reward_np = np.asarray(reward, np.float32)
        term_np, trunc_np = np.asarray(term), np.asarray(trunc)
        # per-episode cumulative deliveries: reset the envs that autoreset this step.
        self._returns = np.where(auto, 0.0, self._returns + reward_np)
        self._autoreset = term_np | trunc_np

        info = {
            "embodiment_rejected": np.zeros(self.num_envs, bool),
            "action_mask": np.asarray(mask),
            "step_count": np.asarray(states.step_count),
        }
        return self._to_obs(view), reward_np, term_np, trunc_np, info

    def _state_at(self, i: int):
        return jax.tree_util.tree_map(lambda x: x[i], self._states)

    def _display_state(self, i: int):
        """Host-side viewer state containing only active shelves.

        The viewer loops over shelves in Python. Copy once to host so each
        scalar access does not launch/synchronize a JAX operation on the GPU.
        Shelf IDs are irrelevant to drawing; the simulation keeps its padded state.
        """
        st = jax.device_get(self._state_at(i))
        seen = np.zeros(self._gen.max_shelves + 1, dtype=bool)
        seen[np.asarray(st.grid[_SHELVES]).reshape(-1)] = True
        shelves = jax.tree_util.tree_map(lambda x: x[seen[1:]], st.shelves)
        return type(st)(
            grid=st.grid, agents=st.agents, shelves=shelves, request_queue=st.request_queue,
            step_count=st.step_count, action_mask=st.action_mask, key=st.key,
        )

    def _overlay_lines(self, i: int) -> List[str]:
        """Compact co-design status lines for env ``i``: budget, fleet, progress, return."""
        cfg = self.cfg
        h, w = emb_mod.grid_size(cfg)
        step = int(np.asarray(self._states.step_count)[i])
        return [
            f"{cfg.name}  ·  {cfg.num_shelves} shelves  ·  {self.num_agents} robots  ·  {h}x{w}",
            f"requested {cfg.request_queue_size}/{cfg.num_shelves}  ·  step {step}/{self._time_limit}",
            f"delivered (return): {self._returns[i]:.0f}",
        ]

    def _overlay(self, frame: np.ndarray, i: int) -> np.ndarray:
        """Add the co-design status panel in a white band below the frame (clear of the grid)."""
        try:
            from PIL import Image, ImageDraw, ImageFont
            from matplotlib import font_manager
        except Exception:  # pragma: no cover
            return frame
        base = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
        font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size=max(16, base.height // 55))
        lines = self._overlay_lines(i)
        pad = font.size // 2
        lh = max(font.getbbox(ln)[3] for ln in lines) + pad
        band = lh * len(lines) + 2 * pad

        canvas = Image.new("RGB", (base.width, base.height + band), (255, 255, 255))
        canvas.paste(base, (0, 0))
        draw = ImageDraw.Draw(canvas)
        for k, ln in enumerate(lines):
            draw.text((pad, base.height + pad + k * lh), ln, fill=(20, 20, 20), font=font)
        return np.asarray(canvas)

    def _render_rgb(self, index):
        """Draw a selected world directly on a raster canvas for evaluation video."""
        from PIL import Image, ImageDraw

        state = self._display_state(index)
        h, w = self._gen.grid_size
        scale = 30
        canvas = Image.new("RGB", (w * scale + 1, h * scale + 1), "#f1f3f5")
        draw = ImageDraw.Draw(canvas)
        for row in range(h + 1):
            draw.line((0, row * scale, w * scale, row * scale), fill="#cbd0d5")
        for col in range(w + 1):
            draw.line((col * scale, 0, col * scale, h * scale), fill="#cbd0d5")
        for col, row in np.asarray(self._gen.goals):
            draw.rectangle((int(col) * scale + 1, int(row) * scale + 1,
                            (int(col) + 1) * scale - 1, (int(row) + 1) * scale - 1),
                           fill="#37a772")
        for row, col, requested in zip(state.shelves.position.x, state.shelves.position.y,
                                       state.shelves.is_requested):
            x, y = int(col) * scale, int(row) * scale
            draw.rectangle((x + 5, y + 5, x + scale - 5, y + scale - 5),
                           fill="#e88832" if requested else "#8f9aa6",
                           outline="#374151")
        directions = ((0, -1), (1, 0), (0, 1), (-1, 0))
        for row, col, direction, carrying in zip(
                state.agents.position.x, state.agents.position.y,
                state.agents.direction, state.agents.is_carrying):
            x, y = int(col) * scale + scale // 2, int(row) * scale + scale // 2
            draw.ellipse((x - 9, y - 9, x + 9, y + 9),
                         fill="#8b5cf6" if carrying else "#2563eb", outline="white", width=2)
            dx, dy = directions[int(direction)]
            draw.line((x, y, x + dx * 10, y + dy * 10), fill="white", width=3)
        frame = self._overlay(np.asarray(canvas), index)
        # H.264's default yuv420p format requires even frame dimensions.
        return np.pad(frame, ((0, frame.shape[0] % 2), (0, frame.shape[1] % 2), (0, 0)),
                      mode="edge")

    def render(self, env_idx=None):
        """Render via Jumanji's viewer with a thin co-design overlay. ``human`` shows env 0;
        ``rgb_array`` returns all frames, or only the requested ``env_idx`` frame."""
        if self.render_mode is None or self._states is None:
            return None
        if env_idx is not None and not 0 <= int(env_idx) < self.num_envs:
            raise IndexError("env_idx out of range")
        if self.render_mode == "human":
            index = 0 if env_idx is None else int(env_idx)
            self.env.render(self._display_state(index))
            self._draw_human_overlay(index)
            return None
        if env_idx is not None:
            index = int(env_idx)
            return self._render_rgb(index)
        return tuple(
            self._render_rgb(i)
            for i in range(self.num_envs)
        )

    def _draw_human_overlay(self, i: int) -> None:  # pragma: no cover - needs a live display
        """Blit the co-design panel onto the viewer's matplotlib figure (human mode)."""
        try:
            import matplotlib.pyplot as plt

            name = getattr(getattr(self.env, "_viewer", None), "_name", "RobotWarehouse")
            if not plt.fignum_exists(name):
                return
            fig = plt.figure(name)
            prev = getattr(self, "_htext", None)
            if prev is not None:
                prev.remove()
            self._htext = fig.text(
                0.01, 0.01, "\n".join(self._overlay_lines(i)), ha="left", va="bottom",
                fontsize=11, family="DejaVu Sans",
                bbox=dict(boxstyle="round", fc=(1, 1, 1, 0.7), ec=(0, 0, 0, 0.3)),
            )
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception:
            return

    def close_extras(self, **kwargs):
        try:
            self.env.close()
        except Exception:
            pass
