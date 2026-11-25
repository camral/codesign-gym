"""Vectorized SoftWalker env on the native Warp backend, under the co-design contract.

Each lane is one Warp ``NativeEnv``. CUDA lanes execute on independent streams
through the engine's batch executor; CPU lanes execute serially. This class
keeps lane ownership so it can add ``reset_mask`` partial resets. It provides:

  - the flat per-cell embodiment (``embodiment.py``), rebuilt into a native bundle at every reset;
  - per-env ``validate_embodiment`` (structural checks plus building the bundle, so it agrees with reset) and
    ``(obs, rejected, info)`` resets, with a rejected lane staying invalid until an
    explicit valid reset (stepping it raises);
  - next-step autoreset with the lane's last accepted design and zero reward: the engine restores the settled state
    the last explicit reset accepted (``NativeEnv._autoreset``), so no settling runs again;
  - ``info`` stacked per key like the other codesign-gym vector envs (irregular values stay per-lane lists);
  - ``info["out_of_domain"]``: an MPM step that failed with the body at the grid's edge (``near_grid_edge``), told
    apart from a numerical blow-up, which the engine reports the same way;
  - the design fixed per episode (``supports_transition_embodiment: False``).
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
from gymnasium import spaces
from gymnasium.error import ResetNeeded

from core.codesign_vec_env import CoDesignVecEnv
from core.seeding import lane_seeds

from . import config as C
from .contracts.batching import bounded_actions, render_lanes, split_embodiment, stack_info as _stack
from . import embodiment as E


def default_device() -> str:
    import warp as wp
    return "cuda" if wp.get_cuda_device_count() > 0 else "cpu"


def beam_action_mask(fields: dict, channels: int, action_size: int) -> np.ndarray:
    """Channels a Beam3D design actuates: routed on a link between two occupied (massive) nodes. The engine's mask
    (``env._mask``) counts every link, so a channel used only by empty cells -- whose links have no stiffness and do
    nothing -- would read as live; recomputed here to leave the engine untouched."""
    live = (np.asarray(fields["mass"])[np.asarray(fields["entities"])] > 0).all(axis=1)
    used = np.any(np.abs(np.asarray(fields["routing"])[live]) > 0, axis=0) if live.any() else np.zeros(channels, bool)
    return np.pad(used, (0, action_size - channels))


def near_grid_edge(x: np.ndarray, live: np.ndarray, origin, spacing: float, shape) -> bool:
    """True when a live MPM particle is within one grid cell of the region the engine's ``support_valid`` accepts,
    ``origin + h/2 <= x < origin + (shape - 1.5) h`` per axis. The engine reports leaving it through the same failure
    as a blow-up; on a failed step this is how the wrapper tells a body that walked off the grid from one that diverged."""
    x = np.asarray(x)[np.asarray(live, bool)]
    if not len(x):
        return False
    lo = np.asarray(origin) + .5 * spacing
    hi = np.asarray(origin) + (np.asarray(shape) - 1.5) * spacing
    return bool(np.any(x < lo + spacing) or np.any(x >= hi - spacing))


class SoftWalkerVectorEnv(CoDesignVecEnv):

    _render_human_automatically = True

    metadata = {**CoDesignVecEnv.metadata, "supports_transition_embodiment": False, "supports_partial_reset": True,
                "render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, num_envs: int, *, driver: str, is_eval_env: bool = False, dtype: str = C.DEFAULT_DTYPE, device: Optional[str] = None,
                 terrain: str = "flat", render_mode: Optional[str] = None, execution: str = "auto",
                 physics_steps_per_action: Optional[int] = None):
        # Gymnasium may close a partly constructed object after validation fails.
        self.lanes = []
        self._executor = None
        self.closed = False
        C.validate_options(num_envs, driver, render_mode, execution)
        self.render_mode = render_mode
        default_timing, timing = C.resolve_timing(driver, physics_steps_per_action)
        from .engine.warp_native.env import NativeEnv
        from .engine.warp_native.batching import EpisodeBatchExecutor
        import warp as wp
        self.driver, self.dtype, self.terrain = driver, np.dtype(dtype), terrain
        self.device = default_device() if device is None else device
        is_cuda = wp.get_device(self.device).is_cuda
        if execution == "parallel":
            if not is_cuda:
                raise ValueError("execution='parallel' requires a CUDA device")
        self.execution = execution
        self._executor = EpisodeBatchExecutor(execution="serial" if execution == "auto" and not is_cuda else execution)
        self.dim, self.grid, self.channels, base, template = C.task_layout(driver, self.dtype, terrain)
        durations = C.timing_metadata(timing, default_timing, float(template.config.dt))
        (self.physics_steps_per_action, self.physics_dt, self.control_dt, self.horizon, self.target_episode_duration_s,
         self.episode_duration_s, self.settling_duration_s) = durations
        self.metadata = dict(type(self).metadata, **durations._asdict())
        for _ in range(num_envs):
            self.lanes.append(NativeEnv(C.IDS[driver], bundle=template, dtype=self.dtype.name, device=self.device,
                                       is_eval_env=is_eval_env, **timing))
        super().__init__(num_envs, single_observation_space=self._box(self.lanes[0].observation_space), single_action_space=self._box(self.lanes[0].action_space),
                         embodiment_space=E.CellEmbodimentSpace(self.grid, self.channels, self.dtype),
                         base_embodiment=base, is_eval_env=is_eval_env)
        self._accepted = [template.fields for _ in range(num_envs)]  # last accepted resolved design per lane (autoreset)
        self._designs = [base.copy() for _ in range(num_envs)]       # ... and its flat per-cell vector (render colours)
        self._returns = np.zeros(num_envs)                           # episode return so far (render overlay)
        self.name = C.preset_id(driver, terrain)
        self._pending = np.zeros(num_envs, bool)                     # lane finished: autoreset on its next step
        self._renderers = {}                                         # lane -> (accepted fields, Newton renderer or None)
        self._was_reset = False                                      # step() before the first reset() raises

    def _box(self, space) -> spaces.Box:
        return spaces.Box(np.asarray(space.low, self.dtype), np.asarray(space.high, self.dtype), dtype=self.dtype)

    # ------------------------------------------------------------------ #
    # Embodiment handling
    # ------------------------------------------------------------------ #
    def _split_embodiment(self, embodiment: Any) -> List[Optional[np.ndarray]]:
        """Normalize to a length-N list of flat vectors / None (same forms as the other codesign-gym vector envs)."""
        return split_embodiment(embodiment, self.num_envs)

    def _errors(self, vector) -> list[str]:
        return E.structural_errors(vector, self.grid, self.channels)

    def _resolve(self, vector) -> tuple[Optional[dict], list[str]]:
        """Flat vector -> resolved Warp embodiment dict, or the structural errors."""
        bundle, errors = C.candidate_bundle(vector, driver=self.driver, dim=self.dim, dtype=self.dtype, terrain=self.terrain,
                                            errors=self._errors(vector))
        return (None if bundle is None else bundle.fields), errors

    def _validate_embodiment(self, embodiment: Any) -> np.ndarray:
        """Structural checks plus the Warp builder's own rules, so validation and reset agree."""
        return np.array([True if c is None else not self._resolve(c)[1] for c in self._split_embodiment(embodiment)], dtype=bool)

    # ------------------------------------------------------------------ #
    # reset / step
    # ------------------------------------------------------------------ #
    def _reset(self, *, seed=None, options: dict | None = None, embodiment: Any = None):
        if self.closed:
            raise RuntimeError("Environment is closed")
        options = options or {}
        if set(options) - {"reset_mask"}:
            raise ValueError("Only the reset_mask option is supported")
        mask = np.asarray(options.get("reset_mask", np.ones(self.num_envs, bool)))
        if mask.shape != (self.num_envs,) or mask.dtype != np.bool_:
            raise ValueError("reset_mask must be a boolean array of shape (num_envs,)")
        self._was_reset = True
        seeds = lane_seeds(seed, self.num_envs)          # an int spawns independent lane seeds; a sequence is per lane
        candidates = self._split_embodiment(embodiment)
        results, resetting = [None] * self.num_envs, {}
        for i, lane in enumerate(self.lanes):
            if not mask[i]:
                if lane.owner is not None and (lane.initialized or self._pending[i]):  # live, or ended and autoresetting on its next step
                    results[i] = (lane._observation(lane.owner.initial), False, {**lane._info(), "valid": True})
                else:
                    results[i] = (np.zeros(lane.observation_size, self.dtype), True, {"valid": False, "embodiment_errors": ["LANE_UNINITIALIZED"]})
                continue
            self._pending[i] = False
            fields, errors = (self._accepted[i], []) if candidates[i] is None else self._resolve(candidates[i])
            if errors:
                lane.initialized, lane.owner = False, None  # a rejected design never falls back to an older one
                results[i] = (np.zeros(lane.observation_size, self.dtype), True, {"valid": False, "embodiment_errors": errors})
                continue
            resetting[i] = fields
        # Every lane's settling runs as one batch (concurrent streams on CUDA).
        for i, (obs, rejected, info) in zip(resetting, self._reset_lanes(resetting, [seeds[i] for i in resetting])):
            self._returns[i] = 0.
            if not bool(rejected[0]):
                self._accepted[i] = resetting[i]
                if candidates[i] is not None:
                    self._designs[i] = np.asarray(candidates[i], self.dtype).copy()
                info["valid"] = True
                obs = self._own_mask(i, obs, info)
            results[i] = (obs, bool(rejected[0]), info)
        obs = np.stack([r[0] for r in results]).astype(self.dtype)
        return obs, np.asarray([r[1] for r in results], bool), _stack([r[2] for r in results])

    def _step(self, actions: Any, embodiment: Any = None):
        # The base has already frozen (and warned about) any step-time embodiment.
        if self.closed:
            raise RuntimeError("Environment is closed")
        if not self._was_reset:
            raise RuntimeError("call reset() before step()")
        actions = bounded_actions(actions, self.action_space.shape)   # the whole batch, before any lane advances or autoresets
        if any(lane.closed or (not self._pending[i] and (not lane.initialized or lane._steps >= lane.horizon))
               for i, lane in enumerate(self.lanes)):
            raise ResetNeeded("Every lane requires a valid reset (or a pending autoreset) before stepping")
        results = [None] * self.num_envs
        active = [i for i in range(self.num_envs) if not self._pending[i]]
        for i in np.flatnonzero(self._pending):
            obs, _, info = self.lanes[i]._autoreset()
            self._pending[i] = False
            self._returns[i] = 0.
            info.update(valid=True, autoreset=True, out_of_domain=False)
            obs = self._own_mask(i, obs, info)
            results[i] = (obs, 0., False, False, info)
        if active:
            from .engine.warp_native.vector_env import step_lanes
            advanced = step_lanes([self.lanes[i] for i in active], actions[active], executor=self._executor)
            for i, result in zip(active, advanced):
                obs, reward, terminated, truncated, info = result
                self._returns[i] += reward
                self._pending[i] = terminated or truncated
                info["autoreset"] = False
                info["out_of_domain"] = bool(terminated) and not info.get("valid", True) and self._off_grid(i)
                results[i] = (obs, reward, terminated, truncated, info)
        return (np.stack([r[0] for r in results]).astype(self.dtype), np.asarray([r[1] for r in results], self.dtype),
                np.asarray([r[2] for r in results], bool), np.asarray([r[3] for r in results], bool), _stack([r[4] for r in results]))

    def render(self, env_idx: Optional[int] = None):
        """RGB frames for all lanes, or a live Newton viewer for the selected lane."""
        if self.render_mode is None:
            return None
        if env_idx is not None and not 0 <= int(env_idx) < self.num_envs:
            raise IndexError("env_idx out of range")
        if self.render_mode == "human":
            self._frame(0 if env_idx is None else int(env_idx))
            return None
        return render_lanes(self.num_envs, None if env_idx is None else int(env_idx), self._frame)

    def _frame(self, i: int):
        """Newton 3D frame (``render3d``), or the Pillow side view (``render``) when no OpenGL context is available."""
        from .render import draw
        from .render3d import make_renderer
        lane = self.lanes[i]
        if not lane.initialized:
            return None
        state, model = lane.owner.initial, lane.owner.model
        positions = state.x.numpy() if self.driver.startswith("mpm") else model.reference.numpy() + state.u.numpy()
        travel = lane._info()["metrics"]["displacement"]
        label = f"{self.name}   step {lane._steps}/{lane.horizon}   travel {travel * 100:+.2f} cm   return {self._returns[i]:+.4f}"
        fields, renderer = self._renderers.get(i, (None, None))
        if fields is not self._accepted[i]:  # new design: its topology changes, so rebuild the studio
            if renderer is not None:
                renderer.close()
            renderer = make_renderer(self.driver, self._accepted[i], self._designs[i], lane.template.terrain,
                                     human=self.render_mode == "human")
            self._renderers[i] = (self._accepted[i], renderer)
        if renderer is not None:
            return renderer.draw(positions, label, capture=self.render_mode != "human")
        return draw(self.driver, positions, self._accepted[i], self._designs[i], lane.template.terrain, label=label)

    def _off_grid(self, i: int) -> bool:
        """MPM lane whose last state has a live particle at the grid's edge (``near_grid_edge``); solids have no grid."""
        if not self.driver.startswith("mpm"):
            return False
        lane = self.lanes[i]
        cfg = lane._episode.step.config
        return near_grid_edge(lane.owner.initial.x.numpy(), lane._mass > 0, cfg.origin, cfg.spacing, cfg.grid_shape)

    def _own_mask(self, i: int, obs: np.ndarray, info: dict) -> np.ndarray:
        """Beam3D: swap in the wrapper's action mask (``beam_action_mask``) for the one the lane was reset with, in the
        lane (commands and later observations read it), the reset ``info`` and the mask block closing the raw obs."""
        lane = self.lanes[i]
        if self.driver != "beam":
            return obs
        lane.action_mask = beam_action_mask(self._accepted[i], lane.channels, lane.action_size)
        info["action_mask"] = lane.action_mask.copy()
        obs = np.array(obs, copy=True)
        obs[lane.raw_observation_size - lane.action_size:lane.raw_observation_size] = lane.action_mask
        return obs

    def _reset_lanes(self, designs: dict, seeds: list) -> list:
        """``lane.reset`` for each ``{lane index: resolved design}``, settling them as one batch."""
        if not designs:
            return []
        from .engine.warp_native.vector_env import reset_lanes
        return reset_lanes([self.lanes[i] for i in designs], list(designs.values()), seeds, executor=self._executor)

    def close(self, **kwargs):
        for _, renderer in getattr(self, "_renderers", {}).values():
            if renderer is not None:
                renderer.close()
        self._renderers = {}
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.close()
        for lane in getattr(self, "lanes", ()):
            lane.close()
        self.closed = True
