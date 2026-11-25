"""Codesign-aware vectorized F1TENTH environment (JAX).

``F1TenthVectorEnv`` wraps f1tenth_gym_jax's single-track car and exposes the codesign-gym
API over a batch. It is natively batched with ``jax.vmap`` -- a batch of *different vehicle
designs* has identical state shapes, so heterogeneous designs vmap directly. And because the
design parameters (a ``Param`` pytree) live *outside* the kinematic state, a design can be
swapped mid-episode with **no state projection** -- so this is the suite's first env that
genuinely supports transition-level embodiment changes.

Design & control:
  - Embodiment = the vehicle setup (5 vars: tyre grip, cornering stiffness x2, drive power,
    CG split); mass and the rest are *derived* via the couplings in ``embodiment.py``.
  - Control policy = ``[steering, longitudinal]`` command.
  - Reward = f1tenth's ``progress`` reward (unchanged).
  - Observation = f1tenth's, except that its first entry, the arc length s along the track, is
    given as a fraction of the lap (s / track length, in [0, 1]), so it has the same range on every
    track (f1tenth gives it in metres: 0-63 m on Levine, 0-343 m on Spielberg).

Path A (per the design discussion): f1tenth ``@jit(static_argnums=...)``s its dynamics with
``Param`` as a *static* argument, which blocks threading a per-env (traced) design. We
therefore unwrap that jitted stack (``.__wrapped__``) so a per-env ``Param`` flows through the
integrator; every other f1tenth component (LiDAR scan, frenet, collision, obs, progress
reward) is design-independent and reused unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # allocate VRAM on demand, not JAX's default ~75% pool
import jax
import jax.numpy as jnp
import gymnasium as gym
from gymnasium import spaces

from core.codesign_vec_env import CoDesignVecEnv
from core.seeding import parse_seed

from .config import F1TenthConfig
from . import embodiment as emb_mod

BatchedEmb = Union[None, np.ndarray, Sequence[Optional[np.ndarray]]]

_UNWRAPPED = False
_MAP_DIR_ENV = "F1TENTH_GYM_JAX_MAP_DIR"


@contextmanager
def _packaged_map_dir(map_name: str):
    """Expose a vendored map through f1tenth_gym_jax's supported lookup hook."""
    map_root = Path(__file__).resolve().parent / "assets" / "maps"
    if not (map_root / map_name).is_dir():
        yield
        return
    previous = os.environ.get(_MAP_DIR_ENV)
    os.environ[_MAP_DIR_ENV] = str(map_root)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_MAP_DIR_ENV, None)
        else:
            os.environ[_MAP_DIR_ENV] = previous


def _ensure_unwrapped() -> None:
    """Strip f1tenth's ``@jit(static_argnums=params)`` off the dynamics stack (once) so a
    per-env, traced ``Param`` can flow through. Same math, only an inner jit layer removed."""
    global _UNWRAPPED
    if _UNWRAPPED:
        return
    from f1tenth_gym_jax.envs import dynamic_models as dm
    for n in ("upper_accel_limit", "accl_constraints", "steering_constraint",
              "vehicle_dynamics_st_switching", "vehicle_dynamics_ks", "vehicle_dynamics_st_smooth"):
        f = getattr(dm, n, None)
        if f is not None and hasattr(f, "__wrapped__"):
            setattr(dm, n, f.__wrapped__)
    from f1tenth_gym_jax.envs import f110_env as f110_mod
    from jax_pf import ray_marching
    static_argnums = {"trace_ray": (5, 14), "get_scan": (3, 7, 16)}
    for n in ("xy_2_rc", "distance_transform", "trace_ray", "get_scan"):
        f = getattr(ray_marching, n)
        while hasattr(f, "__wrapped__"):
            f = f.__wrapped__
        setattr(ray_marching, n, jax.jit(f, static_argnums=static_argnums.get(n, ())))
    f110_mod.get_scan = ray_marching.get_scan
    _UNWRAPPED = True


def _bcast(mask: jnp.ndarray, ndim: int) -> jnp.ndarray:
    return mask.reshape((mask.shape[0],) + (1,) * (ndim - 1))


class F1TenthVectorEnv(CoDesignVecEnv):
    """Synchronous JAX-vmapped batch of F1TENTH vehicle co-design environments."""

    _render_human_automatically = True

    # Vehicle params live outside the kinematic state -> transition-level changes are clean.
    metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP, "supports_transition_embodiment": True,
                "supports_partial_reset": True, "render_modes": ["human", "rgb_array", "web"]}

    def __init__(
        self,
        num_envs: int,
        *,
        cfg: F1TenthConfig,
        is_eval_env: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        _ensure_unwrapped()
        from f1tenth_gym_jax.envs.utils import Param
        from f1tenth_gym_jax.envs.f110_env import F110Env

        self.cfg = cfg
        self.render_mode = render_mode

        self._base = Param(
            map_name=cfg.map_name, num_beams=cfg.num_beams, produce_scans=True,
            collision_on=True, reward_type="progress", model="st", max_steps=10000 if is_eval_env else cfg.time_limit,
        )
        with _packaged_map_dir(cfg.map_name):
            self._env = F110Env(num_agents=1, params=self._base)
        # Instance-level unwrap of the captured integrator/model references.
        self._env.integrator_func = self._env.integrator_func.__wrapped__
        self._env.model_func = self._env.model_func.__wrapped__
        self._agent = self._env.agents[0]

        obs_dim = int(self._env.state_size + self._env.scan_size)
        single_obs = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        # The longitudinal bound is the strongest design's, not the base car's: the
        # dynamics clip each car to its own a_max, so a design with more drive power
        # can use it without the policy's action box hiding it.
        a_max = max(float(self._base.a_max), emb_mod.max_accel(cfg))
        single_act = spaces.Box(
            low=np.array([self._base.sv_min, -a_max], dtype=np.float32),
            high=np.array([self._base.sv_max, a_max], dtype=np.float32),
            dtype=np.float32,
        )
        super().__init__(
            num_envs, single_obs, single_act,
            emb_mod.make_embodiment_space(cfg), emb_mod.base_embodiment(cfg),
            is_eval_env=is_eval_env,
        )

        self._reset_fn = jax.jit(jax.vmap(self._make_reset_one()))
        self._advance = jax.jit(self._make_advance())

        if render_mode is not None and render_mode not in ("rgb_array", "human", "web"):
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        # In "web" mode we record the batched state trajectory for f1tenth's WebRenderer.
        self._record = render_mode == "web"
        self._traj: list = []

        self._key = jax.random.PRNGKey(0)
        self._states = None
        self._dvecs = None
        self._designs = None
        self._last_obs = None
        self._hfig = None
        self._viewer = None
        self._trail = None                 # per-env (x, y) breadcrumb for the full-track view
        self._start_s = None               # per-env arc-length at episode start (progress origin)
        self._autoreset = np.zeros(self.num_envs, dtype=bool)
        self._rejected = np.zeros(self.num_envs, dtype=bool)      # lanes whose last reset rejected their design
        self.last_success = np.zeros(self.num_envs, dtype=bool)
        self.last_diverged = np.zeros(self.num_envs, dtype=bool)

    # ------------------------------------------------------------------ #
    # JAX kernels (closures over the underlying F110Env)
    # ------------------------------------------------------------------ #
    def _lap_fraction(self, obs):
        """The agent's observation with s (entry 0, metres) as a fraction of the lap."""
        obs = obs.astype(jnp.float32)
        return obs.at[0].set(obs[0] / jnp.float32(self._env.track.length))

    def _make_reset_one(self):
        env, a0 = self._env, self._agent
        def reset_one(key):
            obs, state = env.reset(key)
            return self._lap_fraction(obs[a0]), state
        return reset_one

    def _cd_step(self, key, state, action, dvec):
        """f1tenth step with a per-env design ``dvec`` threaded into the integrator."""
        env, a0, fields = self._env, self._agent, emb_mod.PARAM_FIELDS
        params = self._base.replace(**{fields[i]: dvec[i] for i in range(len(fields))})
        x = state.cartesian_states
        us = jnp.where(state.collisions[:, None], 0.0, action[None, :])
        x_and_u = jnp.hstack((x, us))
        integ = jax.vmap(env.integrator_func, in_axes=[None, 0, None])
        new = integ(env.model_func, x_and_u, params)
        new = jnp.where(state.collisions[:, None], x_and_u, new)
        state = state.replace(
            last_cartesian_states=state.cartesian_states,
            cartesian_states=new[:, :-2],
            last_frenet_states=state.frenet_states,
            frenet_states=env.track.vmap_cartesian_to_frenet_jax(new[:, [0, 1, 4]]),
        )
        state = jax.lax.cond(env.params.produce_scans, env._scan, env._ret_orig_state, state, key)
        state = state.replace(step=state.step + 1)
        state = jax.lax.cond(env.params.collision_on, env._collisions, env._ret_orig_state, state)
        obs = env.get_obs(state)
        _, state = env.check_done(state)
        rewards = env.get_reward(state)
        return self._lap_fraction(obs[a0]), rewards[a0], state.done[0], state.collisions[0], state

    def _make_advance(self):
        reset_one = self._make_reset_one()
        def advance(states, actions, dvecs, keys, newkeys, auto):
            obs_s, rew_s, done_s, coll_s, states_s = jax.vmap(self._cd_step)(keys, states, actions, dvecs)
            obs_r, states_r = jax.vmap(reset_one)(newkeys)

            def sel(a, b):
                return jax.tree_util.tree_map(lambda x, y: jnp.where(_bcast(auto, x.ndim), x, y), a, b)

            states_out = sel(states_r, states_s)
            obs = jnp.where(_bcast(auto, obs_s.ndim), obs_r, obs_s)
            reward = jnp.where(auto, 0.0, rew_s)
            diverged = ~(jnp.isfinite(obs).all(axis=1) & jnp.isfinite(reward))
            states_out = jax.tree_util.tree_map(
                lambda new, old: jnp.where(_bcast(diverged, new.ndim), old, new), states_out, states)
            obs = jnp.where(diverged[:, None], 0.0, obs)
            reward = jnp.where(diverged, 0.0, reward)
            term = jnp.where(auto, False, coll_s) | diverged      # collision / divergence terminates
            trunc = jnp.where(auto, False, done_s & (~coll_s)) & ~diverged  # lap / max-steps truncates
            return states_out, obs, reward, term, trunc, diverged
        return advance

    # ------------------------------------------------------------------ #
    # Codesign API
    # ------------------------------------------------------------------ #
    def _validate_embodiment(self, embodiment: BatchedEmb) -> np.ndarray:
        return np.array([
            True if e is None else emb_mod.validate_embodiment(e, self.cfg)
            for e in self._split_embodiment(embodiment)
        ], dtype=bool)

    def _split_embodiment(self, embodiment: BatchedEmb) -> List[Optional[np.ndarray]]:
        if embodiment is None:
            return [None] * self.num_envs
        if isinstance(embodiment, np.ndarray):
            if embodiment.ndim == 1:
                return [embodiment] * self.num_envs
            if embodiment.ndim == 2 and embodiment.shape[0] == self.num_envs:
                return [embodiment[i] for i in range(self.num_envs)]
            raise ValueError(f"embodiment array must be (5,) or (N,5), got {embodiment.shape}")
        seq = list(embodiment)
        if len(seq) != self.num_envs:
            raise ValueError(f"embodiment sequence length {len(seq)} != num_envs {self.num_envs}")
        return [None if e is None else np.asarray(e) for e in seq]

    def _prepare(self, embodiment: BatchedEmb):
        """Return (designs (N,5), rejected (N,)); rejected envs fall back to base."""
        if (isinstance(embodiment, (list, tuple)) and len(embodiment) == self.num_envs
                and all(e is not None and np.shape(e) == (5,) for e in embodiment)):
            embodiment = np.stack([np.asarray(e) for e in embodiment])     # one design per lane, as (N, 5)
        if isinstance(embodiment, np.ndarray) and embodiment.shape in ((5,), (self.num_envs, 5)):
            # The per-lane loop below, done once for the whole batch: a Box per lane made each reset take seconds
            # at 16k lanes. Same check as validate_embodiment (finite and within the design bounds).
            space = emb_mod.make_embodiment_space(self.cfg)
            designs = np.broadcast_to(np.asarray(embodiment, dtype=np.float32), (self.num_envs, 5))
            ok = np.isfinite(designs).all(1) & (designs >= space.low).all(1) & (designs <= space.high).all(1)
            return np.where(ok[:, None], designs, self.base_embodiment).astype(np.float32), ~ok
        embs = self._split_embodiment(embodiment)
        rejected = np.zeros(self.num_envs, dtype=bool)
        out = np.empty((self.num_envs, 5), dtype=np.float32)
        for i, e in enumerate(embs):
            if e is None:
                out[i] = self.base_embodiment
            elif emb_mod.validate_embodiment(e, self.cfg):
                out[i] = np.asarray(e, dtype=np.float32)
            else:
                out[i] = self.base_embodiment
                rejected[i] = True
        return out, rejected

    def _next_keys(self) -> jnp.ndarray:
        keys = jax.random.split(self._key, self.num_envs + 1)
        self._key = keys[0]
        return keys[1:]

    # ------------------------------------------------------------------ #
    # reset / step
    # ------------------------------------------------------------------ #
    def _reset(self, *, seed=None, options=None, embodiment: BatchedEmb = None):
        mask = np.asarray((options or {}).get("reset_mask", np.ones(self.num_envs, bool)))
        if mask.shape != (self.num_envs,) or mask.dtype != np.bool_:
            raise ValueError(f"reset_mask must be a bool array of shape ({self.num_envs},)")
        if self._states is None and not mask.all():
            raise RuntimeError("partial reset requires an initial full reset")
        if not mask.any():
            info = {"design": self._designs.copy(), "step_count": np.asarray(self._states.step).reshape(self.num_envs, -1)[:, 0]}
            return self._last_obs.copy(), np.zeros(self.num_envs, bool), info
        # One key for the batch from an integer; one per lane from a sequence.
        parsed, lane_keys = parse_seed(seed, self.num_envs), None
        if isinstance(parsed, int):
            self._key = jax.random.PRNGKey(parsed)
        elif parsed is not None:
            self._key = jax.random.PRNGKey(parsed[0])
            lane_keys = jnp.stack([jax.random.PRNGKey(s) for s in parsed])
        designs, rejected = self._prepare(embodiment)       # None, or a None lane, is the base design
        keys = lane_keys if lane_keys is not None else self._next_keys()
        fresh_obs, fresh_states = self._reset_fn(keys)
        fresh_obs = np.asarray(fresh_obs, np.float32)
        fresh_dvecs = jnp.asarray(emb_mod.design_to_param_vector(designs, self.cfg))
        if mask.all():
            self._states, self._designs, self._dvecs = fresh_states, designs, fresh_dvecs
            self._autoreset[:] = False
            obs = fresh_obs
        else:
            select = jnp.asarray(mask)
            self._states = jax.tree_util.tree_map(lambda new, old: jnp.where(_bcast(select, new.ndim), new, old), fresh_states, self._states)
            self._designs = np.where(mask[:, None], designs, self._designs)
            self._dvecs = jnp.where(select[:, None], fresh_dvecs, self._dvecs)
            self._autoreset[mask] = False
            obs = np.where(mask[:, None], fresh_obs, self._last_obs)
        self._rejected[mask] = rejected[mask]   # a rejected lane runs nothing until a reset accepts its design
        self.last_success[mask] = False
        self.last_diverged[mask] = False
        if self.render_mode is not None:
            pos = np.asarray(self._states.cartesian_states)[:, 0, :2]
            fs = np.asarray(self._states.frenet_states)[:, 0, 0]
            if mask.all():
                self._trail = [[tuple(pos[i])] for i in range(self.num_envs)]
                self._start_s = fs.copy()
            else:
                for i in np.flatnonzero(mask):
                    self._trail[i] = [tuple(pos[i])]
                    self._start_s[i] = fs[i]
        if self._record:
            frame = np.asarray(self._states.cartesian_states)
            self._traj = [frame] if mask.all() else [*self._traj, frame]
        self._last_obs = np.asarray(obs, np.float32)
        info = {"design": self._designs.copy(), "step_count": np.asarray(self._states.step).reshape(self.num_envs, -1)[:, 0]}
        return self._last_obs.copy(), rejected & mask, info

    def _step(self, actions: np.ndarray, embodiment: BatchedEmb = None):
        if self._states is None:
            raise RuntimeError("call reset() before step()")
        if self._rejected.any():
            raise RuntimeError(f"step called before a successful reset of lanes {np.flatnonzero(self._rejected).tolist()} "
                               "(their embodiment was rejected); reset them with a valid design")
        if embodiment is not None:
            # Transition-level design change (base did not freeze it: transitions are
            # supported). Every lane is checked before anything changes: an invalid
            # design raises (README), and a None entry keeps that lane's design.
            embs = self._split_embodiment(embodiment)
            bad = [i for i, e in enumerate(embs) if e is not None and not emb_mod.validate_embodiment(e, self.cfg)]
            if bad:
                raise ValueError(f"invalid step-time embodiment for lanes {bad}")
            designs = self._designs.copy()
            for i, e in enumerate(embs):
                if e is not None:
                    designs[i] = np.asarray(e, dtype=np.float32)
            self._designs = designs
            self._dvecs = jnp.asarray(emb_mod.design_to_param_vector(designs, self.cfg))

        a = jnp.asarray(np.asarray(actions, np.float32)).reshape(self.num_envs, 2)
        auto = self._autoreset.copy()
        states, obs, reward, term, trunc, diverged = self._advance(
            self._states, a, self._dvecs, self._next_keys(), self._next_keys(), jnp.asarray(auto)
        )
        self._states = states
        self._last_obs = np.asarray(obs, np.float32)
        if self.render_mode is not None and self._trail is not None:
            pos = np.asarray(states.cartesian_states)[:, 0, :2]
            fs = np.asarray(states.frenet_states)[:, 0, 0]
            for i in range(self.num_envs):
                if auto[i]:                        # env autoreset this step -> fresh trail
                    self._trail[i] = []            # (avoids a teleport line to the new start)
                    self._start_s[i] = fs[i]       # new episode -> progress restarts at 0
                self._trail[i].append(tuple(pos[i]))
                if len(self._trail[i]) > 120:      # cap: bounds the per-frame trail draw
                    del self._trail[i][:-120]
        if self._record:
            self._traj.append(np.asarray(states.cartesian_states))
        term_np, trunc_np = np.asarray(term), np.asarray(trunc)
        self.last_diverged = np.asarray(diverged).copy()
        self.last_success = (np.asarray(states.num_laps).reshape(self.num_envs, -1)[:, 0] >= self._base.max_num_laps) & trunc_np
        self._autoreset = term_np | trunc_np
        info = {
            "embodiment_rejected": np.zeros(self.num_envs, bool),
            "design": self._designs.copy(),
            "step_count": np.asarray(states.step),
            "success": self.last_success.copy(),
            "diverged": self.last_diverged.copy(),
        }
        return np.asarray(obs, np.float32), np.asarray(reward, np.float32), term_np, trunc_np, info

    # ------------------------------------------------------------------ #
    # Rendering (track + car + co-design overlay; f1tenth's own renderer is a
    # whole-trajectory HTML dashboard, so we draw a per-frame matplotlib view)
    # ------------------------------------------------------------------ #
    def _overlay_lines(self, i: int) -> List[str]:
        d = np.asarray(self._designs)[i]
        v = np.asarray(self._dvecs)[i]
        s = float(np.asarray(self._states.frenet_states)[i, 0, 0])
        lap = int(np.asarray(self._states.num_laps)[i, 0])
        length = float(self._env.track.length)
        s0 = float(self._start_s[i]) if self._start_s is not None else 0.0
        progress = (s - s0) % length         # from the episode start (0 -> length over a lap)
        return [
            f"{self.cfg.name}",
            f"mu0={d[0]:.2f}  Cs={d[1] + d[2]:.1f}  F={d[3]:.0f}N  lr/L={d[4]:.2f}",
            f"m={v[0]:.2f}kg  mu_eff={v[1]:.2f}  a_max={v[5]:.1f}",
            f"progress {progress:.1f}/{length:.0f}  lap {lap}",
        ]

    def _render_frame(self, i: int) -> np.ndarray:
        if self._viewer is None:
            from .viewer import CanonicalViewer
            self._viewer = CanonicalViewer(self._env.track)
        st = np.asarray(self._states.cartesian_states)
        x, y, psi = float(st[i, 0, 0]), float(st[i, 0, 1]), float(st[i, 0, 4])
        trail = np.asarray(self._trail[i]) if self._trail is not None else None
        return self._viewer.frame(x, y, psi, self._overlay_lines(i), trail=trail)

    def save_dashboard(self, output_path: str = "f1tenth_dashboard.html"):
        """Render the recorded rollout to f1tenth's rich self-contained HTML dashboard.

        Requires ``render_mode="web"`` (which records the batched state trajectory across
        ``reset``/``step``). Returns the path to the written HTML file; every design in the
        batch appears as its own rollout in the dashboard.
        """
        if not self._record:
            raise RuntimeError('save_dashboard() requires render_mode="web"')
        if len(self._traj) < 2:
            raise RuntimeError("no trajectory recorded yet; reset and step first")
        from f1tenth_gym_jax.envs.rendering import WebRenderer

        traj = np.stack(self._traj, axis=0)  # (steps, num_envs, num_agents, state_dim)
        renderer = WebRenderer(self._env, trajectory_layout="step_major")
        return renderer.render(traj, output_path=output_path)

    def render(self, env_idx: Optional[int] = None):
        """``rgb_array`` -> tuple of per-env frames (track + car + design overlay), or only
        ``env_idx``'s frame; ``human`` -> live window for ``env_idx`` (default env 0);
        ``web`` -> use :meth:`save_dashboard`. Returns ``None`` before the first reset."""
        if env_idx is not None and not 0 <= int(env_idx) < self.num_envs:
            raise IndexError("env_idx out of range")
        if self.render_mode == "web":
            return None
        if self.render_mode is None or self._states is None:
            return None
        if self.render_mode == "human":
            import matplotlib.pyplot as plt

            frame = self._render_frame(0 if env_idx is None else int(env_idx))
            if self._hfig is None:
                plt.ion()
                self._hfig = plt.figure("F1Tenth")
                self._hax = self._hfig.add_subplot(111)
                self._hax.axis("off")
                self._hfig.tight_layout(pad=0)
                self._himg = self._hax.imshow(frame, animated=True)
                self._hfig.canvas.draw()
                self._hbg = self._hfig.canvas.copy_from_bbox(self._hax.bbox)
                self._hfig.show()
            else:
                # blit: only the image artist is redrawn each frame (fast)
                self._himg.set_data(frame)
                self._hfig.canvas.restore_region(self._hbg)
                self._hax.draw_artist(self._himg)
                self._hfig.canvas.blit(self._hax.bbox)
            self._hfig.canvas.flush_events()
            return None
        if env_idx is not None:
            return self._render_frame(int(env_idx))
        return tuple(self._render_frame(i) for i in range(self.num_envs))

    def close_extras(self, **kwargs):
        if self._hfig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._hfig)
            self._hfig = None
        return None
