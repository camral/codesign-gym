"""Ball catcher: a legged robot with an arm and glove catches a pitched ball,
with a different body design in every world.

Every world runs the same scene: a legged base MJCF (rocky.xml by default)
carrying an arm MJCF with a glove attached to its arm_tip site,
plus a ball pitched at it. The policy drives the robot's actuators. 
It is rewarded for shortening the relative distance
between the ball and the glove's goal_site, for keeping the ball there, and
for finishing the catch with every foot on the floor.

Code layout:

- ball_catcher_env.py: scene assembly, buffer and kernel-argument setup, spaces,
observation, reward and termination, live tunables, and rendering;
- ball_catcher_kernels.py: the per-step device work: respawn and design
deform, control remapping, and the optional perturbation force on the base;
- offsettable_logic.py -- the link naming convention, the link deform, and the
spawn lift that keeps a deformed robot clear of the floor;
- glove.py -- the glove's box layout;
- parallel_env.py -- the batched base class and its step program.

"""

from __future__ import annotations

import os
import warnings

import mujoco
import numpy as np
import torch
import warp as wp
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from . import ball_catcher_kernels as kern
from .glove import add_glove
from .parallel_env import ParallelMujocoEnv
from .offsettable_logic import (find_segments, length_safe_lo, n_slots, slot_bounds,
                                release_geom_frames, floor_reports,
                                build_seg_arrays, DeformArrays, SpawnLift)


_ASSETS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "assets"))

# Feet the grounded-catch bonus requires on the floor, per base model
_GROUND_GEOMS_DOG = ("fl_knee_geom", "fr_knee_geom", "bl_knee_geom", "br_knee_geom")
_GROUND_GEOMS_ROCKY = ("left_ankle_geom", "right_ankle_geom", "third_ankle_geom")

def _quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    w0, x0, y0, z0 = q.unbind(-1)
    w1, x1, y1, z1 = r.unbind(-1)
    return torch.stack([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1], dim=-1)


class BallCatcherParallelEnv(ParallelMujocoEnv):
    """Batched ball-catcher env with one body design per world.

    Besides the constructor settings, an instance holds:

    - model ids and addresses cached at setup;
    - the design tables of the offsettable links;
    - per-world device buffers the kernels hold warp views of, packed once into
    the kernel argument structs;
    - host-side episode state for the reward, and ``last_*`` outcome flags from
    the latest step.
    """

    _length_floor_warned = False      # see the floor report in _cache_model_addresses

    BASE_BODY = "base"

    def __init__(
        self,
        num_envs: int = 4,
        *,
        # arm_xml is attached to ant_xml's BASE_BODY, the result to scene_xml.
        # The XMLs' timestep is replaced by ``timestep``
        ant_xml: str = "ant_reacher/rocky.xml",
        arm_xml: str = "ant_reacher/rocky_arm.xml",
        scene_xml: str = "_common/scene_ant.xml",
        ground_geoms: tuple[str, ...] = _GROUND_GEOMS_ROCKY,   # must match ant_xml
        # Keyframes combined into the reset pose (missing names are skipped)
        init_key: str | tuple[str, ...] | list[str] = ("init_pos", "init_pos2"),
        ball_size: float = 0.05,
        ball_mass: float = 0.1,
        max_offset: float = 1.0,                  # metres per unit of link embodiment action
        min_link_length: float = 0.05,            # deformed links are clamped to this
        min_length_pad: float = 0.02,             # safety margin above min_link_length for length_safe_lo
        max_return: float = 10.0,
        max_goal_return: float = 9.9,             # return given for the catch (< max_return)
        goal_dwell_steps: int = 10,               # steps at the goal that make a catch
        goal_radius: float = 0.1,
        drop_radius: float = 0.2,                 # leaving it after a catch == drop
        glove_width: float = 0.32,
        glove_height: float = 0.32,
        glove_wall_h: float = 0.24,
        glove_thickness: float = 0.03,
        # Clear only for recording video: the task then has no failure terminal
        terminate_out_of_field: bool = True,
        pitch_from: list[float] = [8, 2, 1],
        pitch_to: list[float] = [0.4648899, 0.2, 3.5],
        pitch_to_radius: float = 0.15,            # target jitter, speed mode only
        pitch_vel: float = 18.0,
        pitch_vel_spread: float = 1.0,
        # Set to switch from speed mode to arrival mode: the ball reaches pitch_to
        # (+/- half widths) after this flight time (+/- spread)
        pitch_arrival_time: float | None = None,
        pitch_arrival_spread: float = 0.0,
        pitch_target_half_widths: tuple[float, float, float] = (0.0, 0.0, 0.0),
        field_pos: list[float] = [0, 0, 0],
        field_size: list[float] = [10, 10, 10],   # half-sizes
        phys_substeps: int = 8,
        max_rel_dist: float = 10.0,               # obs scale of the ball-goal vector
        vel_scale: float = 10.0,                  # obs scale of velocities
        # Random horizontal push on the base, held for force_hold of every
        # force_every control steps
        force_mag: float = 0.0,
        force_every: int = 100,
        force_hold: int = 10,
        force_viz_scale: float = 0.1,
        timestep: float = 0.00625,
        # parallel-runner params
        device: str = "cuda",
        viz_env_id: int = 0,
        max_episode_steps: int = 4000,
        render_mode: str | None = None,
        debug: bool = True,
        use_graph: bool = True,
        check_finite: bool = False,
        njmax: int = 128,
        nconmax: int | None = None,
        seed: int = 1234,
    ):
        self._xmls = (ant_xml, arm_xml, scene_xml)
        self._ground_geom_names = tuple(ground_geoms)
        self._glove = dict(width=glove_width, height=glove_height,
                           wall_h=glove_wall_h, thickness=glove_thickness,
                           goal_radius=goal_radius)
        self.terminate_out_of_field = bool(terminate_out_of_field)
        self.init_key = ((init_key,) if isinstance(init_key, str)
                         else tuple(init_key))
        self.ball_size = float(ball_size)
        self.ball_mass = float(ball_mass)
        self.max_offset = float(max_offset)
        self.min_link_length = float(min_link_length)
        self.min_length_pad = float(min_length_pad)
        self.goal_dwell_steps = int(goal_dwell_steps)
        self.goal_radius = float(goal_radius)
        self.drop_radius = float(drop_radius)
        if self.drop_radius < self.goal_radius:
            raise ValueError("drop_radius must be at least goal_radius")
        # Half the goal budget is given per step over the first goal_dwell_steps
        # at the goal, the other half on a grounded catch. The rest of
        # max_return is given for progress towards the ball
        self.max_goal_return = float(max_goal_return) if self.goal_dwell_steps > 0 else 0.0
        self.at_goal_rew = (0.5 * self.max_goal_return / self.goal_dwell_steps
                            if self.goal_dwell_steps > 0 else 0.0)
        self.grounded_goal_rew = 0.5 * self.max_goal_return
        self.prog_gain = float(max_return) - self.max_goal_return
        self.pitch_from = np.asarray(pitch_from, dtype=np.float32)
        self.pitch_to = np.asarray(pitch_to, dtype=np.float32)
        self.pitch_to_radius = float(pitch_to_radius)
        self.pitch_vel = float(pitch_vel)
        self.pitch_vel_spread = float(pitch_vel_spread)
        self.pitch_arrival_time = 0.0 if pitch_arrival_time is None else float(pitch_arrival_time)   # 0 = speed mode
        self.pitch_arrival_spread = float(pitch_arrival_spread)
        self.pitch_target_half_widths = np.asarray(pitch_target_half_widths, dtype=np.float32)
        if not np.isfinite(self.pitch_arrival_spread) or self.pitch_arrival_spread < 0:
            raise ValueError("pitch_arrival_spread must be finite and nonnegative")
        if pitch_arrival_time is not None and (not np.isfinite(self.pitch_arrival_time) or self.pitch_arrival_time <= self.pitch_arrival_spread):
            raise ValueError("pitch_arrival_time must be finite and exceed pitch_arrival_spread")
        if self.pitch_target_half_widths.shape != (3,) or not np.isfinite(self.pitch_target_half_widths).all() or (self.pitch_target_half_widths < 0).any():
            raise ValueError("pitch_target_half_widths must be three finite nonnegative values")
        if self.pitch_vel_spread < 0:
            raise ValueError("pitch_vel_spread must be nonnegative")
        self.field_pos = np.asarray(field_pos, dtype=np.float32)
        self.field_size = np.asarray(field_size, dtype=np.float32)
        self.max_rel_dist = float(max_rel_dist)
        self.vel_scale = float(vel_scale)
        self.force_mag = float(force_mag)
        self.force_every = int(force_every)
        self.force_hold = int(force_hold)
        self.force_viz_scale = float(force_viz_scale)
        self.force_on = (self.force_every > 0
                              and self.force_hold > 0)
        self.timestep = float(timestep)
        self.check_finite = bool(check_finite)

        super().__init__(
            num_envs, device=device, viz_env_id=viz_env_id,
            max_episode_steps=max_episode_steps, debug=debug,
            seed=seed, phys_substeps=phys_substeps, render_mode=render_mode,
            use_graph=use_graph, njmax=njmax, nconmax=nconmax)

        self._cache_model_addresses()
        self._alloc_device_state()
        self._build_kernel_params()
        self._make_spaces()
        self._build_spawn_lift()

        # Step program (captured as one CUDA graph on GPU):
        self.register_step_launch(self._launch_tick)
        if self.force_on:
            self.register_step_launch(self._launch_force)
        self.register_step_launch(self._launch_ctrl)
        self.register_post_step_launch(self._launch_obs_state_step)

    # ------------------------------------------------------------------ #
    # Model assembly
    # ------------------------------------------------------------------ #
    def _build_base_spec(self) -> mujoco.MjSpec:
        ant_xml, arm_xml, scene_xml = self._xmls
        arm = mujoco.MjSpec.from_file(os.path.join(_ASSETS, arm_xml))
        add_glove(arm, **self._glove)
        arm.compile()   # an attached child's keyframes are namespaced correctly only if compiled first
        ant = mujoco.MjSpec.from_file(os.path.join(_ASSETS, ant_xml))
        ant.attach(arm, prefix="", frame=ant.body(self.BASE_BODY).add_frame())
        ant.compile()

        spec = mujoco.MjSpec.from_file(os.path.join(_ASSETS, scene_xml))
        spec.attach(ant, prefix="", frame=spec.worldbody.add_frame())
        # Added last so its freejoint is the model's last joint - see _cache_model_addresses
        ball = spec.worldbody.add_body(name="ball", pos=self.pitch_from.tolist())
        ball.add_freejoint(name="ball_free")
        ball.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                      size=[self.ball_size, 0, 0], mass=self.ball_mass,
                      contype=1, conaffinity=1, condim=4,
                      friction=[1.0, 0.01, 0.001],
                      rgba=[0.94, 0.26, 0.12, 1.0],
                      solref=[0.015, 1.0], priority=100)
        # Visual-only box marking the bounds the ball and glove must stay inside (group 4: semantic/task)
        spec.worldbody.add_geom(
            name="field", type=mujoco.mjtGeom.mjGEOM_BOX, group=4,
            pos=self.field_pos.tolist(), size=self.field_size.tolist(),
            contype=0, conaffinity=0, density=0.0, mass=0.0,
            rgba=[0.3, 0.6, 1.0, 0.1])
        spec.option.timestep = self.timestep
        return spec

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _cache_model_addresses(self) -> None:
        """Cache model ids and addresses, find the design parts in the model,
        and size the embodiment action."""
        m = self.m
        self._ball_bid = m.body("ball").id
        self._ball_qadr = int(m.joint("ball_free").qposadr[0])
        self._ball_dadr = int(m.joint("ball_free").dofadr[0])
        if self._ball_qadr + 7 != m.nq:
            raise RuntimeError("the ball's freejoint must be the model's last "
                               "joint; obs slices the robot off the front")
        self._nq_robot, self._nv_robot = self._ball_qadr, self._ball_dadr
        self._goal_sid = m.site("goal_site").id
        self._goal_bid = int(m.site_bodyid[self._goal_sid])
        self._goal_sq = torch.as_tensor(m.site_quat[self._goal_sid],
                                        dtype=torch.float32, device=self.device)
        self._root_bid = int(m.body_rootid[self._goal_bid])
        self._base_bid = m.body(self.BASE_BODY).id
        self._floor_geom = m.geom("floor").id
        self._ground_geoms = np.array([m.geom(name).id for name in self._ground_geom_names], dtype=np.int64)
        self._segs = find_segments(m, self.min_link_length)
        self._raw_axis_lo = self._segs["axis_lo"].copy()
        self._segs["axis_lo"] = length_safe_lo(
            self._segs, self._raw_axis_lo, self.max_offset,
            self.min_link_length + self.min_length_pad)
        self.n_links = len(self._segs["body"])
        # A link takes one slot per axis in its name, an _equ group shares one block
        self.n_link_slots = n_slots(self._segs)
        self.n_offsets = self.n_link_slots
        self._init_qpos = self._reset_pose(m)
        # Fields that are identical across variants are uploaded as one shared
        # row, those the reset kernel writes per world must be widened
        if len(self._segs["site_id"]):
            self._make_field_per_world("site_pos")
        hit = floor_reports(m, self._segs, self.max_offset)
        if hit and not BallCatcherParallelEnv._length_floor_warned:
            BallCatcherParallelEnv._length_floor_warned = True    # once per process, the floor can be intended
            warnings.warn(
                f"max_offset={self.max_offset} can shorten "
                f"{len(hit)} link(s) past min_link_length="
                f"{self.min_link_length}, where the deform clamps them:\n  "
                + "\n  ".join(hit), RuntimeWarning, stacklevel=3)

    def _reset_pose(self, m: mujoco.MjModel) -> np.ndarray:
        """qpos0 with every ``init_key`` keyframe applied.

        Attaching fills a keyframe with qpos0 for the joints its model does not
        own, so each keyframe's difference from qpos0 is exactly its own part,
        and the keyframes combine by adding these differences. Two keyframes
        setting the same joint raise.
        """
        keys = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i)
                for i in range(m.nkey)]
        qpos, claimed = np.array(m.qpos0, dtype=np.float64), np.zeros(m.nq, bool)
        for name in self.init_key:
            if name not in keys:
                continue
            delta = m.key_qpos[keys.index(name)] - m.qpos0
            mine = np.abs(delta) > 1e-9
            if (mine & claimed).any():
                raise RuntimeError(
                    f"keyframe {name!r} rewrites qpos "
                    f"{np.flatnonzero(mine & claimed).tolist()}, which an earlier "
                    f"keyframe in {list(self.init_key)} already set")
            qpos, claimed = qpos + delta, claimed | mine
        return qpos.astype(np.float32)

    def _alloc_device_state(self) -> None:
        # The kernel argument structs hold warp views of these buffers, so they
        # must be written in place, never rebound
        n, dev = self.num_envs, self.device
        self._alloc_action_buffers(self.act_dim(), self.n_offsets)
        self._offsets_t = torch.zeros((n, self.n_offsets), device=dev)   # applied design, normalised

        self._force_phase = torch.zeros(n, dtype=torch.int32, device=self.device)
        self._force_cycle = torch.zeros(n, dtype=torch.int32, device=self.device)
        self._force_dir = torch.zeros((n, 2), dtype=torch.float32, device=self.device)

        # Each respawn draws from its lane seed hashed with the tick
        self._tick = torch.zeros(1, dtype=torch.int32, device=dev)
        self._lane_seed = torch.full((n,), int(self.seed), dtype=torch.int32, device=dev)
        self._ep_seed = torch.zeros(n, dtype=torch.int32, device=dev)
        self._tunables = torch.zeros(kern.N_TUNABLES, device=dev)
        self._wp_tunables = wp.from_torch(self._tunables)
        self._qpos_init = torch.as_tensor(self._init_qpos, device=dev)

        # Written in place by obs_state_kernel (inside the CUDA graph)
        self._goal_pos = torch.zeros((n, 3), device=dev)
        self._goal_quat_buf = torch.zeros((n, 4), device=dev)
        self._goal_vel_local = torch.zeros((n, 3), device=dev)
        self._ball_vel_local = torch.zeros((n, 3), device=dev)
        self._rel_local = torch.zeros((n, 3), device=dev)
        self._ball_goal_dist = torch.zeros(n, device=dev)
        self._grounded_buf = torch.zeros(n, dtype=torch.bool, device=dev)

        # Host-side episode state - no kernel views these, so they may be rebound
        self._best_dist = torch.zeros(n, device=dev)   # closest ball-goal distance this episode
        self._d0 = torch.ones(n, device=dev)           # ball-goal distance at respawn
        self._goal_steps = torch.zeros(n, dtype=torch.long, device=dev)
        # Kept on the device so updating them costs no host sync
        self._diverged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._diverged_total = torch.zeros((), dtype=torch.long, device=dev)

        # Obs normalisation constants
        self._fpos = torch.as_tensor(self.field_pos, device=dev)
        self._fsize = torch.as_tensor(self.field_size, device=dev)
        off, scl = self._qpos_norm()
        self._qpos_off = torch.as_tensor(off, device=dev)
        self._qpos_scl = torch.as_tensor(scl, device=dev)


    def _qpos_norm(self) -> tuple[np.ndarray, np.ndarray]:
        """(offset, scale) mapping the robot's qpos into about [-1, 1]: free-joint
        positions by the field box, quaternions unchanged, limited joints by range."""
        m = self.m
        off = np.zeros(self._nq_robot, dtype=np.float32)
        scl = np.ones(self._nq_robot, dtype=np.float32)
        for j in range(m.njnt):
            adr = int(m.jnt_qposadr[j])
            if adr >= self._nq_robot:
                continue
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                off[adr:adr + 3], scl[adr:adr + 3] = self.field_pos, 1.0 / self.field_size
            elif m.jnt_limited[j]:
                scl[adr] = 1.0 / max(np.abs(m.jnt_range[j]).max(), 1e-6)
        return off, scl

    def _build_kernel_params(self) -> None:
        sim = kern.SimArrays()
        for attr in ("qpos", "qvel", "qacc", "qacc_warmstart", "time", "ctrl",
                     "xfrc_applied", "xquat", "cvel", "subtree_com", "site_xpos",
                     "geom_xpos", "geom_xmat"):
            setattr(sim, attr, getattr(self.gpu_data, attr))
        for attr in ("geom_size", "geom_margin"):
            setattr(sim, attr, getattr(self.gpu_model, attr))
        self._sim = sim

        obs_arr = kern.ObsArrays()
        obs_arr.ground_geoms = wp.array(self._ground_geoms.astype(np.int32), dtype=wp.int32, device=self.device)
        obs_arr.goal_pos = wp.from_torch(self._goal_pos, dtype=wp.vec3)
        obs_arr.goal_quat = wp.from_torch(self._goal_quat_buf, dtype=wp.quat)
        obs_arr.goal_vel_local = wp.from_torch(self._goal_vel_local, dtype=wp.vec3)
        obs_arr.ball_vel_local = wp.from_torch(self._ball_vel_local, dtype=wp.vec3)
        obs_arr.rel_local = wp.from_torch(self._rel_local, dtype=wp.vec3)
        obs_arr.dist = wp.from_torch(self._ball_goal_dist)
        obs_arr.grounded = wp.from_torch(self._grounded_buf)
        self._obs_arr = obs_arr

        dfm = DeformArrays()
        for attr in ("geom_pos", "geom_quat", "geom_size", "geom_rbound",
                     "geom_aabb", "body_pos", "body_ipos", "body_iquat",
                     "body_mass", "body_inertia", "site_pos"):
            setattr(dfm, attr, getattr(self.gpu_model, attr))
        self._dfm = dfm

        env = kern.EnvArrays()
        env.mask = wp.from_torch(self.reset_mask)
        env.qpos_init = wp.from_torch(self._qpos_init)
        env.deform = wp.from_torch(self._embodiment_buf)
        env.ctrl_act = wp.from_torch(self._act_buf)
        env.ctrl_scale = wp.from_torch(self.ctrl_scale)
        env.ctrl_mid = wp.from_torch(self.ctrl_mid)
        env.offsets = wp.from_torch(self._offsets_t)
        env.force_phase = wp.from_torch(self._force_phase)
        env.force_cycle = wp.from_torch(self._force_cycle)
        env.force_dir = wp.from_torch(self._force_dir)
        env.tick = wp.from_torch(self._tick)
        env.lane_seed = wp.from_torch(self._lane_seed)
        env.ep_seed = wp.from_torch(self._ep_seed)
        self._env = env

        self._seg = build_seg_arrays(self._segs, self.device)

        c = kern.CatcherConst()
        c.seed = int(self.seed)
        c.nq, c.nv, c.nu = int(self.m.nq), int(self.m.nv), int(self.m.nu)
        c.n_seg = int(self.n_links)
        c.n_slot = int(self.n_link_slots)
        c.ball_qadr, c.ball_dadr = self._ball_qadr, self._ball_dadr
        c.force_every = max(int(self.force_every), 1)
        c.force_hold = int(self.force_hold)
        c.base_bid = self._base_bid
        c.root_bid = self._root_bid
        c.goal_sid = self._goal_sid
        c.goal_bid = self._goal_bid
        c.floor_gid = self._floor_geom
        c.n_ground = int(len(self._ground_geoms))
        sq = self.m.site_quat[self._goal_sid]
        c.goal_sq = wp.quat(float(sq[0]), float(sq[1]), float(sq[2]), float(sq[3]))
        c.gravity = wp.vec3(*self.m.opt.gravity)

        self._consts = c

        self._refresh_tunables()

    def _refresh_tunables(self) -> None:
        # Order must match the T_* indices in ball_catcher_kernels
        self._tunables.copy_(torch.tensor(
            [self.max_offset, self.pitch_vel, self.pitch_to_radius,
             *self.pitch_from, *self.pitch_to, self.force_mag,
             self.pitch_vel_spread, self.pitch_arrival_time, self.pitch_arrival_spread, *self.pitch_target_half_widths],
            dtype=torch.float32))

    def act_dim(self) -> int:
        # Control action only - the embodiment action is a separate step() argument
        return int(self.m.nu)

    def obs_dim(self) -> int:
        # robot qpos + qvel, ball pos/vel, goal pos/quat/vel, ball - goal
        return self._nq_robot + self._nv_robot + 19

    def _qpos_quat_dims(self, nq: int) -> list[int]:
        m, idx = self.m, []
        for j in range(m.njnt):
            a = int(m.jnt_qposadr[j])
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                idx += range(a + 3, a + 7)
            elif m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL:
                idx += range(a, a + 4)
        return [i for i in idx if i < nq]

    @staticmethod
    def _bounds_from_blocks(blocks) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate (size, (lo, hi) or None for unbounded) obs blocks into Box bounds."""
        lo = np.concatenate([np.full(n, -np.inf if b is None else b[0], np.float32) for n, b in blocks])
        hi = np.concatenate([np.full(n, np.inf if b is None else b[1], np.float32) for n, b in blocks])
        return lo, hi

    def _obs_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        # Blocks in _compute_obs order
        lo, hi = self._bounds_from_blocks([(self._nq_robot, None), (self._nv_robot, None), (3, None), (3, None),
                                           (3, None), (4, (-1.0, 1.0)), (3, None), (3, None)])
        q = self._qpos_quat_dims(self._nq_robot)
        lo[q], hi[q] = -1.0, 1.0
        return lo, hi

    def _make_spaces(self) -> None:
        # Only quaternion entries have finite obs bounds: scaled positions and
        # velocities can still leave [-1, 1]. The design is not observed
        low, high = self._obs_bounds()
        assert low.shape == (self.obs_dim(),)
        self.single_observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32, seed=self.seed)
        self.single_action_space = spaces.Box(
            low=np.full(self.act_dim(), -1.0, dtype=np.float32),
            high=np.full(self.act_dim(), 1.0, dtype=np.float32),
            dtype=np.float32, seed=self.seed)
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    # ------------------------------------------------------------------ #
    # Step-program launches (before the physics substeps)
    # ------------------------------------------------------------------ #
    def _launch_tick(self) -> None:
        wp.launch(kern.tick_kernel, dim=1, inputs=[self._env])   # advance the RNG tick

    def _launch_reset(self) -> None:
        """Explicit reset: bump the RNG tick, then respawn the masked worlds."""
        self._launch_tick()
        self._launch_respawn()

    def _launch_respawn(self) -> None:
        wp.launch(kern.reset_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._dfm, self._seg,
                          self._consts,
                          self._wp_tunables])
        self._lift.launch()

    def _build_spawn_lift(self) -> None:
        """Set up the spawn lift, which raises each respawned robot so it starts
        5 mm clear of the floor. Calibrated with one full reset on the all-zero
        design - the RNG tick and reset mask are restored afterwards."""
        self._lift = SpawnLift(self.m, self.gpu_model, self.gpu_data, self._env.mask, self._env.qpos_init,
                               self.num_envs, self.device, margin=0.005, root_body=self._root_bid)
        tick = self._tick.clone()
        self.reset_mask.fill_(True)
        self._embodiment_buf.zero_()
        with self._wp_scope():
            self._launch_reset()
            self._lift.calibrate()
        self._tick.copy_(tick)
        self.reset_mask.fill_(False)

    def _launch_ctrl(self) -> None:
        wp.launch(kern.ctrl_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._consts])

    def _launch_force(self) -> None:
        wp.launch(kern.force_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._consts, self._wp_tunables])

    def _launch_obs_state_step(self) -> None:
        """Run ``obs_state_kernel`` at the tail of the CUDA-graphed step program."""
        wp.launch(kern.obs_state_kernel, dim=self.num_envs,
                  inputs=[self._sim, self._obs_arr, self._consts, self._env.mask, 0])

    def _launch_obs_state(self, only_reset: bool = False) -> None:
        with self._wp_scope():
            wp.launch(kern.obs_state_kernel, dim=self.num_envs,
                      inputs=[self._sim, self._obs_arr, self._consts, self._env.mask, int(only_reset)])

    # ------------------------------------------------------------------ #
    # Obs / reward
    # ------------------------------------------------------------------ #
    def _goal_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """World position and linear velocity of ``goal_site``."""
        site = wp.to_torch(self.gpu_data.site_xpos)[:, self._goal_sid]
        cvel = wp.to_torch(self.gpu_data.cvel)[:, self._goal_bid]
        com = wp.to_torch(self.gpu_data.subtree_com)[:, self._root_bid]
        return site, cvel[:, 3:] + torch.cross(cvel[:, :3], site - com, dim=1)

    def _goal_quat(self) -> torch.Tensor:
        q = _quat_mul(wp.to_torch(self.gpu_data.xquat)[:, self._goal_bid],
                      self._goal_sq)
        return torch.where(q[:, :1] < 0, -q, q)

    def _ball_pos(self) -> torch.Tensor:
        return self.qpos_torch()[:, self._ball_qadr:self._ball_qadr + 3]

    def _all_feet_grounded(self) -> torch.Tensor:
        """Whether every ``ground_geoms`` foot is within contact margin of the
        floor, per world, computed from current kinematics."""
        self._launch_obs_state(only_reset=False)
        return self._grounded_buf

    def _compute_obs(self, only_reset: bool = False, from_step: bool = False) -> torch.Tensor:
        """The observation of every world. When ``from_step`` is True, ``obs_state_kernel``
        already executed inside the CUDA-graphed step program."""
        if not from_step:
            self._launch_obs_state(only_reset=only_reset)
        qpos, qvel = self.qpos_torch(), self.qvel_torch()
        ball_p = self._ball_pos()
        obs = torch.cat([
            (qpos[:, :self._nq_robot] - self._qpos_off) * self._qpos_scl,
            qvel[:, :self._nv_robot] / self.vel_scale,
            (ball_p - self._fpos) / self._fsize,
            self._ball_vel_local / self.vel_scale,
            (self._goal_pos - self._fpos) / self._fsize,
            self._goal_quat_buf,
            self._goal_vel_local / self.vel_scale,
            self._rel_local / self.max_rel_dist,
        ], dim=1)
        if self.check_finite:
            self._assert_finite(obs, "obs")
        # A diverged world never meets a terminal condition (comparisons with NaN
        # are False), so flag it for _reward_terminated_truncated to terminate,
        # and return a zero placeholder row
        diverged = ~torch.isfinite(obs).all(dim=1)
        if only_reset:                 # the other worlds keep their flags and are not counted again
            diverged = torch.where(self.reset_mask, diverged, self._diverged)
            self._diverged_total += (diverged & self.reset_mask).sum()
        else:
            self._diverged_total += diverged.sum()
        self._diverged = diverged
        return obs.masked_fill_(diverged[:, None], 0.0)

    def sample_observations(self, n: int = 1) -> torch.Tensor:
        """``n`` episode-start observations, from resets of the all-zero design.
        Resets the whole batch, so use a separate env."""
        rows, collected = [], 0
        while collected < int(n):
            rows.append(self.reset().clone())
            collected += rows[-1].shape[0]
        return torch.cat(rows)[:int(n)]

    def _rebase_episode(self, mask: torch.Tensor) -> None:
        """Start the progress reward and the goal dwell of the worlds in ``mask`` from their respawn."""
        distance = torch.linalg.norm(self._ball_pos() - self._goal_state()[0], dim=1)
        self._best_dist[mask] = distance[mask]
        self._d0[mask] = distance[mask].clamp(min=1e-3)
        self._goal_steps[mask] = 0

    def _reward_terminated_truncated(self):
        ball = self._ball_pos()
        goal = self._goal_pos
        dist = self._ball_goal_dist

        # Worlds respawned at the start of this step: step() overwrites
        # reset_mask with this step's dones only after this returns
        fresh = self.reset_mask
        self._best_dist = torch.where(fresh, dist, self._best_dist)
        self._d0 = torch.where(fresh, dist.clamp(min=1e-3), self._d0)

        at_goal = dist < self.goal_radius
        retained = dist < self.drop_radius
        self._goal_steps = self._goal_steps.masked_fill(fresh, 0) + at_goal.long()   # total, not consecutive
        dwell_complete = (self._goal_steps >= self.goal_dwell_steps
                          if self.goal_dwell_steps > 0 else torch.zeros_like(at_goal))
        caught = dwell_complete & retained
        dropped = dwell_complete & ~retained
        grounded = self._grounded_buf
        success = caught & grounded

        # Progress is rewarded only when it beats the episode's best distance,
        # and closing the whole starting distance gives prog_gain
        reward = (self.prog_gain / self._d0) * (self._best_dist - dist).clamp(min=0)
        rewarded_at_goal = at_goal & (self._goal_steps <= self.goal_dwell_steps)
        reward = reward + rewarded_at_goal.float() * self.at_goal_rew
        reward = reward + success.float() * self.grounded_goal_rew
        self._best_dist = torch.minimum(self._best_dist, dist)

        left_field = ((ball - self._fpos).abs() > self._fsize).any(dim=1)
        goal_left_field = ((goal - self._fpos).abs() > self._fsize).any(dim=1)

        terminated = success | dropped | self._diverged
        if self.terminate_out_of_field:
            terminated = terminated | left_field | goal_left_field
        truncated = self.world_steps >= self.max_episode_steps
        if self.check_finite:
            self._assert_finite(reward, "reward")
        reward = reward.masked_fill(self._diverged, 0.0)

        self.last_at_goal = at_goal
        self.last_caught = caught
        self.last_dropped = dropped
        self.last_grounded = grounded
        self.last_held_at_goal = success   # read by the baseline evaluators as task success
        self.last_left_field = left_field
        self.last_time_limit = truncated
        self.last_diverged = self._diverged
        return reward, terminated, truncated

    @property
    def diverged_worlds(self) -> int:
        """Worlds ended so far by non-finite physics, should stay 0. Syncs with the device."""
        return int(self._diverged_total)

    # ------------------------------------------------------------------ #
    # Live tunables + host-side views
    # ------------------------------------------------------------------ #
    def change_max_offset(self, new_offset: float | None = None) -> None:
        """Set the metres per unit embodiment action for the links. Does nothing if None"""
        if new_offset is not None:
            self.max_offset = float(new_offset)
            self._segs["axis_lo"] = length_safe_lo(
                self._segs, self._raw_axis_lo, self.max_offset,
                self.min_link_length + self.min_length_pad)
            wp.to_torch(self._seg.axis_lo).copy_(
                torch.as_tensor(self._segs["axis_lo"], dtype=torch.float32, device=self.device))
        self._refresh_tunables()

    def set_pitch(self, vel: float | None = None, to_radius: float | None = None,
                  pitch_from=None, pitch_to=None, vel_spread: float | None = None) -> None:
        """Change the pitch mid-run (e.g. for a curriculum). Does nothing if None"""
        if self.pitch_arrival_time > 0 and any(value is not None for value in (vel, to_radius, vel_spread)):
            raise ValueError("arrival-time pitches use pitch_to and flight time, not speed or target radius")
        if vel is not None:
            self.pitch_vel = float(vel)
        if to_radius is not None:
            self.pitch_to_radius = float(to_radius)
        if pitch_from is not None:
            self.pitch_from = np.asarray(pitch_from, dtype=np.float32)
        if pitch_to is not None:
            self.pitch_to = np.asarray(pitch_to, dtype=np.float32)
        if vel_spread is not None:
            if vel_spread < 0:
                raise ValueError("vel_spread must be nonnegative")
            self.pitch_vel_spread = float(vel_spread)
        self._refresh_tunables()

    def set_force_mag(self, newtons: float) -> None:
        """Set the perturbation force magnitude (0 to disable)."""
        self.force_mag = float(newtons)
        self._refresh_tunables()

    @property
    def env_offsets(self) -> np.ndarray:
        """Host copy of each world's applied design in metres."""
        return self._offsets_t.detach().cpu().numpy() * self.max_offset

    @property
    def env_offsets_norm(self) -> np.ndarray:
        """Host copy of each world's applied design, normalised like the embodiment action."""
        return self._offsets_t.detach().cpu().numpy()

    @property
    def offset_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) of each design entry, narrowed from [-1, 1] where the deform would clip."""
        return slot_bounds(self._segs, self.n_offsets)

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _init_render_model(self, m: mujoco.MjModel) -> None:
        release_geom_frames(m, self._segs)

    def _mirror_render_model(self, m: mujoco.MjModel, env_idx: int) -> None:
        """Copy one world's deformed model values into the CPU render ``MjModel`` ``m``."""
        w = int(env_idx)
        v, gm = m, self.gpu_model
        g, b, kid = self._segs["geom"], self._segs["body"], self._segs["child_id"]
        v.geom_pos[g] = wp.to_torch(gm.geom_pos)[w, g].cpu().numpy()
        v.geom_quat[g] = wp.to_torch(gm.geom_quat)[w, g].cpu().numpy()
        v.body_iquat[b] = wp.to_torch(gm.body_iquat)[w, b].cpu().numpy()
        v.body_ipos[b] = wp.to_torch(gm.body_ipos)[w, b].cpu().numpy()
        v.geom_size[g] = wp.to_torch(gm.geom_size)[w, g].cpu().numpy()
        v.geom_rbound[g] = wp.to_torch(gm.geom_rbound)[w, g].cpu().numpy()
        v.geom_aabb[g] = wp.to_torch(gm.geom_aabb)[w, g].reshape(len(g), 6).cpu().numpy()
        v.body_pos[kid] = wp.to_torch(gm.body_pos)[w, kid].cpu().numpy()
        sid = self._segs["site_id"]
        if len(sid):
            v.site_pos[sid] = wp.to_torch(gm.site_pos)[w, sid].cpu().numpy()
        for bid, gid in zip(self._segs["body"], self._segs["geom"]):
            adr = int(v.body_bvhadr[bid])
            if adr >= 0 and v.body_bvhnum[bid] == 1 and v.body_geomnum[bid] == 1:
                v.bvh_aabb[adr] = v.geom_aabb[gid]

    def _add_debug_visualizations(self, scene: mujoco.MjvScene, env_idx: int) -> None:
        """Draw the force on the base when the perturbation is on."""
        if not self.force_on or scene.ngeom >= scene.maxgeom:
            return
        f = wp.to_torch(self.gpu_data.xfrc_applied)[env_idx, self._base_bid, :2]
        f = f.detach().cpu().numpy().astype(np.float64)
        mag = float(np.linalg.norm(f))
        if mag < 1e-9:
            return

        base = self.xpos_torch()[env_idx, self._base_bid].detach().cpu().numpy()
        base = base.astype(np.float64)
        tip = base + np.array([f[0], f[1], 0.0]) / mag * (mag * self.force_viz_scale)

        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
            np.eye(3).flatten(), np.array([1.0, 0.25, 0.1, 0.9], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.1, base, tip)
        scene.ngeom += 1

    def _setup_camera(self, camera: mujoco.MjvCamera) -> None:
        """Frame Rocky and the incoming pitch arc cleanly."""
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [2.2, 0.4, 1.6]
        camera.elevation = -16.0
        camera.azimuth = 125.0
        camera.distance = 9.5
