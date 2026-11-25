"""Batched Swimmer (v4) over the offsettable mujoco_warp engine.

Reimplements the v4 observation and reward in batched torch; link morphology
(mid + back segment length/angle) is the design axis, deformed per world by the
offsettable reset kernel. Swimmer has no contacts and no termination — it
propels against fluid drag (density/viscosity), which mujoco_warp simulates.
"""

from __future__ import annotations

import numpy as np
import mujoco
import torch
import warp as wp
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from .parallel_env import ParallelMujocoEnv
from .offsettable_logic import (find_segments, n_slots, slot_bounds,
                                build_seg_arrays, DeformArrays,
                                release_geom_frames, floor_reports, SpawnLift, capture_xy_kernel)
from . import reset_kernel as kern
from .config import SwimmerConfig


class SwimmerParallelEnv(ParallelMujocoEnv):

    def __init__(self, num_envs: int = 4, *, config: SwimmerConfig | None = None,
                 device: str = "cuda", viz_env_id: int = 0, render_mode: str | None = None,
                 use_graph: bool = True, debug: bool = False, seed: int = 0):
        self.cfg = config or SwimmerConfig()
        super().__init__(
            num_envs, device=device, viz_env_id=viz_env_id,
            max_episode_steps=self.cfg.max_episode_steps, debug=debug, seed=seed,
            phys_substeps=self.cfg.frame_skip, render_mode=render_mode,
            use_graph=use_graph, njmax=self.cfg.njmax, nconmax=self.cfg.nconmax)

        self._cache_addresses()
        self._alloc_state()
        self._build_kernel_params()
        self._make_spaces()
        self._build_spawn_lift()

        self.register_step_launch(self._launch_tick)
        self.register_step_launch(self._launch_ctrl)

    # ------------------------------------------------------------------ #
    # Model assembly + setup
    # ------------------------------------------------------------------ #
    def _build_base_spec(self) -> mujoco.MjSpec:
        return mujoco.MjSpec.from_file(self.cfg.xml_path)

    def _cache_addresses(self) -> None:
        m = self.m
        self._segs = find_segments(m, self.cfg.min_link_length)
        for v in self.variants:
            release_geom_frames(v, self._segs)
        self.n_offsets = n_slots(self._segs)
        for line in floor_reports(m, self._segs, self.cfg.max_offset):
            print(f"[swimmer] WARNING: {line}")

    def _alloc_state(self) -> None:
        n, dev = self.num_envs, self.device
        self._alloc_action_buffers(self.act_dim(), self.n_offsets)
        self._offsets_t = torch.zeros((n, self.n_offsets), device=dev)
        self._qpos_init = torch.as_tensor(self.m.qpos0.astype(np.float32), device=dev)
        self._tick = torch.zeros(1, dtype=torch.int32, device=dev)
        self._lane_seed = torch.full((n,), int(self.seed), dtype=torch.int32, device=dev)
        self._xy_prev = torch.zeros((n, 2), device=dev)   # root (Humanoid: COM) x, y at the end of the last step or the respawn
        self._diverged = torch.zeros(n, dtype=torch.bool, device=dev)

    def _build_kernel_params(self) -> None:
        sim = kern.SimArrays()
        for attr in ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl"):
            setattr(sim, attr, getattr(self.gpu_data, attr))
        self._sim = sim

        dfm = DeformArrays()
        for attr in ("geom_pos", "geom_quat", "geom_size", "geom_rbound",
                     "geom_aabb", "body_pos", "body_ipos", "body_iquat",
                     "body_mass", "body_inertia", "site_pos"):
            setattr(dfm, attr, getattr(self.gpu_model, attr))
        self._dfm = dfm

        env = kern.EnvArrays()
        env.mask = wp.from_torch(self.reset_mask)
        env.tick = wp.from_torch(self._tick)
        env.lane_seed = wp.from_torch(self._lane_seed)
        env.qpos_init = wp.from_torch(self._qpos_init)
        env.deform = wp.from_torch(self._embodiment_buf)
        env.offsets = wp.from_torch(self._offsets_t)
        env.ctrl_act = wp.from_torch(self._act_buf)
        env.ctrl_scale = wp.from_torch(self.ctrl_scale)
        env.ctrl_mid = wp.from_torch(self.ctrl_mid)
        self._env = env

        self._seg = build_seg_arrays(self._segs, self.device)

        c = kern.Const()
        c.seed = int(self.seed)
        c.nq, c.nv, c.nu = int(self.m.nq), int(self.m.nv), int(self.m.nu)
        c.n_seg = int(len(self._segs["body"]))
        c.reset_noise = float(self.cfg.reset_noise_scale)
        c.max_offset = float(self.cfg.max_offset)
        self._consts = c

    def _build_spawn_lift(self) -> None:
        """Calibrate the design-aware spawn height on the base design, deformed by
        the same launches the resets use. The RNG tick is put back afterwards."""
        self._lift = SpawnLift(self.m, self.gpu_model, self.gpu_data, self._env.mask, self._env.qpos_init,
                               self.num_envs, self.device)
        tick = self._tick.clone()
        self.reset_mask.fill_(True)
        self._embodiment_buf.zero_()
        with self._wp_scope():
            self._launch_reset()
            self._lift.calibrate()
        self._tick.copy_(tick)
        self.reset_mask.fill_(False)

    def _make_spaces(self) -> None:
        self.single_observation_space = spaces.Box(
            -np.inf, np.inf, shape=(self.obs_dim(),), dtype=np.float32, seed=self.seed)
        self.single_action_space = spaces.Box(
            -1.0, 1.0, shape=(self.act_dim(),), dtype=np.float32, seed=self.seed)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def _setup_camera(self, camera: mujoco.MjvCamera) -> None:
        """Use the stock v4 tracking camera so the swimmer stays in frame."""
        camera_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_CAMERA, "track")
        if camera_id < 0:
            raise RuntimeError("Swimmer model is missing its 'track' camera")
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = camera_id

    # ------------------------------------------------------------------ #
    # Dims / bounds
    # ------------------------------------------------------------------ #
    def act_dim(self) -> int:
        return int(self.m.nu)

    def obs_dim(self) -> int:
        # v4 default: qpos without the (x, y) coordinates, then qvel.
        return int(self.m.nq - 2 + self.m.nv)

    @property
    def offset_bounds(self):
        return slot_bounds(self._segs, self.n_offsets)

    # ------------------------------------------------------------------ #
    # Step-program launches
    # ------------------------------------------------------------------ #
    def _launch_tick(self) -> None:
        wp.launch(kern.tick_kernel, dim=1, inputs=[self._env])

    def _launch_reset(self) -> None:
        """Explicit reset: bump the RNG tick, then respawn the masked worlds."""
        self._launch_tick()
        self._launch_respawn()

    def _launch_respawn(self) -> None:
        wp.launch(kern.reset_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._dfm, self._seg, self._consts])
        self._lift.launch()
        # A respawned world's first step measures its displacement from where it respawned.
        wp.launch(capture_xy_kernel, dim=self.num_envs, inputs=[self._env.mask, self.gpu_data.qpos, 0, 1, wp.from_torch(self._xy_prev)])

    def _launch_ctrl(self) -> None:
        wp.launch(kern.ctrl_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._consts])

    # ------------------------------------------------------------------ #
    # Obs / reward
    # ------------------------------------------------------------------ #
    def _compute_obs(self) -> torch.Tensor:
        qpos, qvel = self.qpos_torch(), self.qvel_torch()
        obs = torch.cat([qpos[:, 2:], qvel], dim=1)
        self._diverged = ~torch.isfinite(obs).all(dim=1)
        return obs.masked_fill(self._diverged[:, None], 0.0)

    def _reward_terminated_truncated(self):
        x_now = self.qpos_torch()[:, 0]
        dt = self.phys_substeps * float(self.m.opt.timestep)
        xy_now = self.qpos_torch()[:, 0:2]
        vel = (xy_now - self._xy_prev) / dt
        forward = vel[:, 0]
        self._xy_prev.copy_(xy_now)

        ctrl_cost = self.cfg.ctrl_cost_weight * (self._act_buf ** 2).sum(dim=1)
        reward = self.cfg.forward_reward_weight * forward - ctrl_cost
        self._step_info = {"reward_fwd": self.cfg.forward_reward_weight * forward, "reward_ctrl": -ctrl_cost,   # v4 info
                           "x_position": xy_now[:, 0].clone(), "y_position": xy_now[:, 1].clone(), "distance_from_origin": xy_now.norm(dim=1),
                           "x_velocity": vel[:, 0], "y_velocity": vel[:, 1], "forward_reward": self.cfg.forward_reward_weight * forward}
        reward = torch.where(self._diverged, torch.zeros_like(reward), reward)

        terminated = self._diverged  # Swimmer-v4 has no termination
        truncated = self.world_steps >= self.cfg.max_episode_steps * self.phys_substeps
        return reward, terminated, truncated
