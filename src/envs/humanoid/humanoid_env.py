"""Batched Humanoid (v4) over the offsettable mujoco_warp engine.

The largest port: the 376-dim v4 observation (qpos[2:], qvel, cinert, cvel,
qfrc_actuator, cfrc_ext) and the COM-based forward reward are reproduced from the
batched warp Data. Design axis = 3D length/orientation of the two thighs, two
shins and two upper arms (18-dim). Healthy termination on torso height (fixed per
G2). See config.py for the (approved) PGS->Newton solver note.
"""

from __future__ import annotations

import numpy as np
import mujoco
import mujoco_warp as mjw
import torch
import warp as wp
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from .parallel_env import ParallelMujocoEnv
from .offsettable_logic import (find_segments, n_slots, slot_bounds,
                                build_seg_arrays, DeformArrays,
                                release_geom_frames, floor_reports, SpawnLift, capture_com_xy_kernel)
from . import reset_kernel as kern
from . import radius as rad_mod
from . import arm as arm_mod
from .config import HumanoidConfig


class HumanoidParallelEnv(ParallelMujocoEnv):
    rne_post_constraint = True   # the obs reads cfrc_ext

    def __init__(self, num_envs: int = 4, *, config: HumanoidConfig | None = None,
                 device: str = "cuda", viz_env_id: int = 0, render_mode: str | None = None,
                 use_graph: bool = True, debug: bool = False, seed: int = 0):
        self.cfg = config or HumanoidConfig()
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
        self._segs = find_segments(m, self.cfg.min_link_length, handled_geoms=self.cfg.hand_geoms)   # arm.py deforms the hands
        for v in self.variants:
            release_geom_frames(v, self._segs)
        self._n_capsule = n_slots(self._segs)   # incl. the now-offsettable forearms
        # Design slots after the capsule offsettable slots: foot radii, then hand radii.
        self._rad_tab = rad_mod.find_radius_geoms(m, self.cfg.radius_geoms, self._n_capsule)
        self._arm_tab = arm_mod.find_arms(m, self.cfg.hand_geoms, self._n_capsule + len(self.cfg.radius_geoms))
        self.n_offsets = self._n_capsule + len(self.cfg.radius_geoms) + len(self.cfg.hand_geoms)
        for line in floor_reports(m, self._segs, self.cfg.max_offset):
            print(f"[humanoid] WARNING: {line}")

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
        self._rad = rad_mod.build_rad_arrays(self._rad_tab, self.device)
        self._arm = arm_mod.build_arm_arrays(self._arm_tab, self.device)

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
        """Use the stock v4 tracking camera so the runner stays in frame."""
        camera_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_CAMERA, "track")
        if camera_id < 0:
            raise RuntimeError("Humanoid model is missing its 'track' camera")
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = camera_id

    # ------------------------------------------------------------------ #
    # Dims / bounds
    # ------------------------------------------------------------------ #
    def act_dim(self) -> int:
        return int(self.m.nu)

    def obs_dim(self) -> int:
        # v4: qpos[2:] + qvel + cinert(nbody*10) + cvel(nbody*6) + qfrc_actuator(nv) + cfrc_ext(nbody*6)
        nb = int(self.m.nbody)
        return int(self.m.nq - 2) + int(self.m.nv) + nb * 10 + nb * 6 + int(self.m.nv) + nb * 6

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
        if len(self.cfg.radius_geoms):
            wp.launch(rad_mod.radius_deform, dim=self.num_envs,
                      inputs=[self._env.mask, self._env.deform, self._env.offsets,
                              self._dfm, self._rad, len(self.cfg.radius_geoms),
                              float(self.cfg.radius_frac)])
        if len(self.cfg.hand_geoms):
            # After the capsule deform: track/scale the hands and set the forearms'
            # true 2-geom composite inertia.
            wp.launch(arm_mod.arm_fix, dim=self.num_envs,
                      inputs=[self._env.mask, self._env.deform, self._env.offsets,
                              self._dfm, self._arm, len(self.cfg.hand_geoms),
                              float(self.cfg.radius_frac)])
        self._lift.launch()
        # A respawned world's first step measures its displacement from where it respawned.
        mjw.kinematics(self.gpu_model, self.gpu_data)
        wp.launch(capture_com_xy_kernel, dim=self.num_envs, inputs=[self._env.mask, self.gpu_model.body_mass, self.gpu_data.xipos, wp.from_torch(self._xy_prev)])
        wp.launch(kern.zero_cfrc_kernel, dim=(self.num_envs, int(self.m.nbody)), inputs=[self._env.mask, self.gpu_data.cfrc_ext])

    def _launch_ctrl(self) -> None:
        wp.launch(kern.ctrl_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._consts])

    # ------------------------------------------------------------------ #
    # Obs / reward / healthy termination
    # ------------------------------------------------------------------ #
    def _com_xy(self) -> torch.Tensor:
        # v4 mass_center: sum(body_mass * xipos) / sum(body_mass), from each world's
        # own (design-derived) masses, not the compile-time subtree mass.
        mass = wp.to_torch(self.gpu_model.body_mass).expand(self.num_envs, -1)
        return (mass[:, :, None] * wp.to_torch(self.gpu_data.xipos)[:, :, :2]).sum(1) / mass.sum(1, keepdim=True)

    def _compute_obs(self) -> torch.Tensor:
        n = self.num_envs
        qpos, qvel = self.qpos_torch(), self.qvel_torch()
        cinert = wp.to_torch(self.gpu_data.cinert).reshape(n, -1)
        cvel = wp.to_torch(self.gpu_data.cvel).reshape(n, -1)
        qfrc = wp.to_torch(self.gpu_data.qfrc_actuator).reshape(n, -1)
        cfrc = wp.to_torch(self.gpu_data.cfrc_ext).reshape(n, -1)
        obs = torch.cat([qpos[:, 2:], qvel, cinert, cvel, qfrc, cfrc], dim=1)
        self._diverged = ~torch.isfinite(obs).all(dim=1)
        return torch.nan_to_num(obs).masked_fill(self._diverged[:, None], 0.0)

    def _reward_terminated_truncated(self):
        z = self.qpos_torch()[:, 2]  # torso height
        healthy = ((~self._diverged)
                   & (z > self.cfg.healthy_z_min) & (z < self.cfg.healthy_z_max))
        terminated = ~healthy

        com_now = self._com_xy()
        dt = self.phys_substeps * float(self.m.opt.timestep)
        vel = (com_now - self._xy_prev) / dt
        forward = vel[:, 0]
        self._xy_prev.copy_(com_now)

        # v4 costs the APPLIED ctrl (ctrlrange [-.4, .4]), not the [-1, 1] action.
        applied = wp.to_torch(self.gpu_data.ctrl)
        ctrl_cost = self.cfg.ctrl_cost_weight * (applied ** 2).sum(dim=1)
        # v4 pays healthy_reward as float(is_healthy or terminate_when_unhealthy):
        # these ports always terminate when unhealthy, so every step -- the
        # terminating one included -- earns it.
        reward = (self.cfg.forward_reward_weight * forward
                  + self.cfg.healthy_reward - ctrl_cost)
        self._step_info = {"reward_linvel": self.cfg.forward_reward_weight * forward, "reward_quadctrl": -ctrl_cost,   # v4 info
                           "reward_alive": torch.full_like(forward, self.cfg.healthy_reward),
                           "x_position": com_now[:, 0].clone(), "y_position": com_now[:, 1].clone(), "distance_from_origin": com_now.norm(dim=1),
                           "x_velocity": vel[:, 0], "y_velocity": vel[:, 1], "forward_reward": self.cfg.forward_reward_weight * forward}
        reward = torch.where(self._diverged, torch.zeros_like(reward), reward)

        truncated = self.world_steps >= self.cfg.max_episode_steps * self.phys_substeps
        return reward, terminated, truncated
