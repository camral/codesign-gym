"""NeroGrasp: a Nero arm with a LEAP hand picks a small object off the floor, with designed index and thumb tips."""

from __future__ import annotations

import os
import random
import xml.etree.ElementTree as ET

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from .base_env import EPISODE_STEPS, NeroBaseEnv
from .mesh_switch_env import MeshSwitchMujocoEnv
from .config import (
    FINGERTIP_KEYS,
    FINGERTIP_MESH_ASSETS,
    FINGERTIP_PRESETS,
    MOUNTING_VERTEX_COUNT,
    SHAPE_MAX_BBOX_LIMITS_MM,
    SHAPE_MIN_BBOX_LIMITS_MM,
    NeroGraspConfig,
    resolve_worker_count,
)
from .mujoco_primitive_shapes_utils import (
    OBJECT_BODY_NAME,
    OBJECT_CYL_PREFIX,
    OBJECT_FREEJOINT_NAME,
    OBJECT_GROUPS,
    SPAWN_FLOOR_CLEARANCE_M,
    WASHER_DEFAULT_HALF_HEIGHT,
    WASHER_DEFAULT_RADIUS,
    WASHER_MAX_HALF_HEIGHT_DELTA,
    WASHER_MAX_RANDOMIZATION_DELTA,
    load_object_templates,
    object_body_element,
    object_keypoints,
    place_object,
    resolve_object_names,
)
from .shape_deformation.adapter import load_fingertips
from .traj_utils import (
    ASSETS_DIR,
    LEAP_HAND_INACTIVE_FINGERS_REST,
    NERO_DOF,
    ROBOT_XML,
    leap_hand_start_pose,
    make_arm_demo_qpos_traj,
    make_hand_demo_qpos_traj,
)

STRIPPED_INERTIAL_BODIES = ("leap_if_ds", "leap_mf_ds", "leap_rf_ds", "leap_th_ds")
OBSERVED_FINGERTIPS = ("leap_th_tip", "leap_if_tip")


class NeroGraspEnv(NeroBaseEnv):
    """``num_envs`` NeroGrasp worlds on one device, sharing one episode clock."""

    TYPES_TO_OBJS = OBJECT_GROUPS
    TERMINAL_SUCCESS_BONUS = 1.0
    MIN_ACTIVE_GEOM_SIZE = 1e-4

    def __init__(self, *, config: NeroGraspConfig, num_envs: int, is_eval_env: bool = False, render_mode: str | None = None):
        self.render_mode = render_mode
        self._reset_failed = True
        self._episode_over = False
        self._needs_rebuild = False
        self.config = config
        self.obj_names = resolve_object_names(config.object_name)
        if num_envs % len(self.obj_names) != 0:
            raise ValueError(f"num_envs={num_envs} must be divisible by the number of unique objects ({len(self.obj_names)})")
        self.templates = load_object_templates(ASSETS_DIR / "tasks" / "objects", self.obj_names)
        workers = resolve_worker_count(config.shape_parallel_workers)
        MeshSwitchMujocoEnv.__init__(
            self,
            num_envs,
            device=config.device or ("cpu" if config.use_cpu else "cuda"),
            debug=config.debug,
            verbose=config.verbose,
            compile_workers=workers,
            retain_all_render_models=render_mode is not None,
        )
        self.obj_names_env = [self.obj_names[i % len(self.obj_names)] for i in range(num_envs)]
        self.one_hot_id = config.include_object_one_hot
        self.asymmetric_rl = config.asymmetric_observations
        self._rng = random.Random(config.seed)
        self._lane_rngs: list[random.Random] | None = None

        preset = FINGERTIP_PRESETS[config.fingertip_preset]
        robot_root = self._robot_root(preset.mesh_file)
        self.fingertips = load_fingertips(
            robot_root, str(ASSETS_DIR), preset.deformable_tip_y_percentile,
            mounting_vertex_count=MOUNTING_VERTEX_COUNT,
        )
        self.max_object_boxes = max(len(t.box_geoms) for t in self.templates.values())
        self.max_object_cylinders = max(len(t.cylinder_geoms) for t in self.templates.values())
        self._init_fingertip_pipeline(
            fingertips=self.fingertips,
            mounting_vertex_count=MOUNTING_VERTEX_COUNT,
            max_bbox_limits_mm=SHAPE_MAX_BBOX_LIMITS_MM,
            scene_root=self._scene_root(robot_root),
            workers=workers,
        )

        self.arm_traj = make_arm_demo_qpos_traj(EPISODE_STEPS)
        self.max_episode_steps = self.arm_traj.shape[0]
        self.demo = config.trajectory == "demo"
        self.hand_start_pose = leap_hand_start_pose()
        self.hand_traj = (
            make_hand_demo_qpos_traj(self.max_episode_steps)
            if self.demo
            else np.tile(self.hand_start_pose, (self.max_episode_steps, 1)).astype(np.float32)
        )
        self._gpu_arm_traj = torch.tensor(self.arm_traj, dtype=torch.float32, device=self.device)
        self._gpu_hand_traj = torch.as_tensor(self.hand_traj, dtype=torch.float32, device=self.device)
        self._gpu_inactive_fingers = torch.tensor(LEAP_HAND_INACTIVE_FINGERS_REST, dtype=torch.float32, device=self.device)

        has_noise = config.object_position_randomization > 0.0 or config.object_rotation_randomization_deg > 0.0
        self._init_noise_multiplier(has_noise, per_object=True)

        canonical = {tip.name: self.postprocessor.postprocess(tip, tip.vertices, tip.faces) for tip in self.fingertips}
        self.build_model([canonical] * num_envs)
        self._init_codesign(
            num_envs=num_envs,
            is_eval_env=is_eval_env,
            mounting_vertex_count=MOUNTING_VERTEX_COUNT,
            min_bbox_limits_mm=SHAPE_MIN_BBOX_LIMITS_MM,
            tip_keys=FINGERTIP_KEYS,
            quality_columns=OBSERVED_FINGERTIPS,
            env_label="NeroGrasp",
        )

    # ------------------------------------------------------------------ #
    # Scene & post-build
    # ------------------------------------------------------------------ #
    def _robot_root(self, tip_mesh_file: str) -> ET.Element:
        root = ET.parse(ROBOT_XML).getroot()
        compiler = root.find("compiler")
        if compiler is not None:
            compiler.attrib.pop("meshdir", None)
        for name in FINGERTIP_MESH_ASSETS:
            mesh = root.find(f".//mesh[@name='{name}']")
            if mesh is None:
                raise ValueError(f"Missing fingertip mesh asset '{name}' in {ROBOT_XML}")
            mesh.set("file", tip_mesh_file)
        for mesh in root.iter("mesh"):
            if "file" in mesh.attrib:
                mesh.attrib["file"] = os.path.abspath(os.path.join(ASSETS_DIR, mesh.attrib["file"]))
        for body in root.iter("body"):
            if body.attrib.get("name") in STRIPPED_INERTIAL_BODIES and body.find("inertial") is not None:
                body.remove(body.find("inertial"))
        return root

    def _scene_root(self, robot_root: ET.Element) -> ET.Element:
        robot = ET.fromstring(ET.tostring(robot_root))
        for tip in self.fingertips:
            body = robot.find(f".//body[@name='{tip.name}']")
            body.remove(next(g for g in body.findall("geom") if g.get("name") == tip.name))
            for part_id in range(self.config.convex_max_parts):
                ET.SubElement(
                    body, "geom",
                    {"class": f"{tip.name}_trimesh", "name": f"{tip.name}_part_{part_id:02d}", "type": "mesh", "mesh": tip.mesh_name},
                )
        scene = ET.Element("mujoco")
        for part in (
            ET.parse(ASSETS_DIR / "scenes" / "floor.xml").getroot(),
            ET.parse(ASSETS_DIR / "scenes" / "small_obj_defaults.xml").getroot(),
            object_body_element(self.max_object_boxes, self.max_object_cylinders),
            robot,
        ):
            scene.extend(list(part))
        return scene

    def _place_variant_object(self, spec: mujoco.MjSpec, object_name: str | None) -> None:
        place_object(spec, self.templates[object_name], self.max_object_boxes, self.max_object_cylinders)

    def _robot_dof_addrs(self) -> list[int]:
        return self.arm_qvel_addrs + self.hand_qvel_addrs

    def _post_build(self) -> None:
        m = self.m
        obj_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY_NAME)
        assert obj_body_id != -1, f"Object body '{OBJECT_BODY_NAME}' not found in model"
        self._init_common_model_indices(m, obj_body_id, OBSERVED_FINGERTIPS, obj_colliding_only=False)
        self._init_joint_addresses(m)
        self._init_spawn_state(m)

    def _init_joint_addresses(self, m: mujoco.MjModel) -> None:
        geom_size = self.cp_model.geom_size.numpy()
        self.env_active_geom_mask = torch.tensor(geom_size[:, self.all_obj_geom_ids, 0] > self.MIN_ACTIVE_GEOM_SIZE, dtype=torch.bool, device=self.device)
        obj_slot = np.full(m.ngeom, -1, dtype=np.int64)
        obj_slot[self.all_obj_geom_ids] = np.arange(len(self.all_obj_geom_ids))
        self._geom_obj_slot_t = torch.as_tensor(obj_slot, dtype=torch.long, device=self.device)

        arm = set(self.arm_qpos_addrs)
        self.hand_qpos_addrs, self.hand_qvel_addrs = [], []
        for j in range(m.njnt):
            jtype = m.jnt_type[j]
            if jtype == mujoco.mjtJoint.mjJNT_FREE or m.jnt_qposadr[j] in arm:
                continue
            ball = jtype == mujoco.mjtJoint.mjJNT_BALL
            self.hand_qpos_addrs.extend(range(int(m.jnt_qposadr[j]), int(m.jnt_qposadr[j]) + (4 if ball else 1)))
            self.hand_qvel_addrs.extend(range(int(m.jnt_dofadr[j]), int(m.jnt_dofadr[j]) + (3 if ball else 1)))

        n_obj_onehot = len(self.obj_names) if self.one_hot_id else 0
        actor_prefix = 1 + (0 if self.asymmetric_rl else n_obj_onehot) + len(self.arm_qpos_addrs) + len(self.hand_qpos_addrs) + 3 * len(self.ftip_ids)
        self.rl_critic_obs_dim = actor_prefix + 3 + 4 + 1 + (n_obj_onehot if self.asymmetric_rl else 0) + 1 + 1
        self.rl_obs_dim = actor_prefix if self.asymmetric_rl else self.rl_critic_obs_dim
        self.rl_act_dim = len(self.hand_qpos_addrs)

        obj_joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, OBJECT_FREEJOINT_NAME)
        if obj_joint == -1:
            raise ValueError(f"Object freejoint '{OBJECT_FREEJOINT_NAME}' not found in model.")
        self.obj_qpos_adr = int(m.jnt_qposadr[obj_joint])
        self.obj_qvel_adr = int(m.jnt_dofadr[obj_joint])

    def _init_spawn_state(self, m: mujoco.MjModel) -> None:
        saved_qpos = self.cp_model.qpos0.numpy().copy()
        saved_qvel = np.zeros((self.num_envs, m.nv), dtype=np.float32)
        saved_qpos[:, self.arm_qpos_addrs] = self.arm_traj[0].astype(np.float32)
        saved_qpos[:, self.hand_qpos_addrs] = np.tile(self.hand_start_pose, (self.num_envs, 1)).astype(np.float32)
        if len(self.hand_qpos_addrs) >= 12:
            saved_qpos[:, self.hand_qpos_addrs[4:12]] = LEAP_HAND_INACTIVE_FINGERS_REST

        self.arm_actuator_low = torch.as_tensor(m.actuator_ctrlrange[:NERO_DOF, 0], dtype=torch.float32, device=self.device)
        self.arm_actuator_high = torch.as_tensor(m.actuator_ctrlrange[:NERO_DOF, 1], dtype=torch.float32, device=self.device)
        actual = m.actuator_ctrlrange[NERO_DOF:NERO_DOF + self.rl_act_dim]
        low, high = np.maximum(-1.0, actual[:, 0]), np.minimum(1.0, actual[:, 1])
        for i in np.flatnonzero(low > high):
            low[i] = high[i] = actual[i, 0] if actual[i, 0] >= 1.0 else actual[i, 1]
        self.hand_actuator_low = torch.tensor(low, dtype=torch.float32, device=self.device)
        self.hand_actuator_high = torch.tensor(high, dtype=torch.float32, device=self.device)
        self._finalize_post_build(saved_qpos, saved_qvel, self.cp_model.body_mass.numpy())

    # ------------------------------------------------------------------ #
    # Washer sizing & episode loop
    # ------------------------------------------------------------------ #
    def _sample_washer_dimensions(self) -> tuple[np.ndarray, np.ndarray]:
        radii = np.full(self.num_envs, np.nan, dtype=np.float32)
        half_heights = np.full(self.num_envs, np.nan, dtype=np.float32)
        for i, obj_name in enumerate(self.obj_names_env):
            if obj_name != "washer":
                continue
            scale = float(self.noise_multiplier_t[i].item())
            rng = self._rng if self._lane_rngs is None else self._lane_rngs[i]
            radii[i] = rng.uniform(WASHER_DEFAULT_RADIUS - WASHER_MAX_RANDOMIZATION_DELTA * scale,
                                   WASHER_DEFAULT_RADIUS + WASHER_MAX_RANDOMIZATION_DELTA * scale)
            half_heights[i] = rng.uniform(max(0.0002, WASHER_DEFAULT_HALF_HEIGHT - WASHER_MAX_HALF_HEIGHT_DELTA * scale),
                                          WASHER_DEFAULT_HALF_HEIGHT + 3.0 * WASHER_MAX_HALF_HEIGHT_DELTA * scale)
        return radii, half_heights

    def _apply_washers(self) -> None:
        has_washer = "washer" in self.obj_names_env
        geom_size = self.cp_model.geom_size.numpy()
        geom_aabb = self.cp_model.geom_aabb.numpy().reshape(self.num_envs, -1, 2, 3) if has_washer else None
        geom_rbound = self.cp_model.geom_rbound.numpy() if has_washer else None
        body_mass = self.cp_model.body_mass.numpy() if has_washer else None
        body_subtreemass = self.cp_model.body_subtreemass.numpy() if has_washer else None
        body_inertia = self.cp_model.body_inertia.numpy() if has_washer else None
        keypoints = np.zeros((self.num_envs, self.NUM_KEYPOINTS * 3), dtype=np.float32)
        for i, obj_name in enumerate(self.obj_names_env):
            template = self.templates[obj_name]
            radius = half_height = None
            if obj_name == "washer":
                radius, half_height = float(self.washer_radii_env[i]), float(self.washer_half_heights_env[i])
                r0, h0 = float(template.cylinder_geoms[0].size[0]), float(template.cylinder_geoms[0].size[1])
                mass_scale = (radius / r0) ** 2 * (half_height / h0)
                new_mass = float(template.mass * mass_scale)
                ixy_scale = mass_scale * (3.0 * radius ** 2 + 4.0 * half_height ** 2) / (3.0 * r0 ** 2 + 4.0 * h0 ** 2)
                iz_scale = mass_scale * (radius / r0) ** 2
                base_I = template.fullinertia
                body_mass[i, self.obj_body_id] = body_subtreemass[i, self.obj_body_id] = new_mass
                body_inertia[i, self.obj_body_id] = np.array([base_I[0] * ixy_scale, base_I[1] * ixy_scale, base_I[2] * iz_scale], dtype=np.float32)
                self.obj_mass_obs[i, 0] = new_mass * self.MASS_OBS_SCALE
                self.obj_masses_env[i] = new_mass
                aabb_half = np.array([radius, radius, half_height], dtype=np.float32)
                rbound = float(np.hypot(radius, half_height))
                for j, cylinder in enumerate(template.cylinder_geoms):
                    gid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"{OBJECT_CYL_PREFIX}_{j:02d}")
                    size = np.array([radius, half_height, cylinder.size[2]], dtype=np.float32)
                    geom_size[i, gid] = size
                    geom_aabb[i, gid, 0] = 0.0
                    geom_aabb[i, gid, 1] = aabb_half
                    geom_rbound[i, gid] = rbound
                    variant = self.variants[i] if self.variants is not None else None
                    if variant is not None:
                        variant.geom_size[gid] = size.astype(np.float64)
                        variant.geom_aabb[gid, :3] = 0.0
                        variant.geom_aabb[gid, 3:] = aabb_half
                        variant.geom_rbound[gid] = rbound
            keypoints[i] = object_keypoints(template, radius, half_height, self.NUM_KEYPOINTS).flatten()
        self.cached_keypoint_obs = torch.tensor(keypoints, device=self.device, dtype=torch.float32) * self.KEYPOINT_OBS_SCALE
        with wp.ScopedDevice(self.device):
            wp.copy(self.cp_model.geom_size, wp.array(geom_size, dtype=wp.vec3, device=self.device))
            if has_washer:
                wp.copy(self.cp_model.geom_aabb, wp.array(geom_aabb, dtype=wp.vec3, device=self.device))
                wp.copy(self.cp_model.geom_rbound, wp.array(geom_rbound, dtype=wp.float32, device=self.device))
                wp.copy(self.cp_model.body_mass, wp.array(body_mass, dtype=wp.float32, device=self.device))
                wp.copy(self.cp_model.body_subtreemass, wp.array(body_subtreemass, dtype=wp.float32, device=self.device))
                wp.copy(self.cp_model.body_inertia, wp.array(body_inertia, dtype=wp.vec3, device=self.device))
            mjw.set_const(self.cp_model, self.cp_data)

    def reset_worlds(self) -> dict[str, torch.Tensor]:
        self._zero_episode_buffers()
        self.washer_radii_env, self.washer_half_heights_env = self._sample_washer_dimensions()
        self._apply_washers()
        target_qpos = wp.to_torch(self.saved_qpos).clone()
        target_qvel = wp.to_torch(self.saved_qvel).clone()
        for i, obj_name in enumerate(self.obj_names_env):
            if obj_name == "washer":
                target_qpos[i, self.obj_qpos_adr + 2] = float(SPAWN_FLOOR_CLEARANCE_M + self.washer_half_heights_env[i])
        self._reset_worlds_common(target_qpos, target_qvel)
        return self._get_obs()

    def step_worlds(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        hand_actions = actions.clone().to(dtype=torch.float32, device=self.device)
        traj_idx = min(self.current_steps, len(self.arm_traj) - 1)
        arm_ctrl = torch.clamp(self._gpu_arm_traj[traj_idx].unsqueeze(0).expand(self.num_envs, -1), self.arm_actuator_low, self.arm_actuator_high)
        if self.demo:
            target = self._gpu_hand_traj[traj_idx].unsqueeze(0).expand(self.num_envs, -1)
            hand_ctrl = torch.clamp(target + hand_actions * self.RESIDUAL_ACTION_SCALE, self.hand_actuator_low, self.hand_actuator_high)
        else:
            current = wp.to_torch(self.cp_data.qpos)[:, self.hand_qpos_addrs]
            hand_ctrl = torch.clamp(current + hand_actions / self.REAL_CONTROL_HZ, self.hand_actuator_low, self.hand_actuator_high)
        ctrl = torch.cat([arm_ctrl, hand_ctrl], dim=-1)
        ctrl[:, NERO_DOF + 4:NERO_DOF + 12] = self._gpu_inactive_fingers
        return self._finish_step(ctrl, obj_active_mask=self.env_active_geom_mask)

    # ------------------------------------------------------------------ #
    # Observation, contacts, rewards
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> dict[str, torch.Tensor]:
        xpos = wp.to_torch(self.cp_data.xpos)
        xquat = wp.to_torch(self.cp_data.xquat)
        qpos = wp.to_torch(self.cp_data.qpos)
        obj_xyz, obj_quat = xpos[:, self.obj_body_id, :], xquat[:, self.obj_body_id, :]
        ftip_pos = xpos[:, self.ftip_ids, :].reshape(self.num_envs, -1)
        robot_q = torch.cat([qpos[:, self.arm_qpos_addrs], qpos[:, self.hand_qpos_addrs]], dim=1)
        self._last_contacts = self._compute_contacts()
        in_contact = self._last_contacts[0].unsqueeze(1).float()
        steps = torch.full((self.num_envs, 1), self.current_steps / self.max_episode_steps, dtype=torch.float32, device=self.device)
        obj_mass = torch.as_tensor(self.obj_mass_obs[:, :1], dtype=torch.float32, device=self.device)
        if self.cached_keypoint_obs is None:
            self.cached_keypoint_obs = torch.zeros(self.num_envs, self.NUM_KEYPOINTS * 3, device=self.device, dtype=torch.float32)

        local_kps = (self.cached_keypoint_obs / self.KEYPOINT_OBS_SCALE).view(self.num_envs, self.NUM_KEYPOINTS, 3)
        self._last_world_kps = self._rotate_points(obj_quat, local_kps, obj_xyz)

        one_hot = None
        if self.one_hot_id:
            index = torch.as_tensor([self.obj_names.index(n) for n in self.obj_names_env], dtype=torch.long, device=self.device)
            one_hot = torch.nn.functional.one_hot(index, num_classes=len(self.obj_names)).to(torch.float32)
        actor = [steps, robot_q, ftip_pos] + ([one_hot] if one_hot is not None and not self.asymmetric_rl else [])
        actor_obs = torch.cat(actor, dim=1)
        privileged = [obj_xyz, obj_quat, obj_mass] + ([one_hot] if one_hot is not None and self.asymmetric_rl else [])
        privileged += [in_contact, torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)]
        critic_obs = torch.cat([actor_obs, torch.cat(privileged, dim=1)], dim=1)
        return {"actor": actor_obs if self.asymmetric_rl else critic_obs, "critic": critic_obs}

    def _compute_contacts(self) -> tuple[torch.Tensor, torch.Tensor]:
        touches = torch.zeros((self.num_envs, len(self.ftip_ids)), dtype=torch.bool, device=self.device)
        robot_on_floor = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        num_contacts = int(wp.to_torch(self.cp_data.nacon)[0].item())
        if num_contacts > 0:
            pairs = wp.to_torch(self.cp_data.contact.geom)[:num_contacts]
            worlds = wp.to_torch(self.cp_data.contact.worldid)[:num_contacts].to(torch.long)
            geom_a, geom_b = pairs[:, 0].to(torch.long), pairs[:, 1].to(torch.long)
            for tip_geom, obj_geom in ((geom_a, geom_b), (geom_b, geom_a)):
                col, slot = self._geom_ftip_col_t[tip_geom], self._geom_obj_slot_t[obj_geom]
                hit = (col >= 0) & (slot >= 0)
                if hit.any():
                    w_hit, s_hit = worlds[hit], slot[hit]
                    active = self.env_active_geom_mask[w_hit, s_hit]
                    if active.any():
                        touches[w_hit[active], col[hit][active]] = True
            robot_on_floor = self._floor_contact_mask(geom_a, geom_b, worlds)
        self.ftip_touches_object = touches
        return touches.all(dim=1), robot_on_floor

    def _compute_rewards(self, in_contact: torch.Tensor) -> torch.Tensor:
        obj_height, dist_reward, contact_reward = self._base_distance_and_contact_rewards(in_contact)
        height_reward = obj_height * in_contact.float()
        rewards = height_reward + contact_reward + dist_reward
        terminal_reward = torch.zeros_like(rewards)
        if self.current_steps >= self.max_episode_steps:
            terminal_reward = (in_contact & (obj_height > self.SUCCESS_LIFT_HEIGHT_M)).float() * self.TERMINAL_SUCCESS_BONUS
            rewards = rewards + terminal_reward

        self.last_contact_reward = contact_reward
        self.last_dist_reward_scaled = dist_reward
        self.last_f2f_penalty = torch.zeros_like(contact_reward)
        rewards[self.terminated_mask] = 0.0
        self._accumulate_reward_terms({
            "height_reward": height_reward,
            "contact_reward": contact_reward,
            "dist_reward": dist_reward,
            "f2f_penalty": -self.last_f2f_penalty,
            "terminal_reward": terminal_reward,
            "total": rewards,
        })
        return rewards

    def _add_debug_visualizations(self, scene: mujoco.MjvScene, env_idx: int) -> None:
        if self.cached_keypoint_obs is None or self._render_d is None:
            return
        obj_xpos, obj_xquat = self._render_d.xpos[self.obj_body_id], self._render_d.xquat[self.obj_body_id]
        local = (self.cached_keypoint_obs[env_idx] / self.KEYPOINT_OBS_SCALE).view(self.NUM_KEYPOINTS, 3).detach().cpu().numpy()
        world = Rotation.from_quat([obj_xquat[1], obj_xquat[2], obj_xquat[3], obj_xquat[0]]).apply(local) + obj_xpos
        markers = [(p, self.DEBUG_KEYPOINT_RADIUS, self.DEBUG_KEYPOINT_COLOR) for p in world]
        markers += [(self._render_d.xpos[f].copy(), self.DEBUG_FTIP_RADIUS, self.DEBUG_FTIP_COLOR) for f in self.ftip_ids]
        self._draw_debug_markers(scene, markers)
