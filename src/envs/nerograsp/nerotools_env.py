"""NeroTools: Nero arm with a 2-jaw parallel gripper and designed fingertip pads interacting with articulated tools."""

from __future__ import annotations

import os
import random
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import torch
import warp as wp
from scipy.spatial.transform import Rotation

from .base_env import EPISODE_STEPS, NeroBaseEnv
from .mesh_switch_env import MeshSwitchMujocoEnv
from .config import (
    TOOLS_DEFORMABLE_TIP_Y_PERCENTILE,
    TOOLS_FINGERTIP_BODIES,
    TOOLS_FINGERTIP_KEYS,
    TOOLS_FINGERTIP_MESH,
    TOOLS_MAX_BBOX_LIMITS_MM,
    TOOLS_MIN_BBOX_LIMITS_MM,
    TOOLS_MOUNTING_VERTEX_COUNT,
    TOOLS_SNAP_PLANE_TARGET_XY,
    NeroToolsConfig,
    resolve_worker_count,
)
from .nerotools_objects import TOOL_FREEJOINT_NAME, generate_tool_keypoints, load_tool_xml
from .shape_deformation.adapter import load_fingertips
from .traj_utils import (
    ASSETS_DIR,
    GRIPPER_ROBOT_XML,
    NERO_DOF,
    NERO_NARROW_LIMS_HIGH,
    NERO_NARROW_LIMS_LOW,
    NERO_START_POSE,
    PARALLEL_GRIPPER_MAX_FING_DIST,
    make_arm_demo_qpos_traj,
    make_gripper_demo_qpos_traj,
)

STRIPPED_GRIPPER_INERTIAL_BODIES = ("nero_gripper_link1", "nero_gripper_link2")
GRIPPER_JOINT_NAMES = ("nero_gripper_joint1", "nero_gripper_joint2")
OBSERVED_TOOL_FINGERTIPS = ("nero_gripper_link1_tip", "nero_gripper_link2_tip")


@wp.kernel
def _tools_contacts_kernel(
    nacon: wp.array(dtype=wp.int32),
    geom_pairs: wp.array(dtype=wp.vec2i),
    worldids: wp.array(dtype=wp.int32),
    geom_ftip_col: wp.array(dtype=wp.int32),
    geom_is_obj: wp.array(dtype=wp.uint8),
    geom_is_part_a: wp.array(dtype=wp.uint8),
    geom_is_part_b: wp.array(dtype=wp.uint8),
    geom_is_robot_arm: wp.array(dtype=wp.uint8),
    floor_geom_id: wp.int32,
    touches: wp.array(dtype=wp.uint8, ndim=2),
    touches_a: wp.array(dtype=wp.uint8, ndim=2),
    touches_b: wp.array(dtype=wp.uint8, ndim=2),
    robot_on_floor: wp.array(dtype=wp.uint8, ndim=1),
):
    cid = wp.tid()
    if cid >= nacon[0]:
        return
    w = worldids[cid]
    pair = geom_pairs[cid]
    ga = pair[0]
    gb = pair[1]
    col_a = geom_ftip_col[ga]
    if col_a >= 0 and geom_is_obj[gb] != wp.uint8(0):
        touches[w, col_a] = wp.uint8(1)
        if geom_is_part_a[gb] != wp.uint8(0):
            touches_a[w, col_a] = wp.uint8(1)
        if geom_is_part_b[gb] != wp.uint8(0):
            touches_b[w, col_a] = wp.uint8(1)
    col_b = geom_ftip_col[gb]
    if col_b >= 0 and geom_is_obj[ga] != wp.uint8(0):
        touches[w, col_b] = wp.uint8(1)
        if geom_is_part_a[ga] != wp.uint8(0):
            touches_a[w, col_b] = wp.uint8(1)
        if geom_is_part_b[ga] != wp.uint8(0):
            touches_b[w, col_b] = wp.uint8(1)
    if floor_geom_id >= 0:
        if (ga == floor_geom_id and geom_is_robot_arm[gb] != wp.uint8(0)) or (
            gb == floor_geom_id and geom_is_robot_arm[ga] != wp.uint8(0)
        ):
            robot_on_floor[w] = wp.uint8(1)


class NeroToolsEnv(NeroBaseEnv):
    """``num_envs`` parallel-gripper Nero tool-manipulation worlds sharing one episode clock."""

    EXTRA_TIP_BODIES = tuple(TOOLS_FINGERTIP_BODIES)
    TERMINAL_SUCCESS_BONUS = 10.0
    TARGET_HEIGHT = 0.4
    CARABINER_CLOSED_RAD = -0.51
    ARTICULATION_THRESHOLD = 0.20
    LIFT_MULTIPLIER = 10.0
    ARTICULATION_MULTIPLIER = 2.0

    def __init__(self, *, config: NeroToolsConfig, num_envs: int, is_eval_env: bool = False, render_mode: str | None = None):
        self.render_mode = render_mode
        self._reset_failed = True
        self._episode_over = False
        self._needs_rebuild = False
        self.config = config
        self.obj_name = config.object_name
        self.obj_names = [self.obj_name]
        self.obj_names_env = [self.obj_name] * num_envs
        workers = resolve_worker_count(config.shape_parallel_workers)
        MeshSwitchMujocoEnv.__init__(
            self,
            num_envs,
            device=config.device or ("cpu" if config.use_cpu else "cuda"),
            debug=config.debug,
            verbose=config.verbose,
            compile_workers=workers,
            retain_all_render_models=render_mode is not None,
            njmax=512,
            nconmax=256,
        )
        self.asymmetric_rl = config.asymmetric_observations
        self._rng = random.Random(config.seed)
        self._lane_rngs: list[random.Random] | None = None
        self._action_rng = np.random.default_rng(config.seed)
        self._lane_action_rngs: list[np.random.Generator] | None = None
        self._gpu_action_gen = torch.Generator(device=self.device)
        self._gpu_action_gen.manual_seed(int(config.seed))

        robot_root = self._robot_root()
        self.fingertips = load_fingertips(
            robot_root,
            str(ASSETS_DIR),
            TOOLS_DEFORMABLE_TIP_Y_PERCENTILE,
            fingertip_bodies=TOOLS_FINGERTIP_BODIES,
            snap_plane_target_xy=TOOLS_SNAP_PLANE_TARGET_XY,
            mounting_vertex_count=TOOLS_MOUNTING_VERTEX_COUNT,
        )
        self._init_fingertip_pipeline(
            fingertips=self.fingertips,
            mounting_vertex_count=TOOLS_MOUNTING_VERTEX_COUNT,
            max_bbox_limits_mm=TOOLS_MAX_BBOX_LIMITS_MM,
            scene_root=self._scene_root(robot_root),
            workers=workers,
        )

        self.max_episode_steps = EPISODE_STEPS
        self.demo = config.trajectory == "demo"
        self._gpu_demo_traj: torch.Tensor | None = None
        if self.demo:
            arm_traj = make_arm_demo_qpos_traj(self.max_episode_steps, robot_xml=GRIPPER_ROBOT_XML)
            grip_traj = make_gripper_demo_qpos_traj(self.max_episode_steps)
            opening = (grip_traj[:, 0] - grip_traj[:, 1])[:, None]
            self._gpu_demo_traj = torch.tensor(np.concatenate([arm_traj, opening], axis=1), dtype=torch.float32, device=self.device)

        has_noise = (
            config.object_position_randomization > 0.0
            or config.object_rotation_randomization_deg > 0.0
            or config.gripper_action_noise_std > 0.0
        )
        self._init_noise_multiplier(has_noise, per_object=False)

        self.initial_world_kps = None
        self.initial_obj_pos = None
        self.last_artic_ratio = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        canonical = {tip.name: self.postprocessor.postprocess(tip, tip.vertices, tip.faces) for tip in self.fingertips}
        self.build_model([canonical] * num_envs)
        self._init_codesign(
            num_envs=num_envs,
            is_eval_env=is_eval_env,
            mounting_vertex_count=TOOLS_MOUNTING_VERTEX_COUNT,
            min_bbox_limits_mm=TOOLS_MIN_BBOX_LIMITS_MM,
            tip_keys=TOOLS_FINGERTIP_KEYS,
            quality_columns=OBSERVED_TOOL_FINGERTIPS,
            env_label="NeroGrasp-Tools",
        )

    # ------------------------------------------------------------------ #
    # Scene & post-build
    # ------------------------------------------------------------------ #
    def _robot_root(self) -> ET.Element:
        root = ET.parse(GRIPPER_ROBOT_XML).getroot()
        compiler = root.find("compiler")
        if compiler is not None:
            compiler.attrib.pop("meshdir", None)
            compiler.set("autolimits", "true")
        else:
            root.insert(0, ET.Element("compiler", {"angle": "radian", "autolimits": "true"}))
        keyframe = root.find("keyframe")
        if keyframe is not None:
            root.remove(keyframe)
        for mesh_name in TOOLS_FINGERTIP_BODIES.values():
            mesh_el = root.find(f".//mesh[@name='{mesh_name}']")
            if mesh_el is None:
                raise ValueError(f"Missing fingertip mesh asset '{mesh_name}' in {GRIPPER_ROBOT_XML}")
            mesh_el.set("file", TOOLS_FINGERTIP_MESH)
            mesh_el.attrib.pop("content_type", None)
        for mesh_el in root.iter("mesh"):
            if "file" in mesh_el.attrib:
                mesh_el.attrib["file"] = os.path.abspath(os.path.join(ASSETS_DIR, mesh_el.attrib["file"]))
        for body in root.iter("body"):
            if body.attrib.get("name") in STRIPPED_GRIPPER_INERTIAL_BODIES and body.find("inertial") is not None:
                body.remove(body.find("inertial"))
        return root

    def _scene_root(self, robot_root: ET.Element) -> ET.Element:
        robot = ET.fromstring(ET.tostring(robot_root))
        for tip in self.fingertips:
            body = robot.find(f".//body[@name='{tip.name}']")
            target_geom = next(g for g in body.findall("geom") if g.get("mesh") == tip.mesh_name)
            geom_class = target_geom.get("class", "gripper_geom")
            body.remove(target_geom)
            for part_id in range(self.config.convex_max_parts):
                ET.SubElement(
                    body, "geom",
                    {"class": geom_class, "name": f"{tip.name}_part_{part_id:02d}", "type": "mesh", "mesh": tip.mesh_name},
                )
        scene = ET.Element("mujoco")
        visual = ET.SubElement(scene, "visual")
        ET.SubElement(visual, "global", {"offwidth": "1920", "offheight": "1080"})
        tool_root = load_tool_xml(ASSETS_DIR / "tasks" / "tools", self.obj_name)
        for part in (ET.parse(ASSETS_DIR / "scenes" / "floor.xml").getroot(), robot, tool_root):
            scene.extend(list(part))
        return scene

    def _robot_dof_addrs(self) -> list[int]:
        return self.robot_dof_addrs

    def _post_build(self) -> None:
        m = self.m
        obj_joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, TOOL_FREEJOINT_NAME)
        if obj_joint == -1:
            raise ValueError(f"Tool freejoint '{TOOL_FREEJOINT_NAME}' not found in model.")
        self.obj_qpos_adr = int(m.jnt_qposadr[obj_joint])
        self.obj_qvel_adr = int(m.jnt_dofadr[obj_joint])
        self._init_common_model_indices(m, int(m.jnt_bodyid[obj_joint]), OBSERVED_TOOL_FINGERTIPS, obj_colliding_only=True)
        self._init_joint_addresses(m)
        self._init_spawn_state(m)

    def _init_joint_addresses(self, m: mujoco.MjModel) -> None:
        if self.obj_name == "carabiner":
            self.gate_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link_gate")
            self.spine_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link_bracket")
            self._part_a_body_id, self._part_b_body_id = self.gate_body_id, self.spine_body_id
        elif self.obj_name == "wire_cutters":
            self.gate_body_id = -1
            self.handle1_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wire_cutters_handle")
            self.handle2_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "wire_cutters_handle_2")
            self._part_a_body_id, self._part_b_body_id = self.handle1_body_id, self.handle2_body_id
        else:
            raise ValueError(f"Unsupported tool object: '{self.obj_name}'")

        is_obj = self._geom_is_obj_t.cpu().numpy()
        is_part_a = is_obj & (m.geom_bodyid == self._part_a_body_id)
        is_part_b = is_obj & (m.geom_bodyid == self._part_b_body_id)
        self._geom_is_part_a_t = torch.as_tensor(is_part_a, dtype=torch.bool, device=self.device)
        self._geom_is_part_b_t = torch.as_tensor(is_part_b, dtype=torch.bool, device=self.device)
        with wp.ScopedDevice(self.device):
            self._geom_ftip_col_wp = wp.array(self._geom_ftip_col_t.to(torch.int32).cpu().numpy(), dtype=wp.int32)
            self._geom_is_obj_wp = wp.array(is_obj.astype(np.uint8), dtype=wp.uint8)
            self._geom_is_part_a_wp = wp.array(is_part_a.astype(np.uint8), dtype=wp.uint8)
            self._geom_is_part_b_wp = wp.array(is_part_b.astype(np.uint8), dtype=wp.uint8)
            self._geom_is_robot_arm_wp = wp.array(self._geom_is_robot_arm_t.cpu().numpy().astype(np.uint8), dtype=wp.uint8)
        self._touches_u8 = torch.zeros((self.num_envs, len(self.ftip_ids)), dtype=torch.uint8, device=self.device)
        self._touches_a_u8 = torch.zeros_like(self._touches_u8)
        self._touches_b_u8 = torch.zeros_like(self._touches_u8)
        self._robot_on_floor_u8 = torch.zeros(self.num_envs, dtype=torch.uint8, device=self.device)

        self.gripper_qpos_addrs, self.gripper_qvel_addrs = [], []
        for name in GRIPPER_JOINT_NAMES:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j != -1:
                self.gripper_qpos_addrs.append(int(m.jnt_qposadr[j]))
                self.gripper_qvel_addrs.append(int(m.jnt_dofadr[j]))
        self.robot_qpos_addrs = self.arm_qpos_addrs + self.gripper_qpos_addrs
        self.robot_dof_addrs = self.arm_qvel_addrs + self.gripper_qvel_addrs
        self.gripper_act_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "nero_gripper")
        assert self.gripper_act_id != -1, "Missing 'nero_gripper' actuator in model"
        self.gripper_max_range = float(m.actuator_ctrlrange[self.gripper_act_id, 1])
        self._grip_std_t = (
            self.config.gripper_action_noise_std * self.gripper_max_range * self.noise_multiplier_t.squeeze(-1)
        )
        self._has_gripper_noise = bool(self.config.gripper_action_noise_std > 0.0 and (self._grip_std_t > 0.0).any().item())
        self._noise_mults_np = self.noise_multiplier_t.squeeze(-1).detach().cpu().numpy()

        self.task_joint_name, self.task_joint_adr = None, None
        self.task_joint_lower_limit, self.task_joint_limit = 0.0, 1.0
        for jid in range(m.njnt):
            if m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE and int(m.jnt_bodyid[jid]) in self.obj_body_set:
                self.task_joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
                self.task_joint_adr = int(m.jnt_qposadr[jid])
                lims = m.jnt_range[jid]
                if self.obj_name == "carabiner":
                    self.task_joint_lower_limit = self.CARABINER_CLOSED_RAD
                    self.task_joint_limit = float(lims[1] - self.CARABINER_CLOSED_RAD)
                else:
                    self.task_joint_lower_limit = float(lims[0])
                    self.task_joint_limit = float(max(abs(lims[0]), abs(lims[1])))
                break

        actor_dim = 1 + len(self.robot_qpos_addrs) + 3 * len(self.ftip_ids) + self.NUM_KEYPOINTS * 3
        self.rl_critic_obs_dim = actor_dim + 3 + 4 + 1 + self.NUM_KEYPOINTS * 3 + 1 + 1
        self.rl_obs_dim = actor_dim if self.asymmetric_rl else self.rl_critic_obs_dim
        self.rl_act_dim = NERO_DOF + 1

    def _init_spawn_state(self, m: mujoco.MjModel) -> None:
        tool_kps = generate_tool_keypoints(
            m, self.d, self.obj_name, self.obj_body_id, self.all_obj_geom_ids,
            gate_body_id=self.gate_body_id, handle1_body_id=getattr(self, "handle1_body_id", -1), num_keypoints=self.NUM_KEYPOINTS,
        )
        self.cached_keypoint_obs = (
            torch.tensor(tool_kps, dtype=torch.float32, device=self.device)
            .repeat(self.num_envs, 1).view(self.num_envs, self.NUM_KEYPOINTS * 3) * self.KEYPOINT_OBS_SCALE
        )
        saved_qpos = self.cp_model.qpos0.numpy().copy()
        saved_qvel = np.zeros((self.num_envs, m.nv), dtype=np.float32)
        start_robot = np.concatenate([NERO_START_POSE, np.array([PARALLEL_GRIPPER_MAX_FING_DIST, -PARALLEL_GRIPPER_MAX_FING_DIST], dtype=np.float32)]).astype(np.float32)
        saved_qpos[:, self.robot_qpos_addrs] = start_robot
        self.arm_actuator_low = torch.tensor(NERO_NARROW_LIMS_LOW, dtype=torch.float32, device=self.device)
        self.arm_actuator_high = torch.tensor(NERO_NARROW_LIMS_HIGH, dtype=torch.float32, device=self.device)
        self._finalize_post_build(saved_qpos, saved_qvel, self.cp_model.body_mass.numpy())

    # ------------------------------------------------------------------ #
    # Seeding, noise, episode loop
    # ------------------------------------------------------------------ #
    def _seed_extra(self, parsed) -> None:
        if isinstance(parsed, int):
            self._action_rng, self._lane_action_rngs = np.random.default_rng(parsed), None
            self._gpu_action_gen = torch.Generator(device=self.device)
            self._gpu_action_gen.manual_seed(int(parsed))
        elif parsed is not None:
            self._lane_action_rngs = [np.random.default_rng(s) for s in parsed]

    def _extra_step_info(self) -> dict:
        return {"task/articulation_ratio": self.last_artic_ratio.detach().clone()}

    def _extra_optional_info(self) -> dict:
        return {"rewards/articulation": getattr(self, "last_articulation_reward", None)}

    def _apply_pose_randomization(self, target_qpos: torch.Tensor) -> None:
        pos_rand, yaw_rand = float(self.config.object_position_randomization), float(self.config.object_rotation_randomization_deg)
        if pos_rand <= 0.0 and yaw_rand <= 0.0:
            return
        mults = self._noise_mults_np
        dx = np.zeros(self.num_envs, dtype=np.float32)
        dy = np.zeros(self.num_envs, dtype=np.float32)
        yaw_rad = np.zeros(self.num_envs, dtype=np.float32)
        deg_to_rad = np.pi / 180.0
        for i in range(self.num_envs):
            mult = float(mults[i])
            if mult <= 0.0:
                continue
            rng = self._rng if self._lane_rngs is None else self._lane_rngs[i]
            dx[i] = rng.uniform(-1.0, 1.0) * pos_rand * mult
            dy[i] = rng.uniform(-1.0, 1.0) * pos_rand * mult
            yaw_rad[i] = rng.uniform(-1.0, 1.0) * yaw_rand * mult * deg_to_rad
        adr = self.obj_qpos_adr
        deltas = torch.as_tensor(np.stack([dx, dy], axis=1), dtype=torch.float32, device=self.device)
        target_qpos[:, adr:adr + 2].add_(deltas)
        if yaw_rand > 0.0:
            half = torch.as_tensor(yaw_rad * 0.5, dtype=torch.float32, device=self.device)
            c = torch.cos(half)
            s = torch.sin(half)
            w = target_qpos[:, adr + 3]
            x = target_qpos[:, adr + 4]
            y = target_qpos[:, adr + 5]
            z = target_qpos[:, adr + 6]
            # Left-multiply by z-axis rotation quat (c, 0, 0, s) in wxyz order
            nw = c * w - s * z
            nx = c * x - s * y
            ny = c * y + s * x
            nz = c * z + s * w
            target_qpos[:, adr + 3] = nw
            target_qpos[:, adr + 4] = nx
            target_qpos[:, adr + 5] = ny
            target_qpos[:, adr + 6] = nz

    def _sample_gripper_noise(self) -> torch.Tensor:
        if self._lane_action_rngs is None:
            return torch.randn(self.num_envs, generator=self._gpu_action_gen, dtype=torch.float32, device=self.device)
        draws = np.array([rng.standard_normal() for rng in self._lane_action_rngs], dtype=np.float32)
        return torch.as_tensor(draws, dtype=torch.float32, device=self.device)

    def _on_non_finite_respawn(self, fatal: torch.Tensor) -> None:
        if self.initial_obj_pos is not None:
            new_obj_pos = wp.to_torch(self.cp_data.xpos)[:, self.obj_body_id, :]
            self.initial_obj_pos = torch.where(fatal.unsqueeze(-1), new_obj_pos, self.initial_obj_pos)

    def reset_worlds(self) -> dict[str, torch.Tensor]:
        self._zero_episode_buffers()
        self.initial_world_kps = None
        self.last_artic_ratio = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        target_qpos = wp.to_torch(self.saved_qpos).clone()
        target_qvel = wp.to_torch(self.saved_qvel).clone()
        self._apply_pose_randomization(target_qpos)
        self._reset_worlds_common(target_qpos, target_qvel)
        self.initial_obj_pos = wp.to_torch(self.cp_data.xpos)[:, self.obj_body_id, :].clone()
        return self._get_obs()

    def step_worlds(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        rl_actions = actions.clone().to(dtype=torch.float32, device=self.device)
        ctrl = torch.zeros((self.num_envs, self.m.nu), dtype=torch.float32, device=self.device)
        if self._gpu_demo_traj is not None:
            traj_idx = min(self.current_steps, self.max_episode_steps - 1)
            demo_ctrl = self._gpu_demo_traj[traj_idx].unsqueeze(0).expand(self.num_envs, -1)
            ctrl[:, :NERO_DOF] = torch.clamp(demo_ctrl[:, :NERO_DOF] + rl_actions[:, :NERO_DOF] * self.RESIDUAL_ACTION_SCALE, self.arm_actuator_low, self.arm_actuator_high)
            scaled_grip = rl_actions[:, NERO_DOF] * self.RESIDUAL_ACTION_SCALE * self.gripper_max_range
            ctrl[:, self.gripper_act_id] = torch.clamp(demo_ctrl[:, NERO_DOF] - scaled_grip, 0.0, self.gripper_max_range)
        else:
            current_arm = wp.to_torch(self.cp_data.qpos)[:, self.arm_qpos_addrs]
            ctrl[:, :NERO_DOF] = torch.clamp(current_arm + rl_actions[:, :NERO_DOF] / self.REAL_CONTROL_HZ, self.arm_actuator_low, self.arm_actuator_high)
            open_ratio = (1.0 - torch.clamp(rl_actions[:, NERO_DOF], -1.0, 1.0)) / 2.0
            ctrl[:, self.gripper_act_id] = open_ratio * self.gripper_max_range

        if self._has_gripper_noise:
            ctrl[:, self.gripper_act_id] = torch.clamp(
                ctrl[:, self.gripper_act_id] + self._sample_gripper_noise() * self._grip_std_t,
                0.0,
                self.gripper_max_range,
            )
        return self._finish_step(ctrl)

    # ------------------------------------------------------------------ #
    # Observation, split-part contacts, rewards
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> dict[str, torch.Tensor]:
        xpos = wp.to_torch(self.cp_data.xpos)
        xquat = wp.to_torch(self.cp_data.xquat)
        qpos = wp.to_torch(self.cp_data.qpos)
        obj_xyz, obj_quat = xpos[:, self.obj_body_id, :], xquat[:, self.obj_body_id, :]
        ftip_pos = xpos[:, self.ftip_ids, :].reshape(self.num_envs, -1)
        robot_q = qpos[:, self.robot_qpos_addrs]
        self._last_contacts = self._compute_contacts()
        in_contact = self._last_contacts[0].unsqueeze(1).float()
        steps = torch.full((self.num_envs, 1), self.current_steps / self.max_episode_steps, dtype=torch.float32, device=self.device)
        obj_mass = torch.as_tensor(self.obj_mass_obs[:, :1], dtype=torch.float32, device=self.device)

        local_kps = (self.cached_keypoint_obs / self.KEYPOINT_OBS_SCALE).view(self.num_envs, self.NUM_KEYPOINTS, 3)
        half = self.NUM_KEYPOINTS // 2
        moving_body_id = self.gate_body_id if self.obj_name == "carabiner" else self.handle1_body_id
        moving_kps = self._rotate_points(xquat[:, moving_body_id, :], local_kps[:, :half, :], xpos[:, moving_body_id, :])
        base_kps = self._rotate_points(obj_quat, local_kps[:, half:, :], obj_xyz)
        world_kps = torch.cat([moving_kps, base_kps], dim=1)
        self._last_world_kps = world_kps
        world_kps_flat = world_kps.view(self.num_envs, self.NUM_KEYPOINTS * 3)
        if self.current_steps == 0 or self.initial_world_kps is None:
            self.initial_world_kps = world_kps_flat.clone()

        artic_angle = (
            qpos[:, self.task_joint_adr:self.task_joint_adr + 1]
            if self.task_joint_adr is not None
            else torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        )
        actor_obs = torch.cat([steps, robot_q, ftip_pos, self.initial_world_kps], dim=1)
        privileged = torch.cat([obj_xyz, obj_quat, obj_mass, world_kps_flat, in_contact, artic_angle], dim=1)
        critic_obs = torch.cat([actor_obs, privileged], dim=1)
        return {"actor": actor_obs if self.asymmetric_rl else critic_obs, "critic": critic_obs}

    def _compute_contacts(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._touches_u8.zero_()
        self._touches_a_u8.zero_()
        self._touches_b_u8.zero_()
        self._robot_on_floor_u8.zero_()
        max_contacts = int(self.cp_data.contact.worldid.shape[0])
        if max_contacts > 0:
            with wp.ScopedDevice(self.device):
                stream = (
                    wp.stream_from_torch(torch.cuda.current_stream(self.device))
                    if (not self.use_cpu and torch.device(self.device).type == "cuda")
                    else None
                )
                wp.launch(
                    _tools_contacts_kernel,
                    dim=max_contacts,
                    inputs=[
                        self.cp_data.nacon,
                        self.cp_data.contact.geom,
                        self.cp_data.contact.worldid,
                        self._geom_ftip_col_wp,
                        self._geom_is_obj_wp,
                        self._geom_is_part_a_wp,
                        self._geom_is_part_b_wp,
                        self._geom_is_robot_arm_wp,
                        int(self.floor_geom_id),
                        wp.from_torch(self._touches_u8, dtype=wp.uint8),
                        wp.from_torch(self._touches_a_u8, dtype=wp.uint8),
                        wp.from_torch(self._touches_b_u8, dtype=wp.uint8),
                        wp.from_torch(self._robot_on_floor_u8, dtype=wp.uint8),
                    ],
                    stream=stream,
                )
        touches = self._touches_u8.bool()
        touches_a = self._touches_a_u8.bool()
        touches_b = self._touches_b_u8.bool()
        robot_on_floor = self._robot_on_floor_u8.bool()
        self.ftip_touches_object = touches
        cross_contact = (touches_a[:, 0] & touches_b[:, 1]) | (touches_b[:, 0] & touches_a[:, 1])
        return cross_contact, robot_on_floor

    def _compute_rewards(self, in_contact: torch.Tensor) -> torch.Tensor:
        qpos = wp.to_torch(self.cp_data.qpos)
        obj_height, dist_reward, contact_reward = self._base_distance_and_contact_rewards(in_contact)
        articulation_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        is_valid_lift = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if self.task_joint_adr is not None:
            joint_ratio = torch.clamp(qpos[:, self.task_joint_adr] - self.task_joint_lower_limit, min=0.0) / self.task_joint_limit
            self.last_artic_ratio = joint_ratio.detach()
            is_valid_lift = joint_ratio > self.ARTICULATION_THRESHOLD
            articulation_reward = self.ARTICULATION_MULTIPLIER * joint_ratio * in_contact.float()

        height_reward = self.LIFT_MULTIPLIER * (self.TARGET_HEIGHT - torch.abs(obj_height - self.TARGET_HEIGHT)) * in_contact.float() * is_valid_lift.float()
        rewards = height_reward + contact_reward + dist_reward + articulation_reward
        terminal_reward = torch.zeros_like(rewards)
        if self.current_steps >= self.max_episode_steps:
            terminal_reward = (in_contact & is_valid_lift & (obj_height > self.SUCCESS_LIFT_HEIGHT_M)).float() * self.TERMINAL_SUCCESS_BONUS
            rewards = rewards + terminal_reward

        self.last_contact_reward = contact_reward
        self.last_dist_reward_scaled = dist_reward
        self.last_articulation_reward = articulation_reward
        self.last_f2f_penalty = torch.zeros_like(contact_reward)
        rewards[self.terminated_mask] = 0.0
        self._accumulate_reward_terms({
            "height_reward": height_reward,
            "contact_reward": contact_reward,
            "dist_reward": dist_reward,
            "articulation_reward": articulation_reward,
            "f2f_penalty": -self.last_f2f_penalty,
            "terminal_reward": terminal_reward,
            "total": rewards,
        })
        return rewards

    def _add_debug_visualizations(self, scene: mujoco.MjvScene, env_idx: int) -> None:
        if self.cached_keypoint_obs is None or self._render_d is None:
            return
        local = (self.cached_keypoint_obs[env_idx] / self.KEYPOINT_OBS_SCALE).view(self.NUM_KEYPOINTS, 3).detach().cpu().numpy()
        half = self.NUM_KEYPOINTS // 2
        obj_xpos, obj_xquat = self._render_d.xpos[self.obj_body_id], self._render_d.xquat[self.obj_body_id]
        rot_obj = Rotation.from_quat([obj_xquat[1], obj_xquat[2], obj_xquat[3], obj_xquat[0]])
        moving_body_id = self.gate_body_id if self.obj_name == "carabiner" else self.handle1_body_id
        mov_xpos, mov_xquat = self._render_d.xpos[moving_body_id], self._render_d.xquat[moving_body_id]
        rot_mov = Rotation.from_quat([mov_xquat[1], mov_xquat[2], mov_xquat[3], mov_xquat[0]])
        world = np.concatenate([rot_mov.apply(local[:half]) + mov_xpos, rot_obj.apply(local[half:]) + obj_xpos], axis=0)
        markers = [(p, self.DEBUG_KEYPOINT_RADIUS, self.DEBUG_KEYPOINT_COLOR if i < half else self.DEBUG_KEYPOINT_COLOR_B) for i, p in enumerate(world)]
        markers += [(self._render_d.xpos[f].copy(), self.DEBUG_FTIP_RADIUS, self.DEBUG_FTIP_COLOR) for f in self.ftip_ids]
        self._draw_debug_markers(scene, markers)
