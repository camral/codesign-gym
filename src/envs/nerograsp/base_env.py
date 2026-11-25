"""Shared MuJoCo Warp + CoDesignVecEnv base for NeroGrasp and NeroTools."""

from __future__ import annotations

import concurrent.futures
import hashlib
import random
import xml.etree.ElementTree as ET
from collections import OrderedDict

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from gymnasium.spaces import Box, Dict
from gymnasium.vector import AutoresetMode

from core.codesign_vec_env import CoDesignVecEnv
from core.mujoco_rendering import close_human, render_batch
from core.seeding import parse_seed

from .config import DECOMPOSITION_CACHE_BYTES
from .mesh_switch_env import MeshSwitchMujocoEnv, build_tip_micro_spec, reset_worlds_kernel
from .physics_utils import colorize_parts, contact_bits, rotate_points, world_geom_aabbs
from .shape_deformation.adapter import Fingertip
from .shape_deformation.convex_decomposition import (
    ConvexDecomposer,
    ConvexDecompositionError,
    ConvexPart,
    DecompositionJob,
    content_address,
)
from .shape_deformation.mesh_postprocessing import Postprocessor
from .shape_deformation.quality_metrics import compute_final_mesh_quality_metrics, summarize_quality_records
from .shape_deformation.shape_validation import is_degenerate_mesh, mesh_volume, rejection_reasons, thickness_metrics
from .spaces import Mesh, MeshSpace
from .traj_utils import DEMO_ARM_EE_ABS_X, DEMO_ARM_EE_Y, GRASP_DEMO_ASCEND_FRAC, NERO_DOF, NERO_JOINT_NAMES

EPISODE_STEPS = 100
UNUSED_SLOT_EDGE_M = 1.4e-4
SOLVER_STATE_FIELDS = ("qacc", "qacc_warmstart", "qfrc_applied", "ctrl", "cvel")


def _unused_slot_mesh(tip: Fingertip) -> str:
    return f"{tip.name}_unused_slot"


def _same_meshes(a, b) -> bool:
    if a is None or b is None or len(a) != len(b):
        return False
    return all(
        x.keys() == y.keys() and all(np.array_equal(x[k][0], y[k][0]) and np.array_equal(x[k][1], y[k][1]) for k in x)
        for x, y in zip(a, b)
    )


def _same_embodiments(a, b) -> bool:
    if a is None or b is None or len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if not isinstance(x, dict) or not isinstance(y, dict) or x.keys() != y.keys():
            return False
        for k in x:
            mx, my = x[k], y[k]
            if not isinstance(mx, Mesh) or not isinstance(my, Mesh):
                return False
            if not (np.array_equal(mx.vertices, my.vertices) and np.array_equal(mx.faces, my.faces)):
                return False
    return True


def _design_digest(design) -> bytes | None:
    """Content digest for a single fingertip design dict, or None if not a valid Mesh dict."""
    if not isinstance(design, dict) or not design:
        return None
    h = hashlib.blake2b(digest_size=16)
    for key in sorted(design):
        mesh = design[key]
        if not isinstance(mesh, Mesh):
            return None
        v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        h.update(str(key).encode("utf-8"))
        h.update(b"\0")
        h.update(str(v.dtype).encode("ascii"))
        h.update(np.asarray(v.shape, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(v).tobytes())
        h.update(str(f.dtype).encode("ascii"))
        h.update(np.asarray(f.shape, dtype=np.int64).tobytes())
        h.update(np.ascontiguousarray(f).tobytes())
    return h.digest()


class NeroBaseEnv(CoDesignVecEnv, MeshSwitchMujocoEnv):
    """Unified MuJoCo Warp + CoDesignVecEnv base for NeroGrasp and NeroTools."""

    metadata = {
        "supports_transition_embodiment": False,
        "autoreset_mode": AutoresetMode.DISABLED,
        "render_modes": ["human", "rgb_array"],
    }

    PENALTY_FORCE_THRESHOLD = 50.0        # finger-object force (N) over dt_ctrl
    PENALTY_AABB_MARGIN = 0.02            # allowed AABB overlap (m) before tunneling
    SAFEGUARD_STEPS = 2                   # steps force/tunneling must persist
    SAFEGUARD_IN_AIR_STEPS = 5            # 0.5 s persistence for object-in-air without contact
    SAFEGUARD_IN_AIR_THRESHOLD_M = 0.015
    DIST_REWARD_SCALE = 0.003
    DIST_REWARD_SIGMA = 0.05
    CONTACT_BONUS = 0.02
    RESIDUAL_ACTION_SCALE = 0.1
    DIST_REWARD_CUTOFF_FRAC = GRASP_DEMO_ASCEND_FRAC
    MAX_OBJ_DIST_FOR_TERMINATION = 1.0
    SUCCESS_LIFT_HEIGHT_M = 0.03
    MASS_OBS_SCALE = 10.0
    KEYPOINT_OBS_SCALE = 50.0
    NUM_KEYPOINTS = 12
    DEBUG_KEYPOINT_RADIUS = 0.002
    DEBUG_KEYPOINT_COLOR = [1.0, 0.0, 1.0, 1.0]
    DEBUG_KEYPOINT_COLOR_B = [0.1, 0.9, 0.9, 1.0]
    DEBUG_FTIP_RADIUS = 0.003
    DEBUG_FTIP_COLOR = [0.0, 1.0, 0.0, 0.8]
    EXTRA_TIP_BODIES: tuple[str, ...] = ()

    # ------------------------------------------------------------------ #
    # Shared init helpers
    # ------------------------------------------------------------------ #
    def _init_fingertip_pipeline(
        self,
        *,
        fingertips: list[Fingertip],
        mounting_vertex_count: int,
        max_bbox_limits_mm,
        scene_root: ET.Element,
        workers: int,
    ) -> None:
        self.fingertips = fingertips
        if any(not np.array_equal(np.sort(tip.non_tip_indices), np.arange(mounting_vertex_count)) for tip in fingertips):
            raise RuntimeError(f"Canonical mounting vertices must occupy indices 0-{mounting_vertex_count - 1}")
        self.base_tip_volumes = {tip.name: mesh_volume(tip.vertices, tip.faces) for tip in fingertips}
        min_bp = self.config.rejectors.min_backplate_area_ratio
        self.postprocessor = Postprocessor(
            snap_plane_clamp=self.config.snap_plane_clamp,
            min_backplate_area_ratio=0.0 if min_bp is None else min_bp,
            max_bbox_limits_mm=max_bbox_limits_mm,
        )
        self.max_parts = self.config.convex_max_parts
        options = self.config.decomposition_options()
        if self.config.decomposition_backend == "cuacd" and not self.use_cpu:
            dev = torch.device(self.device)
            if dev.type == "cuda":
                options.setdefault("device_id", dev.index or 0)
        self.decomposer = ConvexDecomposer(
            backend=self.config.decomposition_backend,
            max_parts=self.max_parts,
            options=options,
            workers=workers,
            cache_max_bytes=DECOMPOSITION_CACHE_BYTES,
            verbose=self.verbose,
        )

        self._base_spec = mujoco.MjSpec.from_string(ET.tostring(scene_root, encoding="unicode"))
        self._tip_by_name = {tip.name: tip for tip in fingertips}
        self._tip_micro_specs: dict[str, mujoco.MjSpec] = {}
        for tip in fingertips:
            centre = tip.vertices[tip.non_tip_indices].mean(axis=0)
            corners = centre + UNUSED_SLOT_EDGE_M / (2 * np.sqrt(2)) * np.array(
                [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64
            )
            self._base_spec.add_mesh(
                name=_unused_slot_mesh(tip),
                uservert=corners.reshape(-1).tolist(),
                userface=[0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2],
                inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX,
            )
            self._tip_micro_specs[tip.name] = build_tip_micro_spec(
                scene_root, tip.name, _unused_slot_mesh(tip), corners, mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX
            )

        self.ep_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.current_steps = 0
        self.terminated_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cached_keypoint_obs = None

    def _init_codesign(
        self,
        *,
        num_envs: int,
        is_eval_env: bool,
        mounting_vertex_count: int,
        min_bbox_limits_mm,
        tip_keys: dict[str, str],
        quality_columns: tuple[str, ...],
        env_label: str,
    ) -> None:
        self._min_bbox_limits_mm = min_bbox_limits_mm
        self._quality_columns = quality_columns
        self._env_label = env_label
        self.num_unique_objects = len(self.obj_names)
        self.num_embodiments = num_envs // self.num_unique_objects
        self._mesh_keys = {tip.name: tip_keys[tip.name] for tip in self.fingertips}
        space = MeshSpace(
            {self._mesh_keys[t.name]: Mesh(t.vertices.copy(), t.faces.copy()) for t in self.fingertips},
            {self._mesh_keys[t.name]: mounting_vertex_count for t in self.fingertips},
            sample_perturbation_m=self.config.sample_perturbation_m,
        )
        space.batch_size = self.num_embodiments
        CoDesignVecEnv.__init__(
            self,
            num_envs,
            Dict({
                "actor": Box(-np.inf, np.inf, shape=(self.rl_obs_dim,)),
                "critic": Box(-np.inf, np.inf, shape=(self.rl_critic_obs_dim,)),
            }),
            Box(-1.0, 1.0, shape=(self.rl_act_dim,)),
            space,
            [space.base_meshes for _ in range(self.num_embodiments)],
            is_eval_env=is_eval_env,
        )
        self.base_embodiment_is_batched = True
        self.current_embodiment = [space.base_meshes for _ in range(self.num_embodiments)]
        self.embodiment_metadata = {
            "encoding": "full_triangle_meshes",
            "units": "m",
            "fingertips": {
                tip.name: {
                    "key": self._mesh_keys[tip.name],
                    "mounting_vertex_count": mounting_vertex_count,
                    "deformable_vertex_indices": tip.deformable_vertex_indices.copy(),
                    "canonical_vertices": tip.vertices.copy(),
                }
                for tip in self.fingertips
            },
        }
        self.last_quality_metrics: list[dict] = []
        self.last_quality_summary: dict[str, float] = {}
        self._cached_quality_records = None
        self._cached_quality_info: dict[str, torch.Tensor] | None = None
        self._cached_emb_info_ref = None
        self._cached_emb_info_val = None
        self._embodiment_id_t = torch.repeat_interleave(
            torch.arange(self.num_embodiments, dtype=torch.long, device=self.device),
            self.num_unique_objects,
        )
        self._design_assess_cache: OrderedDict[bytes, tuple[bool, str, list[dict], list[dict], int]] = OrderedDict()
        self._design_assess_cache_max = max(256, 2 * self.num_embodiments)
        self.last_held_at_goal = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.last_diverged = torch.zeros_like(self.last_held_at_goal)
        self._finished = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.force_safeguard_counts = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.tunneling_safeguard_counts = torch.zeros_like(self.force_safeguard_counts)
        self.in_air_safeguard_counts = torch.zeros_like(self.force_safeguard_counts)

    def _init_noise_multiplier(self, has_noise: bool, *, per_object: bool = False) -> None:
        self.noise_multiplier_t = torch.ones((self.num_envs, 1), dtype=torch.float32, device=self.device)
        if not has_noise:
            self.noise_multiplier_t[:] = 0.0
            return
        groups: dict[str, list[int]] = {}
        if per_object:
            for i, name in enumerate(self.obj_names_env):
                groups.setdefault(name, []).append(i)
        else:
            groups["all"] = list(range(self.num_envs))
        for indices in groups.values():
            count = len(indices)
            if count <= 2:
                continue
            num_zero = count // 4 if count >= 4 else count // 2
            num_low = count // 4 if count >= 4 else count // 3
            if num_zero:
                self.noise_multiplier_t[indices[:num_zero]] = 0.0
            if num_low:
                self.noise_multiplier_t[indices[num_zero:num_zero + num_low]] = 0.25

    @property
    def codesign_config(self):
        return self.config

    @property
    def _render_human_automatically(self) -> bool:
        return not self._reset_failed

    # ------------------------------------------------------------------ #
    # Shared model build & post-build
    # ------------------------------------------------------------------ #
    def _place_variant_object(self, spec: mujoco.MjSpec, object_name: str | None) -> None:
        """Subclass hook to place per-world object geometry into ``spec``."""

    def _variant_spec(
        self,
        parts_by_tip: dict[str, list[ConvexPart]],
        object_name: str | None = None,
        extra_parts=(),
        collide_unused: bool = False,
    ) -> mujoco.MjSpec:
        spec = self._base_spec.copy()
        added: set[str] = set()
        for part in [*extra_parts, *(p for tip in self.fingertips for p in parts_by_tip[tip.name])]:
            if part.name not in added:
                spec.add_mesh(
                    name=part.name,
                    uservert=part.vertices.reshape(-1).tolist(),
                    userface=part.faces.reshape(-1).tolist(),
                    inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX,
                )
                added.add(part.name)
        for tip in self.fingertips:
            parts = parts_by_tip[tip.name]
            for j in range(self.max_parts):
                geom = spec.geom(f"{tip.name}_part_{j:02d}")
                if j < len(parts):
                    geom.meshname = parts[j].name
                    continue
                geom.meshname = _unused_slot_mesh(tip)
                geom.group = 4
                geom.mass = 0.0
                geom.rgba = (0.0, 0.0, 0.0, 0.0)
                if not collide_unused:
                    geom.contype = 0
                    geom.conaffinity = 0
        self._place_variant_object(spec, object_name)
        return spec

    def _tip_micro_variant_spec(self, tip_name: str, parts: list[ConvexPart]) -> mujoco.MjSpec:
        spec = self._tip_micro_specs[tip_name].copy()
        tip = self._tip_by_name[tip_name]
        added: set[str] = set()
        for part in parts:
            if part.name not in added:
                spec.add_mesh(
                    name=part.name,
                    uservert=part.vertices.reshape(-1).tolist(),
                    userface=part.faces.reshape(-1).tolist(),
                    inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX,
                )
                added.add(part.name)
        for j in range(self.max_parts):
            geom = spec.geom(f"{tip_name}_part_{j:02d}")
            if j < len(parts):
                geom.meshname = parts[j].name
                continue
            geom.meshname = _unused_slot_mesh(tip)
            geom.group = 4
            geom.mass = 0.0
            geom.rgba = (0.0, 0.0, 0.0, 0.0)
            geom.contype = 0
            geom.conaffinity = 0
        return spec

    def build_model(self, tip_meshes: list[dict[str, tuple[np.ndarray, np.ndarray]]]) -> None:
        jobs = [DecompositionJob((w, tip.name), *meshes[tip.name], tip.mesh_name) for w, meshes in enumerate(tip_meshes) for tip in self.fingertips]
        parts = self.decomposer.decompose(jobs)
        world_parts, total, unique = content_address([{tip.name: parts[(w, tip.name)] for tip in self.fingertips} for w in range(self.num_envs)])
        if self.verbose:
            print(f"[ASSETS] Convex parts: {unique}/{total} unique.")
        keys = [
            (tuple(tuple(p.name for p in parts_by_tip[tip.name]) for tip in self.fingertips), self.obj_names_env[w])
            for w, parts_by_tip in enumerate(world_parts)
        ]
        by_key = {key: world_parts[w] for w, key in reversed(list(enumerate(keys)))}
        all_parts = [p for parts_by_tip in world_parts for tip in self.fingertips for p in parts_by_tip[tip.name]]
        global_spec = self._variant_spec(world_parts[0], self.obj_names_env[0], extra_parts=all_parts, collide_unused=True)
        unique_objs = list(dict.fromkeys(self.obj_names_env))
        if not self.retain_all_render_models and len(by_key) > len(unique_objs):
            base_specs = {obj: self._variant_spec(world_parts[0], obj) for obj in unique_objs}
            tip_names = [tip.name for tip in self.fingertips]
            tip_parts_by_key: dict[tuple[str, tuple[str, ...]], list[ConvexPart]] = {}
            world_tip_keys = []
            for parts_by_tip in world_parts:
                tmap = {}
                for tname in tip_names:
                    tkey = tuple(p.name for p in parts_by_tip[tname])
                    tmap[tname] = tkey
                    tip_parts_by_key.setdefault((tname, tkey), parts_by_tip[tname])
                world_tip_keys.append(tmap)
            self.compile_worlds_fast(
                global_spec,
                base_specs,
                self.obj_names_env,
                lambda tname, tkey: self._tip_micro_variant_spec(tname, tip_parts_by_key[(tname, tkey)]),
                world_tip_keys,
                tip_names,
                self.max_parts,
            )
        else:
            self.compile_worlds(global_spec, lambda key: self._variant_spec(by_key[key], key[1]), keys)
        self.tip_meshes = tip_meshes
        self._post_build()

    def _init_common_model_indices(self, m: mujoco.MjModel, obj_body_id: int, observed_ftips: tuple[str, ...], *, obj_colliding_only: bool) -> None:
        self.obj_body_id = obj_body_id
        self.ftip_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) for name in observed_ftips]
        assert all(i != -1 for i in self.ftip_ids), f"Missing fingertip bodies: {observed_ftips}"

        obj_bodies = {obj_body_id}
        grown = True
        while grown:
            grown = False
            for b in range(m.nbody):
                if b not in obj_bodies and int(m.body_parentid[b]) in obj_bodies:
                    obj_bodies.add(b)
                    grown = True
        self.obj_body_set = obj_bodies

        ftip_bodies = set(self.ftip_ids)
        colliding = (m.geom_contype > 0) | (m.geom_conaffinity > 0)
        self.ftip_geom_ids = {g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in ftip_bodies and colliding[g]}
        self.floor_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.all_obj_geom_ids = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in obj_bodies and (not obj_colliding_only or colliding[g])]
        self.robot_geom_ids = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) not in obj_bodies and g != self.floor_geom_id and colliding[g]]

        is_ftip = np.zeros(m.ngeom, dtype=bool)
        is_ftip[list(self.ftip_geom_ids)] = True
        is_obj = np.zeros(m.ngeom, dtype=bool)
        is_obj[self.all_obj_geom_ids] = True
        ftip_of_body = np.full(m.nbody, -1, dtype=np.int64)
        ftip_of_body[self.ftip_ids] = np.arange(len(self.ftip_ids))
        ftip_col = np.where(is_ftip, ftip_of_body[m.geom_bodyid], -1)
        is_robot_arm = (~is_ftip) & (~is_obj)
        if self.floor_geom_id != -1:
            is_robot_arm[self.floor_geom_id] = False
        self._geom_ftip_col_t = torch.as_tensor(ftip_col, dtype=torch.long, device=self.device)
        self._geom_is_obj_t = torch.as_tensor(is_obj, dtype=torch.bool, device=self.device)
        self._geom_is_robot_arm_t = torch.as_tensor(is_robot_arm, dtype=torch.bool, device=self.device)

        self._sg_obj = list(self.all_obj_geom_ids)
        self._sg_ftip = sorted(self.ftip_geom_ids)
        self._sg_floor = [g for g in self.robot_geom_ids if g not in self.ftip_geom_ids]
        self._sg_geoms = torch.tensor(self._sg_obj + self._sg_ftip + self._sg_floor, dtype=torch.long, device=self.device)

        self.arm_qpos_addrs, self.arm_qvel_addrs = [], []
        for name in NERO_JOINT_NAMES[:NERO_DOF]:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if j != -1:
                self.arm_qpos_addrs.append(int(m.jnt_qposadr[j]))
                self.arm_qvel_addrs.append(int(m.jnt_dofadr[j]))

    def _finalize_post_build(self, saved_qpos: np.ndarray, saved_qvel: np.ndarray, body_mass: np.ndarray) -> None:
        self.obj_mass_obs = (body_mass[:, self.obj_body_id:self.obj_body_id + 1] * self.MASS_OBS_SCALE).astype(np.float32)
        self.obj_masses_env = torch.as_tensor(body_mass[:, self.obj_body_id], dtype=torch.float32, device=self.device)
        with wp.ScopedDevice(self.device):
            self.saved_qpos = wp.array(saved_qpos.astype(np.float32), dtype=wp.float32)
            self.saved_qvel = wp.array(saved_qvel, dtype=wp.float32)
            self._full_mask_wp = wp.array(np.ones(self.num_envs, dtype=np.uint8), dtype=wp.uint8)
        self._respawn(self.saved_qpos, self.saved_qvel)
        self._capture_step_graph()
        if self.step_graph is not None:
            self._respawn(self.saved_qpos, self.saved_qvel)

    def _configure_compiled(self, model: mujoco.MjModel) -> None:
        if self.config.fidelity == 0:
            model.opt.timestep, model.opt.iterations, model.opt.ls_iterations = 0.01, 2, 4
        else:
            model.opt.timestep, model.opt.iterations, model.opt.ls_iterations = 0.005, 15, 24
        contact_bits(model, extra_tip_bodies=self.EXTRA_TIP_BODIES)
        colorize_parts(model)

    def _setup_camera(self, camera: mujoco.MjvCamera) -> None:
        camera.lookat[:] = [-DEMO_ARM_EE_ABS_X, DEMO_ARM_EE_Y, 0.07]
        camera.distance = 0.5
        camera.elevation = -25
        camera.azimuth = 80

    # ------------------------------------------------------------------ #
    # Shared physics, safeguards, rewards
    # ------------------------------------------------------------------ #
    def _respawn(self, qpos: wp.array, qvel: wp.array, mask: wp.array | None = None) -> None:
        with wp.ScopedDevice(self.device):
            wp.launch(
                reset_worlds_kernel,
                dim=self.num_envs,
                inputs=[self.cp_data.qpos, self.cp_data.qvel, qpos, qvel, self._full_mask_wp if mask is None else mask],
            )
            mjw.forward(self.cp_model, self.cp_data)
            wp.synchronize()

    def _reset_worlds_common(self, target_qpos: torch.Tensor, target_qvel: torch.Tensor) -> None:
        with wp.ScopedDevice(self.device):
            wp.launch(
                reset_worlds_kernel,
                dim=self.num_envs,
                inputs=[
                    self.cp_data.qpos,
                    self.cp_data.qvel,
                    wp.from_torch(target_qpos, dtype=wp.float32),
                    wp.from_torch(target_qvel, dtype=wp.float32),
                    self._full_mask_wp,
                ],
            )
            for name in SOLVER_STATE_FIELDS:
                getattr(self.cp_data, name).zero_()
            mjw.forward(self.cp_model, self.cp_data)
            wp.synchronize()

    def _zero_episode_buffers(self) -> None:
        self.ep_returns.fill_(0.0)
        self.current_steps = 0
        self.terminated_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.force_safeguard_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.tunneling_safeguard_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.in_air_safeguard_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_obj_vel = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

    def _finish_step(self, ctrl: torch.Tensor, *, obj_active_mask: torch.Tensor | None = None) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        self._physics_step(ctrl)
        self._terminate_non_finite()
        self.current_steps += 1
        obs = self._get_obs()
        in_contact, robot_touches_floor = self._last_contacts
        self._check_safeguards(in_contact, robot_touches_floor, obj_active_mask=obj_active_mask)
        rewards = self._compute_rewards(in_contact)
        rewards[self.terminated_mask] = 0.0
        self.ep_returns += rewards
        dones = self.terminated_mask.clone()
        if self.current_steps >= self.max_episode_steps:
            dones[:] = True
        return obs, rewards, dones

    def _on_non_finite_respawn(self, fatal: torch.Tensor) -> None:
        pass

    def _terminate_non_finite(self) -> None:
        qpos, qvel = wp.to_torch(self.cp_data.qpos), wp.to_torch(self.cp_data.qvel)
        fatal = (~torch.isfinite(qpos)).any(dim=1) | (~torch.isfinite(qvel)).any(dim=1) | (qvel.abs() > 200.0).any(dim=1)
        if not torch.any(fatal):
            return
        self.terminated_mask |= fatal
        with wp.ScopedDevice(self.device):
            wp.launch(
                reset_worlds_kernel,
                dim=self.num_envs,
                inputs=[self.cp_data.qpos, self.cp_data.qvel, self.saved_qpos, self.saved_qvel, wp.from_torch(fatal.to(torch.uint8))],
            )
            for name in SOLVER_STATE_FIELDS:
                getattr(self.cp_data, name).zero_()
            mjw.forward(self.cp_model, self.cp_data)
            wp.synchronize()
        self._on_non_finite_respawn(fatal)

    @staticmethod
    def _rotate_points(quat_wxyz: torch.Tensor, local_pts: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        return rotate_points(quat_wxyz, local_pts, offset)

    def _floor_contact_mask(self, geom_a: torch.Tensor, geom_b: torch.Tensor, worlds: torch.Tensor) -> torch.Tensor:
        robot_on_floor = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.floor_geom_id != -1:
            on_floor = ((geom_a == self.floor_geom_id) & self._geom_is_robot_arm_t[geom_b]) | (
                (geom_b == self.floor_geom_id) & self._geom_is_robot_arm_t[geom_a]
            )
            if on_floor.any():
                robot_on_floor[worlds[on_floor]] = True
        return robot_on_floor

    def _base_distance_and_contact_rewards(self, in_contact: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xpos = wp.to_torch(self.cp_data.xpos)
        obj_height = xpos[:, self.obj_body_id, 2]
        ftip_pos = xpos[:, self.ftip_ids, :].view(self.num_envs, len(self.ftip_ids), 3)
        dists = torch.linalg.norm(ftip_pos.unsqueeze(2) - self._last_world_kps.unsqueeze(1), dim=-1).min(dim=2).values.clone()
        dists[self.ftip_touches_object] = 0.0
        max_dist = dists.max(dim=1).values
        self.terminated_mask |= max_dist > self.MAX_OBJ_DIST_FOR_TERMINATION
        dist_reward = self.DIST_REWARD_SCALE / (1.0 + (max_dist / self.DIST_REWARD_SIGMA) ** 2)
        if self.DIST_REWARD_CUTOFF_FRAC > 0.0 and self.current_steps >= int(self.max_episode_steps * (1 - self.DIST_REWARD_CUTOFF_FRAC)):
            dist_reward = torch.zeros_like(dist_reward)
        contact_reward = in_contact.float() * self.CONTACT_BONUS
        return obj_height, dist_reward, contact_reward

    def _check_safeguards(
        self,
        in_contact: torch.Tensor,
        robot_touches_floor: torch.Tensor,
        *,
        obj_active_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 1. Sustained excessive contact force on object
        obj_vel = wp.to_torch(self.cp_data.cvel)[:, self.obj_body_id, 3:6]
        dt_ctrl = self.control_substeps * self.m.opt.timestep
        force = self.obj_masses_env * torch.linalg.norm((obj_vel - self.prev_obj_vel) / dt_ctrl, dim=1)
        self.prev_obj_vel = obj_vel.clone()
        force_exceeded = (force > self.PENALTY_FORCE_THRESHOLD) & in_contact
        self.force_safeguard_counts[force_exceeded] += 1
        self.force_safeguard_counts[~force_exceeded] = 0
        high_force = self.force_safeguard_counts >= self.SAFEGUARD_STEPS

        # 2. Fingertip AABB tunneling deep into object
        geom_xpos = wp.to_torch(self.cp_data.geom_xpos)
        lo_all, hi_all = world_geom_aabbs(
            geom_xpos, wp.to_torch(self.cp_data.geom_xmat), wp.to_torch(self.cp_model.geom_aabb), self._sg_geoms
        )
        n_obj, n_ftip = len(self._sg_obj), len(self._sg_ftip)
        if n_obj:
            if obj_active_mask is not None:
                active = obj_active_mask.unsqueeze(-1)
                o_min = torch.where(active, lo_all[:, :n_obj], float("inf")).min(dim=1).values
                o_max = torch.where(active, hi_all[:, :n_obj], -float("inf")).max(dim=1).values
            else:
                o_min = lo_all[:, :n_obj].min(dim=1).values
                o_max = hi_all[:, :n_obj].max(dim=1).values
        else:
            o_min = o_max = torch.zeros((self.num_envs, 3), device=self.device)
        lo, hi = lo_all[:, n_obj:n_obj + n_ftip], hi_all[:, n_obj:n_obj + n_ftip]
        overlap = (
            torch.clamp(torch.minimum(hi, o_max.unsqueeze(1)) - torch.maximum(lo, o_min.unsqueeze(1)), min=0.0)
            > self.PENALTY_AABB_MARGIN
        ).all(dim=-1)
        tunneling = overlap.any(dim=1)
        self.tunneling_safeguard_counts[tunneling] += 1
        self.tunneling_safeguard_counts[~tunneling] = 0
        sustained_tunneling = self.tunneling_safeguard_counts >= self.SAFEGUARD_STEPS

        # 3. Robot arm hitting or penetrating floor
        floor_z = geom_xpos[:, self.floor_geom_id, 2] if self.floor_geom_id != -1 else torch.zeros(self.num_envs, device=self.device)
        below = lo_all[:, n_obj + n_ftip:, 2] < (floor_z - 0.02).unsqueeze(1)
        floor_safeguard = robot_touches_floor | below.any(dim=1)

        # 4. Object floating in air without contact
        in_air = ((o_min[:, 2] - floor_z) > self.SAFEGUARD_IN_AIR_THRESHOLD_M) & (~in_contact)
        self.in_air_safeguard_counts[in_air] += 1
        self.in_air_safeguard_counts[~in_air] = 0
        in_air_safeguard = self.in_air_safeguard_counts >= self.SAFEGUARD_IN_AIR_STEPS

        flags = high_force | sustained_tunneling | floor_safeguard | in_air_safeguard
        self.terminated_mask |= flags
        return flags

    def _accumulate_reward_terms(self, terms: dict[str, torch.Tensor]) -> None:
        if self.current_steps <= 1 or not hasattr(self, "ep_reward_term_sums"):
            self.ep_reward_term_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in terms}
        alive = (~self.terminated_mask).float()
        for k, v in terms.items():
            self.ep_reward_term_sums[k] += v.detach() * alive

    def get_episode_reward_term_means(self) -> dict[str, float]:
        if not hasattr(self, "ep_reward_term_sums"):
            return {}
        objects = np.asarray(self.obj_names_env)
        out = {f"all_reward/{term}": sums.mean().item() for term, sums in self.ep_reward_term_sums.items()}
        for obj in sorted(set(self.obj_names_env)):
            mask = torch.as_tensor(objects == obj, dtype=torch.bool, device=self.device)
            for term, sums in self.ep_reward_term_sums.items():
                out[f"{obj}_reward/{term}"] = sums[mask].mean().item()
        return out

    @staticmethod
    def _draw_debug_markers(scene: mujoco.MjvScene, markers: list[tuple[np.ndarray, float, list[float]]]) -> None:
        for pos, radius, color in markers:
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius] * 3, dtype=np.float32),
                pos,
                np.eye(3, dtype=np.float32).flatten(),
                np.array(color, dtype=np.float32),
            )
            scene.ngeom += 1

    # ------------------------------------------------------------------ #
    # CoDesignVecEnv hooks
    # ------------------------------------------------------------------ #
    def _expand_embodiments_to_envs(self, embodiment: list) -> list:
        return [design for design in embodiment for _ in range(self.num_unique_objects)]

    def _prepare_direct_meshes(
        self,
        embodiment: list[dict[str, Mesh]],
        first_row: int = 0,
        *,
        skip_unchanged_validation: bool = False,
        update_state: bool = True,
    ):
        rejectors = self.config.rejectors
        meshes = [{} for _ in embodiment]
        records, messages = [], []
        for tip in self.fingertips:
            key = self._mesh_keys[tip.name]
            metrics, final = [], []
            for row, design in enumerate(embodiment):
                candidate = design[key]
                cand_v = np.asarray(candidate.vertices, dtype=np.float64)
                vertices, faces = self.postprocessor.postprocess(tip, cand_v, candidate.faces)
                mesh = (np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64))
                if not (
                    skip_unchanged_validation
                    and np.array_equal(mesh[0], cand_v)
                    and np.array_equal(mesh[1], candidate.faces)
                ):
                    self.embodiment_space._validate_mesh(key, Mesh(*mesh))
                meshes[row][tip.name] = mesh
                final.append(mesh)
                metrics.append({
                    "name": tip.name,
                    "volume": mesh_volume(*mesh),
                    "bbox_min_mm": mesh[0].min(axis=0) * 1000.0,
                    "bbox_max_mm": mesh[0].max(axis=0) * 1000.0,
                })
                if rejectors.reject_degenerate_meshes and is_degenerate_mesh(*mesh):
                    messages.append(f"embodiment row {first_row + row}, fingertip {tip.name}: degenerate mesh")
            if rejectors.min_thickness_mm is not None:
                for metric, mesh in zip(metrics, final):
                    metric.update(thickness_metrics(*mesh))
            reasons = rejection_reasons(
                metrics, tip,
                base_volume=self.base_tip_volumes[tip.name],
                min_volume_ratio=0.0 if rejectors.min_volume_ratio is None else rejectors.min_volume_ratio,
                min_bbox_limits_mm_=self._min_bbox_limits_mm if rejectors.enforce_minimum_bbox else None,
                snap_plane_clamp=self.config.snap_plane_clamp,
                min_thickness_mm=rejectors.min_thickness_mm,
            )
            for row, row_reasons in enumerate(reasons):
                messages.extend(f"embodiment row {first_row + row}, fingertip {tip.name}: {reason}" for reason in row_reasons)
            for row, mesh in enumerate(final):
                quality = compute_final_mesh_quality_metrics(*mesh)
                records.extend({"env_id": row * self.num_unique_objects + k, "effector": tip.name, **quality} for k in range(self.num_unique_objects))
        if messages:
            raise ValueError("Invalid embodiment:\n" + "\n".join(messages))
        expanded = self._expand_embodiments_to_envs(meshes)
        if not update_state:
            return expanded, records
        self.last_quality_metrics = records
        self.last_quality_summary = summarize_quality_records(records)
        return expanded

    def _assess_one_design(self, design, row: int) -> tuple[bool, str, list[dict], list[dict], int]:
        if not self.embodiment_space.contains(design):
            reason = "; ".join(self.embodiment_space.rejection_reasons(design))
            return True, reason, [], [], row
        if self.config.snap_plane_clamp and any(
            len(design[key].vertices) != len(base.vertices) or not np.array_equal(design[key].faces, base.faces)
            for key, base in self.embodiment_space.base_meshes.items()
        ):
            return True, "Topology changes are not supported with snap_plane_clamp enabled", [], [], row
        try:
            row_meshes, row_records = self._prepare_direct_meshes(
                [design], first_row=row, skip_unchanged_validation=True, update_state=False
            )
        except ValueError as error:
            return True, str(error), [], [], row
        return False, "", row_meshes, row_records, row

    def _validate_embodiment_batch(self, embodiment, check_bounds: bool = True) -> list:
        if not isinstance(embodiment, list) or len(embodiment) != self.num_embodiments:
            raise ValueError(f"embodiment must be a list of {self.num_embodiments} design dictionaries")
        if check_bounds and not all(self.embodiment_space.contains(design) for design in embodiment):
            raise ValueError("embodiment contains designs outside embodiment_space")
        return embodiment

    def _assess_embodiment(self, embodiment):
        values = self._validate_embodiment_batch(embodiment, check_bounds=False)
        rejected = np.zeros(self.num_envs, dtype=bool)
        reasons = [""] * self.num_envs
        meshes = [{} for _ in range(self.num_envs)]
        records = []
        cache = self._design_assess_cache
        cache_max = self._design_assess_cache_max

        # Deduplicate uncached designs first so unique candidates can be assessed in parallel.
        # Keep all results for this batch in a local dict so LRU evictions cannot drop entries
        # needed by the current batch when unique designs exceed cache_max.
        row_digests: list[bytes | None] = []
        batch_results: dict[bytes, tuple[bool, str, list[dict], list[dict], int]] = {}
        unique_misses: dict[bytes, tuple[int, object]] = {}
        unhashable_rows: list[tuple[int, object]] = []
        for row, design in enumerate(values):
            digest = _design_digest(design)
            row_digests.append(digest)
            if digest is None:
                unhashable_rows.append((row, design))
            elif digest in cache:
                if digest not in batch_results:
                    batch_results[digest] = cache[digest]
                    cache.move_to_end(digest)
            elif digest not in unique_misses:
                unique_misses[digest] = (row, design)

        if unique_misses:
            # Warm postprocessor invariants on the main thread before parallel assessment.
            for tip in self.fingertips:
                self.postprocessor._get_invariants(tip)
            miss_items = list(unique_misses.items())
            workers = min(getattr(self, "compile_workers", 1), len(miss_items))
            use_threads = workers > 1 and len(miss_items) >= 4 and self.config.rejectors.min_thickness_mm is not None
            if use_threads:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    assessed_list = list(pool.map(lambda it: self._assess_one_design(it[1][1], it[1][0]), miss_items))
            else:
                assessed_list = [self._assess_one_design(design, row) for _, (row, design) in miss_items]
            for (digest, _), entry in zip(miss_items, assessed_list):
                batch_results[digest] = entry
                cache[digest] = entry
                if len(cache) > cache_max:
                    cache.popitem(last=False)

        unhashable_results = {row: self._assess_one_design(design, row) for row, design in unhashable_rows}

        for row, digest in enumerate(row_digests):
            start, stop = row * self.num_unique_objects, (row + 1) * self.num_unique_objects
            if digest is not None:
                entry = batch_results[digest]
            else:
                entry = unhashable_results[row]
            is_rej, reason, row_meshes, row_records, cached_row = entry
            if is_rej:
                rejected[start:stop] = True
                if cached_row != row and f"embodiment row {cached_row}" in reason:
                    reason = reason.replace(f"embodiment row {cached_row}", f"embodiment row {row}")
                reasons[start:stop] = [reason] * self.num_unique_objects
            else:
                meshes[start:stop] = row_meshes
                records.extend({**record, "env_id": int(record["env_id"]) + start} for record in row_records)
        return values, meshes, rejected, records, tuple(reasons)

    def _fill_base(self, embodiment):
        if isinstance(embodiment, list) and any(design is None for design in embodiment):
            return [self.base_embodiment[i] if design is None else design for i, design in enumerate(embodiment)]
        return embodiment

    def _validate_embodiment(self, embodiment):
        try:
            return ~self._assess_embodiment(self._fill_base(embodiment))[2]
        except (ValueError, TypeError, OverflowError):
            return np.zeros(self.num_envs, dtype=bool)

    def _seed_extra(self, parsed) -> None:
        pass

    def _reset(self, *, seed=None, options=None, embodiment=None):
        if options and "reset_mask" in options and not np.asarray(options["reset_mask"], dtype=bool).all():
            raise NotImplementedError(f"{self._env_label} worlds share one episode clock; only a full reset is possible")
        previously_valid = not self._reset_failed and not self._needs_rebuild
        self._reset_failed = True
        parsed = parse_seed(seed, self.num_envs)
        if isinstance(parsed, int):
            self._rng, self._lane_rngs = random.Random(parsed), None
        elif parsed is not None:
            self._lane_rngs = [random.Random(s) for s in parsed]
        self._seed_extra(parsed)

        records, reasons = None, ("",) * self.num_envs
        previous_embodiment, previous_meshes = self.current_embodiment, self.tip_meshes
        previous_needs_rebuild = self._needs_rebuild
        assessed_values = None
        meshes = previous_meshes
        if embodiment is not None:
            filled = self._fill_base(embodiment)
            if not (previously_valid and _same_embodiments(filled, self.current_embodiment)):
                assessed_values, meshes, rejected, records, reasons = self._assess_embodiment(filled)
                if rejected.any():
                    return self._get_obs(), torch.as_tensor(rejected, dtype=torch.bool, device=self.device), self._reset_info(reasons)
                if not _same_embodiments(assessed_values, self.current_embodiment):
                    self.current_embodiment = [{name: mesh.copy() for name, mesh in design.items()} for design in assessed_values]
        rebuild = self._needs_rebuild or not _same_meshes(meshes, self.tip_meshes)
        try:
            if rebuild:
                self._needs_rebuild = True
                self.build_model(meshes)
                self._needs_rebuild = False
            observation = self.reset_worlds()
        except ConvexDecompositionError as error:
            self.current_embodiment = previous_embodiment
            self.tip_meshes, self._needs_rebuild = previous_meshes, previous_needs_rebuild
            if assessed_values is None:
                raise
            rejected = np.zeros(self.num_envs, dtype=bool)
            decomp_reasons = [""] * self.num_envs
            failed_rows: dict[int, list[str]] = {}
            for job_id in error.failed_job_ids:
                if isinstance(job_id, tuple) and len(job_id) == 2:
                    world_idx, tip_name = int(job_id[0]), str(job_id[1])
                    row = world_idx // self.num_unique_objects
                    failed_rows.setdefault(row, []).append(tip_name)
            if not failed_rows:
                raise
            cache = self._design_assess_cache
            cache_max = self._design_assess_cache_max
            for row, tip_names in failed_rows.items():
                start, stop = row * self.num_unique_objects, (row + 1) * self.num_unique_objects
                reason = f"embodiment row {row}, fingertip {', '.join(sorted(set(tip_names)))}: convex decomposition failed"
                rejected[start:stop] = True
                decomp_reasons[start:stop] = [reason] * self.num_unique_objects
                digest = _design_digest(assessed_values[row])
                if digest is not None:
                    cache[digest] = (True, reason, [], [], row)
                    if len(cache) > cache_max:
                        cache.popitem(last=False)
            self._reset_failed = True
            return self._get_obs(), torch.as_tensor(rejected, dtype=torch.bool, device=self.device), self._reset_info(tuple(decomp_reasons))
        except Exception:
            self.current_embodiment = previous_embodiment
            if rebuild:
                self.tip_meshes, self._needs_rebuild = previous_meshes, True
            raise
        if records is not None:
            self.last_quality_metrics = records
            self.last_quality_summary = summarize_quality_records(records)
        self._reset_failed = False
        self._episode_over = False
        self._finished.zero_()
        return observation, torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), self._reset_info(reasons)

    def _step(self, action, embodiment=None):
        if self._reset_failed:
            raise RuntimeError("step() cannot follow a rejected or failed reset; call reset() successfully first")
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if self._episode_over:
            raise RuntimeError(f"{self._env_label} episodes end together: call reset() before stepping again")
        if action.shape != (self.num_envs, self.single_action_space.shape[0]):
            raise ValueError(f"Expected actions of shape {(self.num_envs, self.single_action_space.shape[0])}, got {tuple(action.shape)}")
        observation, reward, done = self.step_worlds(action)
        terminal_reward = getattr(self, "ep_reward_term_sums", {}).get("terminal_reward")
        self.last_held_at_goal.zero_() if terminal_reward is None else self.last_held_at_goal.copy_(terminal_reward > 0)
        self.last_diverged.copy_(self.terminated_mask)
        terminated = self.terminated_mask & ~self._finished
        truncated = done & ~self.terminated_mask & ~self._finished
        self._finished |= terminated | truncated
        episode_over = (self.current_steps >= self.max_episode_steps) or bool(done.all())
        self._episode_over = episode_over
        info = self._step_info(reward=reward, terminated=terminated, truncated=truncated)
        info["finished"] = self._finished.clone()
        if episode_over:
            info["episode/completed"] = done.detach().clone()
            info["episode/final_return"] = self.ep_returns.detach().clone()
            for key, value in self.get_episode_reward_term_means().items():
                info[key] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
            object_names = np.asarray(self.obj_names_env)
            for object_name in sorted(set(self.obj_names_env)):
                mask = torch.as_tensor(object_names == object_name, dtype=torch.bool, device=self.device)
                info[f"object_return/{object_name}"] = self.ep_returns[mask].mean().detach().clone()
            for key, value in self.last_quality_summary.items():
                info[key] = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        return observation, reward, terminated, truncated, info

    def _reset_info(self, rejection_reasons) -> dict:
        info = self._step_info()
        if getattr(self, "_cached_emb_info_ref", None) is not self.current_embodiment:
            mesh_tensor_cache: dict[int, dict[str, torch.Tensor]] = {}
            per_design = []
            for design in self.current_embodiment:
                entry = {}
                for name, mesh in design.items():
                    mid = id(mesh)
                    cached_t = mesh_tensor_cache.get(mid)
                    if cached_t is None:
                        cached_t = {
                            "vertices": torch.as_tensor(mesh.vertices, dtype=torch.float64, device=self.device).clone(),
                            "faces": torch.as_tensor(mesh.faces, dtype=torch.long, device=self.device).clone(),
                        }
                        mesh_tensor_cache[mid] = cached_t
                    entry[name] = cached_t
                per_design.append(entry)
            self._cached_emb_info_val = self._expand_embodiments_to_envs(per_design)
            self._cached_emb_info_ref = self.current_embodiment
        info["embodiment/value"] = self._cached_emb_info_val
        info["embodiment/rejection_reasons"] = rejection_reasons
        info.update(self._quality_info())
        return info

    def _quality_info(self) -> dict[str, torch.Tensor]:
        records = self.last_quality_metrics
        if not records:
            return {}
        if getattr(self, "_cached_quality_records", None) is records and self._cached_quality_info is not None:
            return self._cached_quality_info
        keys = sorted({k for r in records for k, v in r.items() if k not in ("env_id", "effector") and isinstance(v, (int, float, np.number))})
        col_index = {name: idx for idx, name in enumerate(self._quality_columns)}
        arrays = {key: np.full((self.num_envs, len(col_index)), np.nan, dtype=np.float32) for key in keys}
        for record in records:
            env_id = int(record.get("env_id", -1))
            col = col_index.get(str(record.get("effector", "")))
            if col is None or not (0 <= env_id < self.num_envs):
                continue
            for key in keys:
                val = record.get(key)
                if val is not None:
                    arrays[key][env_id, col] = float(val)
        output = {f"mesh_quality/{key}": torch.as_tensor(arr, dtype=torch.float32, device=self.device) for key, arr in arrays.items()}
        self._cached_quality_records = records
        self._cached_quality_info = output
        return output

    def _extra_step_info(self) -> dict:
        return {}

    def _extra_optional_info(self) -> dict:
        return {}

    def _step_info(self, reward=None, terminated=None, truncated=None) -> dict:
        emb_id = getattr(self, "_embodiment_id_t", None)
        if emb_id is None:
            emb_id = torch.repeat_interleave(torch.arange(self.num_embodiments, dtype=torch.long, device=self.device), self.num_unique_objects)
        info = {
            "episode/step": torch.full((self.num_envs,), self.current_steps, dtype=torch.long, device=self.device),
            "episode/return": self.ep_returns.detach().clone(),
            "embodiment/id": emb_id,
            "task/object": tuple(self.obj_names_env),
            "contacts/fingertip_object": self.ftip_touches_object.detach().clone(),
            "safeguards/force_count": torch.as_tensor(self.force_safeguard_counts, device=self.device).clone(),
            "safeguards/tunneling_count": torch.as_tensor(self.tunneling_safeguard_counts, device=self.device).clone(),
            "safeguards/in_air_count": torch.as_tensor(self.in_air_safeguard_counts, device=self.device).clone(),
            **self._extra_step_info(),
        }
        optional = {
            "rewards/total": reward,
            "rewards/contact": getattr(self, "last_contact_reward", None),
            "rewards/distance": getattr(self, "last_dist_reward_scaled", None),
            "rewards/f2f_penalty": getattr(self, "last_f2f_penalty", None),
            "episode/terminated": terminated,
            "episode/truncated": truncated,
            **self._extra_optional_info(),
        }
        info.update({key: value.detach().clone() for key, value in optional.items() if value is not None})
        return info

    def render(self, env_idx=None):
        return render_batch(self, lambda env, env_idx: env._render_world(env_idx), env_idx)

    def close(self) -> None:
        close_human(self)
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.closed = True
