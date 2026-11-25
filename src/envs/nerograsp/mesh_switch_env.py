"""Batched MuJoCo Warp base class for worlds whose models differ in their meshes.

Unlike the envs that compile one ``MjSpec`` and vary it on the device, here
every world has its own geometry (its fingertip convex parts and its object),
which only the MuJoCo compiler can turn into masses, inertias and bounding
volumes. So a model build:

1. compiles one model per distinct world variant (a variant spec holds only
   that variant's own meshes, so each compile stays small however many worlds
   there are);
2. compiles one *global* model: world 0's variant plus every mesh of every
   world. Its mesh table and every field not listed in ``PER_WORLD_MODEL_FIELDS``
   are shared by all worlds;
3. uploads the global model to mujoco_warp and replaces each per-world field
   with the stack of the worlds' own values, mapping mesh ids by mesh name.

Fields MuJoCo Warp keeps global (``geom_contype``, ``geom_conaffinity``, ...)
therefore come from world 0.

A control step copies the controls and runs ``control_substeps`` physics
substeps with gravity compensation on the robot dofs; on CUDA the substeps
are captured once per build as a graph and replayed.

Code layout:

- VariantMujocoEnv: the base class, subclasses implement the "Subclass hooks";
- PER_WORLD_MODEL_FIELDS: the model fields stacked per world.
"""

from __future__ import annotations

import concurrent.futures
import copy
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Hashable, Sequence

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from mujoco_warp._src.types import vec5 as mjw_vec5


@wp.kernel
def reset_worlds_kernel(
    qpos: wp.array(dtype=wp.float32, ndim=2),
    qvel: wp.array(dtype=wp.float32, ndim=2),
    saved_qpos: wp.array(dtype=wp.float32, ndim=2),
    saved_qvel: wp.array(dtype=wp.float32, ndim=2),
    mask: wp.array(dtype=wp.uint8, ndim=1),
):
    env_id = wp.tid()
    if mask[env_id] > wp.uint8(0):
        for i in range(qpos.shape[1]):
            qpos[env_id, i] = saved_qpos[env_id, i]
        for i in range(qvel.shape[1]):
            qvel[env_id, i] = saved_qvel[env_id, i]


@wp.kernel
def gravity_compensation_kernel(
    qfrc_applied: wp.array(dtype=wp.float32, ndim=2),
    qfrc_bias: wp.array(dtype=wp.float32, ndim=2),
    dof_addrs: wp.array(dtype=wp.int32, ndim=1),
):
    env_id = wp.tid()
    for i in range(qfrc_applied.shape[1]):
        qfrc_applied[env_id, i] = 0.0
    for j in range(dof_addrs.shape[0]):
        dof = dof_addrs[j]
        qfrc_applied[env_id, dof] = qfrc_bias[env_id, dof]

PER_WORLD_MODEL_FIELDS = {
    "qpos0": float,
    "geom_dataid": int,
    "geom_size": wp.vec3,
    "geom_aabb": wp.vec3,
    "geom_rbound": float,
    "geom_pos": wp.vec3,
    "geom_quat": wp.quat,
    "geom_friction": wp.vec3,
    "geom_solref": wp.vec2,
    "geom_solimp": mjw_vec5,
    "geom_rgba": wp.vec4,
    "body_pos": wp.vec3,
    "body_quat": wp.quat,
    "body_mass": float,
    "body_subtreemass": float,
    "body_inertia": wp.vec3,
    "body_invweight0": wp.vec2,
    "body_ipos": wp.vec3,
    "body_iquat": wp.quat,
    "dof_frictionloss": float,
}
RENDER_H, RENDER_W = 1080, 1920


def _mesh_names(model: mujoco.MjModel) -> list[str]:
    return [model.mesh(i).name for i in range(model.nmesh)]


def _extract_per_world_fields(model: mujoco.MjModel, global_ids: dict[str, int]) -> dict[str, np.ndarray]:
    """Copy ``PER_WORLD_MODEL_FIELDS`` from one compiled variant, remapping mesh IDs into ``global_ids``."""
    remap = np.array([global_ids[name] for name in _mesh_names(model)], dtype=np.int64)
    out = {}
    for field in PER_WORLD_MODEL_FIELDS:
        value = np.array(model.geom_aabb.reshape(model.ngeom, 2, 3) if field == "geom_aabb" else getattr(model, field))
        if field == "geom_dataid":
            value = np.where(value >= 0, remap[np.maximum(value, 0)], -1).astype(value.dtype)
        out[field] = value
    return out


def _map_in_parallel(items, submit_fn, on_done, max_workers: int) -> None:
    """Run ``submit_fn(item)`` in a bounded thread pool, invoking ``on_done(item, result)`` as each finishes."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending = {}
        for item in items:
            pending[pool.submit(submit_fn, item)] = item
            if len(pending) >= max_workers:
                done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    on_done(pending.pop(future), future.result())
        for future in concurrent.futures.as_completed(pending):
            on_done(pending[future], future.result())


def _patch_tip_micro_model(
    stacked: dict[str, np.ndarray],
    global_ids: dict[str, int],
    tip_body_id: int,
    tip_part_gids: list[int],
    tip_name: str,
    worlds: list[int],
    micro_model: mujoco.MjModel,
    max_parts: int,
) -> None:
    """Write one compiled 1-body fingertip micro-model's inertia and part geoms into ``stacked``."""
    w_idx = np.asarray(worlds, dtype=np.int64)
    local_body = mujoco.mj_name2id(micro_model, mujoco.mjtObj.mjOBJ_BODY, tip_name)
    stacked["body_mass"][w_idx, tip_body_id] = micro_model.body_mass[local_body]
    stacked["body_inertia"][w_idx, tip_body_id] = micro_model.body_inertia[local_body]
    stacked["body_ipos"][w_idx, tip_body_id] = micro_model.body_ipos[local_body]
    stacked["body_iquat"][w_idx, tip_body_id] = micro_model.body_iquat[local_body]

    mesh_remap = np.array([global_ids.get(name, -1) for name in _mesh_names(micro_model)], dtype=np.int64)
    local_gids = [mujoco.mj_name2id(micro_model, mujoco.mjtObj.mjOBJ_GEOM, f"{tip_name}_part_{j:02d}") for j in range(max_parts)]
    micro_aabb = micro_model.geom_aabb.reshape(micro_model.ngeom, 2, 3)
    for local_gid, global_gid in zip(local_gids, tip_part_gids):
        data_id = int(micro_model.geom_dataid[local_gid])
        stacked["geom_dataid"][w_idx, global_gid] = mesh_remap[data_id] if data_id >= 0 else -1
        stacked["geom_aabb"][w_idx, global_gid] = micro_aabb[local_gid]
        stacked["geom_rbound"][w_idx, global_gid] = micro_model.geom_rbound[local_gid]
        stacked["geom_pos"][w_idx, global_gid] = micro_model.geom_pos[local_gid]
        stacked["geom_quat"][w_idx, global_gid] = micro_model.geom_quat[local_gid]
        stacked["geom_size"][w_idx, global_gid] = micro_model.geom_size[local_gid]
        stacked["geom_rgba"][w_idx, global_gid] = micro_model.geom_rgba[local_gid]


def _recompute_subtree_mass(body_mass: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Propagate updated per-world body masses bottom-up to compute ``body_subtreemass``."""
    subtree = body_mass.copy()
    for body_id in range(len(parents) - 1, 0, -1):
        subtree[:, parents[body_id]] += subtree[:, body_id]
    return subtree


def build_tip_micro_spec(
    scene_root: ET.Element,
    tip_body_name: str,
    unused_mesh_name: str,
    unused_corners: np.ndarray,
    inertia: mujoco.mjtMeshInertia,
) -> mujoco.MjSpec:
    """Build a minimal 1-body MjSpec containing only ``tip_body_name`` and its assets.

    Compiling this micro-spec computes the exact same MuJoCo body mass/inertia/ipos/iquat
    and part geom AABBs/rbounds as compiling the full robot, at ~50-80x lower CPU cost.
    """
    tip_body = scene_root.find(f".//body[@name='{tip_body_name}']")
    if tip_body is None:
        raise ValueError(f"Tip body '{tip_body_name}' not found in scene XML.")

    micro = ET.Element("mujoco")
    for tag in ("compiler", "default"):
        for el in scene_root.findall(tag):
            micro.append(copy.deepcopy(el))

    # Collect mesh names referenced directly by this tip body or any default class.
    needed_meshes: set[str] = set()
    for g in tip_body.iter("geom"):
        if g.get("mesh"):
            needed_meshes.add(g.get("mesh"))
    for d in scene_root.iter("default"):
        for g in d.findall("geom"):
            if g.get("mesh"):
                needed_meshes.add(g.get("mesh"))

    asset = ET.SubElement(micro, "asset")
    for src_asset in scene_root.findall("asset"):
        for child in src_asset:
            if child.tag == "material" and not child.get("texture"):
                asset.append(copy.deepcopy(child))
            elif child.tag == "mesh" and child.get("name") in needed_meshes:
                asset.append(copy.deepcopy(child))

    worldbody = ET.SubElement(micro, "worldbody")
    body_copy = copy.deepcopy(tip_body)
    body_copy.set("pos", "0 0 0")
    body_copy.set("quat", "1 0 0 0")
    worldbody.append(body_copy)

    spec = mujoco.MjSpec.from_string(ET.tostring(micro, encoding="unicode"))
    spec.add_mesh(
        name=unused_mesh_name,
        uservert=unused_corners.reshape(-1).tolist(),
        userface=[0, 1, 2, 0, 3, 1, 0, 2, 3, 1, 3, 2],
        inertia=inertia,
    )
    return spec


class MeshSwitchMujocoEnv:
    """Steps ``num_envs`` MuJoCo worlds in lockstep, each with its own compiled model variant.

    Holds the global ``m`` (and a CPU ``d``), the batched ``cp_model``/``cp_data``,
    the captured substep graph, and the per-world render models in ``variants``
    (``None`` for worlds that are never drawn).
    """

    REAL_CONTROL_HZ = 10.0
    GRAVITY_COMPENSATION = True
    # Rendering-only geom-group overrides. Physics/contact behavior is unchanged.
    RENDER_SHOW_VISUAL_GROUP = False
    RENDER_HIDE_COLLISION_GROUP = False

    def __init__(self, num_envs: int, *, device: str, viz_env_id: int = 0, debug: bool = False, verbose: bool = False,
                 compile_workers: int = 1, retain_all_render_models: bool = False, njmax: int = 256, nconmax: int = 192):
        wp.init()
        self.num_envs = num_envs
        self.device = device
        self.use_cpu = device == "cpu"
        self.viz_env_id = viz_env_id
        self.debug = debug
        self.verbose = verbose or debug
        self.compile_workers = max(1, min(num_envs, int(compile_workers)))
        self.retain_all_render_models = retain_all_render_models
        self.njmax, self.nconmax = njmax, nconmax
        self.m = self.d = self.cp_model = self.cp_data = self.variants = None
        self.step_graph = None
        self.renderer = self.camera = None
        self._renderer_m = self._render_d = self._render_m = None

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    def _configure_compiled(self, model: mujoco.MjModel) -> None:
        """Adjust a freshly compiled model (options, contact bits, colours) before its fields are read."""

    def _setup_camera(self, camera: mujoco.MjvCamera) -> None:
        """Configure the render and viewer camera."""

    def _add_debug_visualizations(self, scene: mujoco.MjvScene, env_idx: int) -> None:
        """Inject visual-only markers into a scene (``debug=True``)."""

    def _robot_dof_addrs(self) -> list[int]:
        """Dofs that receive gravity compensation."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Model build
    # ------------------------------------------------------------------ #
    def compile_worlds(self, global_spec: mujoco.MjSpec, variant_spec: Callable[[Hashable], mujoco.MjSpec],
                       world_keys: Sequence[Hashable]) -> None:
        """Build the batched model.

        ``global_spec`` is world 0's variant with every world's meshes added;
        ``variant_spec(key)`` returns a fresh spec of one variant; world ``i``
        uses variant ``world_keys[i]``.
        """
        started = time.perf_counter()
        self._release_model()
        retain = set(range(self.num_envs)) if self.retain_all_render_models else {0, self.viz_env_id}
        worlds_of: dict[Hashable, list[int]] = {}
        for i, key in enumerate(world_keys):
            worlds_of.setdefault(key, []).append(i)

        def compile_one(spec):
            model = spec.compile()
            self._configure_compiled(model)
            return model

        global_model = compile_one(global_spec)
        global_ids = {name: i for i, name in enumerate(_mesh_names(global_model))}
        rows: dict[str, dict[Hashable, np.ndarray]] = {field: {} for field in PER_WORLD_MODEL_FIELDS}
        variants: list = [None] * self.num_envs

        def consume(key, model):
            """Copy the variant's per-world fields out; keep the model only for the worlds drawn."""
            for field, value in _extract_per_world_fields(model, global_ids).items():
                rows[field][key] = value
            # Each drawn world gets a model of its own: resets resize the washer in place.
            drawn = [i for i in worlds_of[key] if i in retain]
            for n, i in enumerate(drawn):
                variants[i] = model if n == 0 else copy.copy(model)

        # Specs are built on this thread, at most compile_workers ahead; only the compiles run in parallel.
        _map_in_parallel(
            list(worlds_of),
            lambda key: compile_one(variant_spec(key)),
            consume,
            self.compile_workers,
        )
        stacked = {field: np.stack([values[key] for key in world_keys]) for field, values in rows.items()}
        self._upload_compiled_model(global_model, stacked, variants)
        if self.verbose:
            print(f"[TIMING] MuJoCo compile ({len(worlds_of)} variants + global) and GPU transfer took {time.perf_counter() - started:.3f}s.")

    def compile_worlds_fast(
        self,
        global_spec: mujoco.MjSpec,
        base_template_specs: dict[Hashable, mujoco.MjSpec],
        world_base_keys: Sequence[Hashable],
        tip_micro_spec: Callable[[str, Hashable], mujoco.MjSpec],
        world_tip_keys: Sequence[dict[str, Hashable]],
        tip_names: Sequence[str],
        max_parts: int,
    ) -> None:
        """Fast batched model build using 1-body fingertip micro-specs instead of N full-robot compiles."""
        started = time.perf_counter()
        self._release_model()

        def compile_one(spec: mujoco.MjSpec) -> mujoco.MjModel:
            model = spec.compile()
            self._configure_compiled(model)
            return model

        global_model = compile_one(global_spec)
        global_ids = {name: i for i, name in enumerate(_mesh_names(global_model))}

        # 1. Compile one full-robot template per unique base object.
        base_rows: dict[str, dict[Hashable, np.ndarray]] = {field: {} for field in PER_WORLD_MODEL_FIELDS}
        for bkey, bspec in base_template_specs.items():
            for field, value in _extract_per_world_fields(compile_one(bspec), global_ids).items():
                base_rows[field][bkey] = value
        stacked = {field: np.stack([values[bkey] for bkey in world_base_keys]) for field, values in base_rows.items()}

        # 2. Deduplicate and compile 1-body fingertip micro-specs in parallel, then patch into stacked rows.
        unique_tips: dict[tuple[str, Hashable], list[int]] = {}
        for world_id, tip_map in enumerate(world_tip_keys):
            for tip_name in tip_names:
                unique_tips.setdefault((tip_name, tip_map[tip_name]), []).append(world_id)
        tip_body_ids = {t: mujoco.mj_name2id(global_model, mujoco.mjtObj.mjOBJ_BODY, t) for t in tip_names}
        tip_part_gids = {
            t: [mujoco.mj_name2id(global_model, mujoco.mjtObj.mjOBJ_GEOM, f"{t}_part_{j:02d}") for j in range(max_parts)]
            for t in tip_names
        }
        _map_in_parallel(
            list(unique_tips.items()),
            lambda item: compile_one(tip_micro_spec(item[0][0], item[0][1])),
            lambda item, mm: _patch_tip_micro_model(
                stacked, global_ids, tip_body_ids[item[0][0]], tip_part_gids[item[0][0]], item[0][0], item[1], mm, max_parts
            ),
            self.compile_workers,
        )

        # 3. Propagate updated tip masses to body_subtreemass and upload.
        stacked["body_subtreemass"] = _recompute_subtree_mass(stacked["body_mass"], global_model.body_parentid)
        variants: list = [None] * self.num_envs
        variants[0] = global_model
        self._upload_compiled_model(global_model, stacked, variants)
        if self.verbose:
            print(
                f"[TIMING] MuJoCo fast compile ({len(base_template_specs)} base + {len(unique_tips)} tip micro-specs + global) "
                f"and GPU transfer took {time.perf_counter() - started:.3f}s."
            )

    def _upload_compiled_model(self, global_model: mujoco.MjModel, stacked: dict[str, np.ndarray], variants: list) -> None:
        can_reuse_data = (
            self.cp_data is not None
            and self.m is not None
            and self.m.nq == global_model.nq
            and self.m.nv == global_model.nv
            and self.m.nbody == global_model.nbody
            and self.m.ngeom == global_model.ngeom
            and self.m.nu == global_model.nu
        )
        self.m = global_model
        self.d = mujoco.MjData(self.m)
        self.control_substeps = max(1, int((1.0 / self.REAL_CONTROL_HZ) / self.m.opt.timestep))
        mujoco.mj_forward(self.m, self.d)
        self.variants = variants
        with wp.ScopedDevice(self.device):
            self.cp_model = mjw.put_model(self.m)
            # fidelity=0/1 intentionally cap solver/linesearch iterations;
            # silence those expected cap warnings while keeping buffer-overflow warnings enabled.
            if hasattr(self.cp_model.opt, "warn_overflow") and hasattr(mjw, "OverflowType"):
                self.cp_model.opt.warn_overflow &= ~(
                    int(mjw.OverflowType.ITERATIONS) | int(mjw.OverflowType.LS_ITERATIONS)
                )
            if not can_reuse_data:
                self.cp_data = mjw.make_data(self.m, njmax=self.njmax, nconmax=self.nconmax, nworld=self.num_envs)
            for field, wp_type in PER_WORLD_MODEL_FIELDS.items():
                setattr(self.cp_model, field, wp.array(stacked[field], dtype=wp_type))
            mjw.set_const(self.cp_model, self.cp_data)

    def _release_model(self) -> None:
        """Drop the previous compiled model before building its replacement (keeping cp_data for reuse)."""
        if any(getattr(self, attr, None) is not None for attr in ("cp_model", "d", "variants", "renderer")):
            self.cp_model = self.d = self.variants = None
            self.step_graph = None
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
            self._renderer_m = self._render_d = self._render_m = None

    # ------------------------------------------------------------------ #
    # Physics
    # ------------------------------------------------------------------ #
    def _substeps(self) -> None:
        for _ in range(self.control_substeps):
            if self.GRAVITY_COMPENSATION:
                mjw.step1(self.cp_model, self.cp_data)
                wp.launch(gravity_compensation_kernel, dim=self.num_envs,
                          inputs=[self.cp_data.qfrc_applied, self.cp_data.qfrc_bias, self._robot_dofs_wp], device=self.device)
                mjw.step2(self.cp_model, self.cp_data)
            else:
                mjw.step(self.cp_model, self.cp_data)

    def _capture_step_graph(self) -> None:
        """Compile the substep kernels (by running them once), then capture them as a CUDA graph.

        The run advances the physics: the caller resets the worlds afterwards.
        """
        self._robot_dofs_wp = wp.array(self._robot_dof_addrs(), dtype=wp.int32, device=self.device)
        self.step_graph = None
        if self.use_cpu or not wp.get_device(self.device).is_cuda:
            return
        with wp.ScopedDevice(self.device):
            self._substeps()
            wp.capture_begin()
            self._substeps()
            self.step_graph = wp.capture_end()

    def _physics_step(self, ctrl: torch.Tensor) -> None:
        with wp.ScopedDevice(self.device):
            if ctrl.shape[1] < self.m.nu:
                padded = torch.zeros((self.num_envs, self.m.nu), dtype=torch.float32, device=self.device)
                padded[:, :ctrl.shape[1]] = ctrl
                ctrl = padded
            if not self.use_cpu and ctrl.device.type == "cuda":
                stream = wp.stream_from_torch(torch.cuda.current_stream(ctrl.device))
                with wp.ScopedStream(stream):
                    wp.copy(self.cp_data.ctrl, wp.from_torch(ctrl))
                    if self.step_graph is not None:
                        wp.capture_launch(self.step_graph, stream=stream)
                    else:
                        self._substeps()
            elif self.use_cpu:
                wp.copy(self.cp_data.ctrl, wp.from_torch(ctrl))
                # Zero-copy torch views avoid a kernel launch per substep on CPU.
                dofs = self._robot_dof_addrs()
                for _ in range(self.control_substeps):
                    if self.GRAVITY_COMPENSATION:
                        mjw.step1(self.cp_model, self.cp_data)
                        qfrc_applied = wp.to_torch(self.cp_data.qfrc_applied)
                        qfrc_applied.zero_()
                        qfrc_applied[:, dofs] = wp.to_torch(self.cp_data.qfrc_bias)[:, dofs]
                        mjw.step2(self.cp_model, self.cp_data)
                    else:
                        mjw.step(self.cp_model, self.cp_data)
            else:
                wp.copy(self.cp_data.ctrl, wp.from_torch(ctrl))
                self._substeps()

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _render_world(self, env_idx: int | None = None, qpos=None) -> np.ndarray:
        """An RGB frame of one world, drawn from its retained model."""
        if env_idx is None:
            env_idx = self.viz_env_id
        render_m = self.m
        if self.variants and 0 <= env_idx < len(self.variants):
            if self.variants[env_idx] is not None:
                render_m = self.variants[env_idx]
            elif env_idx != 0:
                raise ValueError(f"No retained render model for environment {env_idx}")
        if self._render_d is None or self._render_m is not render_m:
            self._render_d = mujoco.MjData(render_m)
            self._render_m = render_m
        if self.renderer is None or self._renderer_m is not render_m:
            if self.renderer is not None:
                self.renderer.close()
            self.renderer = mujoco.Renderer(render_m, RENDER_H, RENDER_W)
            self._renderer_m = render_m
            if self.camera is None:
                self.camera = mujoco.MjvCamera()
                mujoco.mjv_defaultFreeCamera(render_m, self.camera)
                self._setup_camera(self.camera)
        if qpos is not None:
            self._render_d.qpos[:] = qpos
        else:
            if not self.use_cpu and torch.cuda.is_available():
                torch.cuda.current_stream(self.device).synchronize()
            self._render_d.qpos[:] = wp.to_torch(self.cp_data.qpos)[env_idx].detach().cpu().numpy()
            self._render_d.qvel[:] = wp.to_torch(self.cp_data.qvel)[env_idx].detach().cpu().numpy()
        mujoco.mj_forward(render_m, self._render_d)
        opt = mujoco.MjvOption()
        if self.RENDER_HIDE_COLLISION_GROUP:
            opt.geomgroup[3] = 0
        show_group_4 = int(self.debug or self.RENDER_SHOW_VISUAL_GROUP)
        opt.geomgroup[4] = opt.sitegroup[4] = show_group_4
        self.renderer.update_scene(self._render_d, camera=self.camera, scene_option=opt)
        if self.debug:
            self._add_debug_visualizations(self.renderer.scene, env_idx)
        return self.renderer.render().copy()
