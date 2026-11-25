"""Rotate in hand: a hand turns an object to target orientations, with a
different hand shape in every world.

Every world runs the same scene: a hand (body ``palm``, optionally on a wrist)
whose palm and fingers carry heightfield pads, and one object dropped into it.
The embodiment action raises the pads' cells and stretches or resizes the
offsettable links, at respawn or during deform windows of the rollout. The
policy drives the hand's actuators. A fixed ``goal_pose`` is one goal per
episode, a randomised goal is a stream, each drawn ``goal_dist`` from the
object's orientation when the last is met. The object is observed in the
palm's frame. The goal is in the palm's frame too (``goal_frame="palm"``), so
turning the wrist turns it along with the object, or fixed in the world
(``goal_frame="world"``). It is observed in the palm's frame either way.

Rewards are given only for touching. The shaping is potential-based, the
previous minus the current angle to the goal, so turning away costs what
turning back gives and no loop gains anything (gains are given times the touch
multiplier, losses in full). Each step held at the goal is rewarded too.
Dropping the object or running out of time per goal ends the episode.

Code layout:

- rotate_in_hand_env.py: scene assembly, buffer and kernel-argument setup,
spaces, observation, reward and termination, and rendering. Its copy in
``hand_pick_up`` is that env's base, which overrides the hooks marked as such;
- rotate_in_hand_kernels.py: the device work, per step and after physics;
- offsettable_logic.py, hfield_logic.py: link and pad deformation;
- deform_color.py: optional pad colouring by relief;
- parallel_env.py: the batched base class and its step program.
"""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from . import rotate_in_hand_kernels as kern
from .deform_color import COLOR_SCALE, PadPainter, add_pad_textures, parse_scale
from .hfield_logic import (build_pad_arrays, empty_pads, find_pads, n_cells,
                           overflow_reports, relief_weights, subdivide_hfields,
                           tile_hfields, world_geom_dataid)
from .offsettable_logic import (DeformArrays, build_seg_arrays, find_segments,
                                floor_reports, n_slots, release_geom_frames,
                                slot_bounds)
from .parallel_env import ParallelMujocoEnv

_ASSETS = Path(__file__).resolve().parent / "assets"

# The palm body the hand must have, and the folders bare asset names sit in
PALM_BODY = "palm"
_HAND_DIR = "rotate_in_hand"
_SCENE_DIR = "_common"
_OBJECT_DIR = "_objects"

_SPAWN_CLEARANCE = 0.005                   # drop height above the pads, on top of the object's reach
CONTACT_STEPS = 2                          # control steps a contact is remembered
_PI = float(np.pi)


def _asset(name: str, folder: str) -> Path:
    # A bare name sits in folder, a path is relative to assets/
    p = Path(name)
    return _ASSETS / p if len(p.parts) > 1 else _ASSETS / folder / name


def _object_asset(name: str) -> Path:
    # A bare name is a folder under assets/_objects holding a robot.xml
    p = Path(name)
    if len(p.parts) > 1 or p.suffix == ".xml":
        return _ASSETS / p
    return _ASSETS / _OBJECT_DIR / name / "robot.xml"


def _quat_mat(q: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(q, dtype=np.float64))
    return mat.reshape(3, 3)


def _unit_quat(q, name: str) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64)
    if out.shape != (4,):
        raise ValueError(f"{name} is a w,x,y,z quaternion, got {q}")
    norm = float(np.linalg.norm(out))
    if norm < 1e-9:
        raise ValueError(f"{name} {q} has no direction")
    return out / norm


def _check_fraction(f: float) -> None:
    if not 0.0 <= float(f) <= 1.0:
        raise ValueError(f"hfield_offset_fraction is a share of each pad's "
                         f"MJCF elevation range and lives in [0, 1], got {f}")


def _joint_widths(jnt_type) -> tuple[int, int]:
    if jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        return 7, 6
    if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        return 4, 3
    return 1, 1


def _geom_top(m: mujoco.MjModel, d: mujoco.MjData, g: int) -> float:
    """World z of the top of geom ``g``, from its box (a bounding sphere would put
    a flat palm's top far too high)."""
    aabb = np.asarray(m.geom_aabb[g], dtype=np.float64).reshape(2, 3)
    up = np.asarray(d.geom_xmat[g], dtype=np.float64).reshape(3, 3)[2]
    return float(d.geom_xpos[g][2] + up @ aabb[0] + np.abs(up) @ aabb[1])


def _empty_segs() -> dict[str, np.ndarray]:
    """An empty link table shaped like ``find_segments``', for a run without
    link deformation. Index columns stay integer so they can index arrays."""
    width = {"dir": 3, "tip0": 3, "axis_dir": 3, "child_pos": 3, "site_pos": 3,
             "size0": 3, "min_size": 3, "box_pos": 3, "pad_map": 3,
             "box_quat": 4, "pad_size0": 4}
    ints = {"body", "geom", "is_cyl", "axis_adr", "axis_num", "axis_act",
            "child_adr", "child_num", "child_id", "site_adr", "site_num",
            "site_id", "is_box", "pad_adr", "pad_num", "pad_geom",
            "pad_hfield", "pad_map"}
    keys = ("body", "geom", "dir", "radius", "half0", "density", "is_cyl",
            "tip0", "min_half", "axis_adr", "axis_num", "child_adr",
            "child_num", "site_adr", "site_num", "axis_act", "axis_lo",
            "axis_hi", "axis_dir", "child_id", "site_id", "child_pos",
            "site_pos", "is_box", "size0", "min_size", "box_pos", "box_quat",
            "pad_adr", "pad_num", "pad_geom", "pad_hfield", "pad_map",
            "pad_size0")
    return {k: np.zeros((0, width[k]) if k in width else 0,
                        dtype=np.int64 if k in ints else np.float64)
            for k in keys}


class RotateInHandParallelEnv(ParallelMujocoEnv):
    """Batched in-hand reorientation env with one hand shape per world.

    Besides the constructor settings, an instance holds:

    - model ids and addresses cached at setup (palm, object, hand joints) and
    the spawn point;
    - the embodiment tables: pads (cells first in the action) and links, and
    per-world views of the device-side heightfield copies the pads are written
    into;
    - per-world device buffers the kernels hold warp views of, packed once into
    the kernel argument structs, including the goal, object and contact state
    the post-physics kernels write;
    - host-side shaping state and ``last_*`` outcome flags;
    - the optional pad painter.

    Subclasses adjust it through the class attributes and the override hooks.
    """

    _length_floor_warned = False      # see _warn_shape_limits


    # Written per world by the deform, and geom_dataid, which points each world's
    # pads at its own heightfield copies
    PER_WORLD_MODEL_FIELDS = ("geom_pos", "geom_quat", "geom_size", "geom_rbound",
                              "geom_aabb", "body_pos", "body_ipos", "body_iquat",
                              "body_mass", "body_inertia", "site_pos",
                              "geom_dataid")

    HAND_QUAT = (1.0, 0.0, 0.0, 0.0)          # frame the hand is attached with, wxyz
    EXTRA_OBS = 0                             # obs columns a subclass appends via _extra_obs
    TABLE_TOUCH = 0                           # 1: world contact halves the reward, see window_kernel

    @classmethod
    def accepted_kwargs(cls) -> set[str]:
        """Every keyword the constructor takes, subclasses included."""
        out = set()
        for klass in cls.__mro__:
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            for name, p in inspect.signature(init).parameters.items():
                if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    out.add(name)
        return out - {"self"}

    def __init__(
        self,
        num_envs: int = 4,
        *,
        hand_xml: str = "hand_no_xyz.xml",
        object_xml: str = "set_screw_big",
        scene_xml: str = "scene_rotate_in_hand.xml",
        # Keyframes combined into the reset pose, matched by suffix since
        # attaching prefixes them
        init_key: str | tuple[str, ...] | list[str] = ("init_pos",),

        # Respawn orientation in the palm frame at the reset pose, if not
        # randomised. It must be one the object rests in, or it topples and the
        # first goal is off
        init_quat=(1.0, 0.0, 0.0, 0.0),
        randomise_init_quat: bool = True,

        color_deforms: bool = False,              # colour pads by relief, see deform_color
        color_scale: list[list[float]] | None = None,
        color_interp: bool = True,

        # Deform windows from the respawn [s], 0 deforms only at respawn
        deform_hfield_during_rollout_s: float = 0,
        deform_links_during_rollout_s: float = 0,

        max_link_offset: float = 0.02,            # metres per unit of link embodiment action
        hfield_offset_fraction: float = 1.0,      # share of each pad's range a cell at 1 rises to
        # Heightfield samples per pad cell per side (0 or 1: one per cell), filled
        # by a degree-spline_degree spline through the cells; the embodiment keeps
        # one entry per MJCF cell either way. See hfield_logic
        grid_subd: int = 0,
        spline_degree: int = 2,
        # Heightfield samples (after subdivision) over which the pads along each
        # finger link taper to 0 at the joints, 0 turns it off. See relief_weights
        relief_length: int = 0,
        min_link_length: float = 0.005,           # deformed links are clamped to this

        max_pose_return: float = 5.0,             # shaping return per goal
        max_goal_return: float = 5.0,             # return for holding a goal, per goal
        goal_dwell_steps: int = 0,                # total steps at the goal that meet it (0 means 1)
        goal_rot_tol: float = 0.1,
        contact_steps: int = CONTACT_STEPS,

        goal_pose=[1, 0, 0, 0],                   # fixed goal quat (in goal_frame), or None for random goals
        goal_frame: str = "palm",                 # "palm": goals turn with the wrist; "world": they stay put
        goal_dist: float = 0.5,                   # angle of a new random goal from the current pose [rad]

        # Random horizontal push on the object, held for force_hold of every
        # force_every control steps
        force_mag: float = 0.0,
        force_every: int = 0,
        force_hold: int = 0,

        max_rel_dist: float = 0.1,                # obs scale of the object position
        vel_scale: float = 1.0,                   # obs scale of velocities
        force_viz_scale: float = 0.02,
        goal_viz_scale: float = 1.0,              # length of the debug object and goal frames, x the default

        timestep: float = 0.005,
        phys_substeps: int = 8,
        # GJK/EPA termination tolerance [m]. At MuJoCo's 1e-6 EPA keeps iterating
        # on a curved object against a pad's prisms long after the depth settles;
        # 1e-4 moves depths by under 0.1 mm and takes about a third off the step
        ccd_tolerance: float = 1e-4,
        goal_limit_s: float = 5,                  # time to meet one random goal
        total_limit_s: float = 10,                # episode length (a truncation)

        # parallel-runner params
        device: str = "cuda",
        viz_env_id: int = 0,
        render_mode: str | None = None,
        debug: bool = False,
        use_graph: bool = True,
        check_finite: bool = False,
        njmax: int | None = 256,
        nconmax: int | None = None,
        seed: int = 1234,
    ):
        if max_link_offset < 0.0:
            raise ValueError(f"max_link_offset must be >= 0, got "
                             f"{max_link_offset}")
        _check_fraction(hfield_offset_fraction)
        if min_link_length <= 0.0:
            raise ValueError("min_link_length must be positive")
        if not 0.0 < goal_dist <= _PI:
            raise ValueError(f"goal_dist is a rotation angle in (0, pi], got "
                             f"{goal_dist}; pi is as far apart as two "
                             "orientations can be")
        if not 0.0 < goal_rot_tol <= _PI:
            raise ValueError("goal_rot_tol is an angle in (0, pi], got "
                             f"{goal_rot_tol}")
        if not 1 <= int(contact_steps) <= 30:   # one bit per step in an int32
            raise ValueError(f"contact_steps must be in [1, 30], got "
                             f"{contact_steps}")
        if ccd_tolerance <= 0.0:
            raise ValueError(f"ccd_tolerance is a distance and must be > 0, "
                             f"got {ccd_tolerance}")
        self._xmls = (hand_xml, object_xml, scene_xml)
        self.init_key = ((init_key,) if isinstance(init_key, str)
                         else tuple(init_key))
        self.init_quat = _unit_quat(init_quat, "init_quat")
        self.randomise_init_quat = bool(randomise_init_quat)
        self.deform_hfield_s = float(deform_hfield_during_rollout_s)
        self.deform_links_s = float(deform_links_during_rollout_s)
        self.max_link_offset = float(max_link_offset)
        self.hfield_offset_fraction = float(hfield_offset_fraction)
        self.min_link_length = float(min_link_length)
        if int(grid_subd) < 0:
            raise ValueError(f"grid_subd is a count of samples per cell, got {grid_subd}")
        if not 1 <= int(spline_degree) <= 5:
            raise ValueError(f"spline_degree is in 1..5, got {spline_degree}")
        self.grid_subd = int(grid_subd)
        self.spline_degree = int(spline_degree)
        if int(relief_length) != relief_length or relief_length < 0:
            raise ValueError(f"relief_length is a count of samples >= 0 (0 is off), got {relief_length}")
        self.relief_length = int(relief_length)
        self.deform_links = self.max_link_offset > 0.0
        self.deform_hfield = self.hfield_offset_fraction > 0.0
        if color_deforms and not self.deform_hfield:
            raise ValueError(
                "color_deforms=True needs pads in the deform action, but "
                "hfield_offset_fraction is 0, so no cell is ever raised. Raise "
                "it, or leave the pads their MJCF colour.")
        self.color_deforms = bool(color_deforms)
        self.color_scale = color_scale
        self.color_interp = bool(color_interp)
        self._painter: PadPainter | None = None
        if self.color_deforms:
            # Validate before the model is built, rather than after compiling all worlds
            parse_scale(COLOR_SCALE if color_scale is None else color_scale)
        self.max_pose_return = float(max_pose_return)
        self.max_goal_return = float(max_goal_return)
        self.goal_dwell_steps = max(int(goal_dwell_steps), 1)   # either way a met goal gives max_goal_return
        self.goal_rot_tol = float(goal_rot_tol)
        self.contact_steps = int(contact_steps)
        if goal_frame not in ("palm", "world"):
            raise ValueError(f"goal_frame is 'palm' or 'world', got {goal_frame!r}")
        self.goal_frame = goal_frame
        self.goal_fixed = goal_pose is not None
        self.goal_quat = (_unit_quat(goal_pose, "goal_pose") if self.goal_fixed
                          else np.array([1.0, 0.0, 0.0, 0.0]))
        self.goal_dist = float(goal_dist)
        self.force_mag = float(force_mag)
        self.force_every = int(force_every)
        self.force_hold = int(force_hold)
        self.force_viz_scale = float(force_viz_scale)
        self.goal_viz_scale = float(goal_viz_scale)
        self.force_on = self.force_every > 0 and self.force_hold > 0
        self.max_rel_dist = float(max_rel_dist)
        self.vel_scale = float(vel_scale)
        self.timestep = float(timestep)
        self.ccd_tolerance = float(ccd_tolerance)
        self.goal_limit_s = float(goal_limit_s)
        self.total_limit_s = float(total_limit_s)
        self.check_finite = bool(check_finite)

        super().__init__(
            num_envs, device=device, viz_env_id=viz_env_id,
            max_episode_steps=int(round(total_limit_s / timestep)), debug=debug,
            seed=seed, phys_substeps=phys_substeps, render_mode=render_mode,
            use_graph=use_graph, njmax=njmax, nconmax=nconmax)

        self._cache_model_addresses()
        if self.color_deforms:                  # needs the pad table from _cache_model_addresses
            self._painter = PadPainter(
                self.m, self._pads, self.hfield_offset_fraction,
                scale=self.color_scale, interp=self.color_interp)
        self._alloc_device_state()
        self._build_kernel_params()
        self._make_spaces()
        self._warn_shape_limits()

        # Step program: the RNG tick and the deform (skipped on device when design is unchanged).
        # Post-step task kernels run inside the captured graph; finished worlds respawn outside it.
        self.register_step_launch(self._launch_head)
        if self.force_on:
            self.register_step_launch(self._launch_force)
        self.register_step_launch(self._launch_ctrl)
        self.register_post_step_launch(self._launch_task_step)

    # ------------------------------------------------------------------ #
    # Model assembly
    # ------------------------------------------------------------------ #
    def _geom_densities(self, m: mujoco.MjModel) -> np.ndarray:
        """Each geom's density from the spec (the compiled model drops it), NaN for
        a geom with an explicit mass. find_segments masses box links with it."""
        spec = self._build_world_spec()
        check = spec.compile()
        if check.ngeom != m.ngeom or not np.array_equal(check.geom_type, m.geom_type) or not np.allclose(check.geom_size, m.geom_size):
            raise RuntimeError("the spec's geoms do not line up with the compiled model's")
        return np.asarray([np.nan if not np.isnan(g.mass) else float(g.density) for g in spec.geoms], dtype=np.float64)

    def _build_base_spec(self) -> mujoco.MjSpec:
        """One world's scene. The per-world heightfield copies are made on the
        device, after the upload (_fill_world_rows)."""
        return self._build_world_spec()

    def _build_world_spec(self) -> mujoco.MjSpec:
        """Override hook: one world's scene, the hand on its frame, the object and the scene."""
        hand_xml, object_xml, scene_xml = self._xmls
        hand = mujoco.MjSpec.from_file(str(_asset(hand_xml, _HAND_DIR)))
        hand.compile()   # an attached child's keyframes are namespaced correctly only if compiled first

        obj = mujoco.MjSpec.from_file(str(_object_asset(object_xml)))
        bodies = obj.worldbody.bodies
        if len(bodies) != 1:
            raise RuntimeError(
                f"{object_xml}: an object model holds exactly one top-level "
                f"body, this one has {len(bodies)}")
        self._obj_body = bodies[0].name

        spec = mujoco.MjSpec.from_file(str(_asset(scene_xml, _SCENE_DIR)))
        hand_frame = spec.worldbody.add_frame()
        hand_frame.quat = np.asarray(self.HAND_QUAT, dtype=np.float64)
        spec.attach(hand, prefix="", frame=hand_frame)
        spec.attach(obj, prefix="", frame=spec.worldbody.add_frame())
        spec.option.timestep = self.timestep
        # Subdivided pads compile to the MJCF's own relief (the spline is linear)
        if self.grid_subd > 1:
            subdivide_hfields(spec, self.grid_subd, self.spline_degree)
        spec.option.ccd_tolerance = self.ccd_tolerance
        if self.color_deforms:                  # after the attach, so spec.hfields has the pads
            add_pad_textures(spec)
        return spec

    def _fill_world_rows(self) -> None:
        # A copy of every heightfield per world on the device, each world's pads
        # drawing its own, viewed as one row per world (zero-copy)
        n, m = self.num_envs, self.m
        self._hfield_data, self._hfield_size = tile_hfields(self.gpu_model, m, n)
        self.gpu_model.geom_dataid.assign(world_geom_dataid(m, n))
        # overflow_reports says once at build which pads can overflow a pair's
        # contact budget; the per-step print is noise. The flag is baked into the
        # kernels when first built, so clear it before the first forward
        self.gpu_model.opt.warn_overflow &= ~mjw.OverflowType.HFIELD

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _cache_model_addresses(self) -> None:
        """Cache model ids and addresses, the hand's joints and the spawn point,
        find the embodiment parts, and size the embodiment action."""
        m = self.m
        self._palm_bid = m.body(PALM_BODY).id
        self._obj_bid = m.body(self._obj_body).id

        free = [j for j in range(m.njnt)
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                and int(m.jnt_bodyid[j]) == self._obj_bid]
        if len(free) != 1:
            raise RuntimeError(
                f"body {self._obj_body!r} has {len(free)} free joints;"
                "expected one")
        self._obj_qadr = int(m.jnt_qposadr[free[0]])
        self._obj_dadr = int(m.jnt_dofadr[free[0]])

        # Every joint but the object's is the hand's. Limited joints are scaled by range
        qadr, dadr, scale = [], [], []
        for j in range(m.njnt):
            if j == free[0]:
                continue
            nq, nv = _joint_widths(m.jnt_type[j])
            a, d = int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
            qadr += list(range(a, a + nq))
            dadr += list(range(d, d + nv))
            span = (float(np.abs(m.jnt_range[j]).max())
                    if m.jnt_limited[j] else 1.0)
            scale += [1.0 / max(span, 1e-6)] * nq
        self._hand_qadr = np.asarray(qadr, dtype=np.int64)
        self._hand_dadr = np.asarray(dadr, dtype=np.int64)
        self._hand_qscale = np.asarray(scale, dtype=np.float32)

        self._init_qpos = self._reset_pose(m)

        d = mujoco.MjData(m)
        d.qpos[:] = self._init_qpos
        mujoco.mj_forward(m, d)
        self._palm_pos = np.asarray(d.xpos[self._palm_bid], dtype=np.float64)
        self._palm_quat0 = np.asarray(d.xquat[self._palm_bid], dtype=np.float64)
        self._palm_root = int(m.body_rootid[self._palm_bid])
        self._obj_root = int(m.body_rootid[self._obj_bid])
        self._obj_reach = max(
            (float(m.geom_rbound[g]) for g in range(m.ngeom)
             if int(m.geom_bodyid[g]) == self._obj_bid), default=0.0)
        self._spawn_pos = self._spawn_point(m, d)

        # The embodiment action: the pads' cells first, then the links' entries
        self._pads = (find_pads(m, subd=self.grid_subd) if self.deform_hfield
                      else empty_pads())
        self.n_cell = n_cells(self._pads)
        self.n_pad = len(self._pads["hfield"])
        if self.deform_links:
            self._segs = find_segments(m, self.min_link_length, self._geom_densities(m))
            self.n_links = len(self._segs["body"])
            self.n_slots = n_slots(self._segs)
            self._segs["axis_act"] = self._segs["axis_act"] + self.n_cell   # links after the pads
        else:
            self._segs = _empty_segs()
            self.n_links = 0
            self.n_slots = 0
        self.n_offsets = self.n_cell + self.n_slots

    def _spawn_point(self, m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
        # Over the palm, above its tallest geom by the object's reach, so no
        # orientation starts touching a pad
        top = max((_geom_top(m, d, g) for g in range(m.ngeom)
                   if int(m.geom_bodyid[g]) == self._palm_bid),
                  default=float(self._palm_pos[2]))
        return np.array([self._palm_pos[0], self._palm_pos[1],
                         top + self._obj_reach + _SPAWN_CLEARANCE])

    def _reset_pose(self, m: mujoco.MjModel) -> np.ndarray:
        """qpos0 with every ``init_key`` keyframe applied, matched by suffix.

        Attaching fills a keyframe with qpos0 for the joints its model does not
        own, so each keyframe's difference from qpos0 is exactly its own part,
        and the keyframes combine by adding these differences. Two keyframes
        setting the same joint raise.
        """
        keys = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, i)
                for i in range(m.nkey)]
        qpos, claimed = np.array(m.qpos0, dtype=np.float64), np.zeros(m.nq, bool)
        for name in self.init_key:
            for i, key in enumerate(keys):
                if key != name and not key.endswith("_" + name):
                    continue
                delta = m.key_qpos[i] - m.qpos0
                mine = np.abs(delta) > 1e-9
                if (mine & claimed).any():
                    raise RuntimeError(
                        f"keyframe {key!r} rewrites qpos "
                        f"{np.flatnonzero(mine & claimed).tolist()}, which an "
                        f"earlier keyframe in {list(self.init_key)} already set")
                qpos, claimed = qpos + delta, claimed | mine
        return qpos.astype(np.float32)

    def _alloc_device_state(self) -> None:
        # The kernel argument structs hold warp views of most of these buffers,
        # so those must be written in place, never rebound
        n, dev = self.num_envs, self.device
        self._alloc_action_buffers(self.act_dim(), self.n_offsets)
        self._offsets_t = torch.zeros((n, self.n_offsets), device=dev)
        # A rigid hand has no embodiment action, but the kernels still need arrays
        # to bind, so give them a one-column dummy (no kernel reads it)
        self._deform_dev = (self._reset_act_buf if self.n_offsets
                            else torch.zeros((n, 1), device=dev))
        self._offsets_dev = (self._offsets_t if self.n_offsets
                             else torch.zeros((n, 1), device=dev))
        self._tick = torch.zeros(1, dtype=torch.int32, device=dev)
        self._lane_seed = torch.full((n,), int(self.seed), dtype=torch.int32, device=dev)
        self._ep_seed = torch.zeros(n, dtype=torch.int32, device=dev)
        self._tunables = torch.zeros(kern.N_TUNABLES, device=dev)
        self._wp_tunables = wp.from_torch(self._tunables)
        self._qpos_init_t = torch.as_tensor(self._init_qpos, device=dev)

        i32 = dict(dtype=torch.int32, device=dev)
        self._force_dir = torch.zeros((n, 2), device=dev)
        self._force_phase = torch.zeros(n, **i32)
        self._force_cycle = torch.zeros(n, **i32)
        self._deform_dirty = torch.ones(1, **i32)

        # Goal and object state, written by the kernels
        self._goal_quat_t = torch.zeros((n, 4), device=dev)
        self._goal_obs_t = torch.zeros((n, 4), device=dev)   # in the palm's frame
        self._goal_t0 = torch.zeros(n, device=dev)
        self._dwell = torch.zeros(n, **i32)
        self._scored = torch.zeros(n, **i32)
        self._rebase = torch.zeros(n, **i32)
        self._dist = torch.zeros(n, device=dev)
        self._reached = torch.zeros(n, device=dev)
        self._paid = torch.zeros(n, **i32)
        self._elapsed = torch.zeros(n, device=dev)
        self._gate = torch.zeros(n, **i32)
        self._obj_pos = torch.zeros((n, 3), device=dev)
        self._obj_quat = torch.zeros((n, 4), device=dev)
        self._obj_vel = torch.zeros((n, 3), device=dev)
        self._obj_angvel = torch.zeros((n, 3), device=dev)
        self._obj_off_w = torch.zeros((n, 3), device=dev)
        self._pos_dist = torch.zeros(n, device=dev)

        # Contact memory, see ContactArrays
        for name in ("_raw_hand", "_raw_world", "_hist_hand", "_hist_world"):
            setattr(self, name, torch.zeros(n, **i32))
        self._touch_hand = torch.zeros(n, device=dev)
        self._touch_world = torch.zeros(n, device=dev)
        self._mult = torch.zeros(n, device=dev)
        body = np.asarray(self.m.geom_bodyid)
        kind = np.full(self.m.ngeom, kern.KIND_HAND, dtype=np.int32)
        kind[body == self._obj_bid] = kern.KIND_OBJECT
        kind[body == 0] = kern.KIND_WORLD
        self._geom_kind = torch.as_tensor(kind, device=dev)

        # Host-side shaping state: the distance to the goal a step ago, and the
        # angle the goal was set at
        self._prev_dist = torch.zeros(n, device=dev)
        self._dist0 = torch.full((n,), self.goal_dist, device=dev)
        self._diverged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._diverged_total = torch.zeros((), dtype=torch.long, device=dev)
        self._goals_total = torch.zeros(n, dtype=torch.long, device=dev)

        self._qadr_t = torch.as_tensor(self._hand_qadr, device=dev)
        self._dadr_t = torch.as_tensor(self._hand_dadr, device=dev)
        self._qscale_t = torch.as_tensor(self._hand_qscale, device=dev)
        # One deform window per block in the embodiment action, pads first
        self._can_deform_t = torch.as_tensor(
            [w for w, on in ((self.deform_hfield_s, self.deform_hfield),
                             (self.deform_links_s, self.deform_links)) if on],
            dtype=torch.float32, device=dev)

    def _build_kernel_params(self) -> None:
        dev = self.device
        sim = kern.SimArrays()
        for attr in ("qpos", "qvel", "qacc", "qacc_warmstart", "time", "ctrl",
                     "xpos", "xquat", "xfrc_applied", "cvel", "subtree_com"):
            setattr(sim, attr, getattr(self.gpu_data, attr))
        self._sim = sim

        dfm = DeformArrays()
        for attr in self.PER_WORLD_MODEL_FIELDS:
            if attr != "geom_dataid":           # fixed at build
                setattr(dfm, attr, getattr(self.gpu_model, attr))
        dfm.hfield_size = self._hfield_size
        self._dfm = dfm

        env = kern.EnvArrays()
        env.mask = wp.from_torch(self.reset_mask)
        env.qpos_init = wp.from_torch(self._qpos_init_t)
        env.deform = wp.from_torch(self._deform_dev)
        env.ctrl_act = wp.from_torch(self._act_buf)
        env.ctrl_scale = wp.from_torch(self.ctrl_scale)
        env.ctrl_mid = wp.from_torch(self.ctrl_mid)
        env.offsets = wp.from_torch(self._offsets_dev)
        env.tick = wp.from_torch(self._tick)
        env.lane_seed = wp.from_torch(self._lane_seed)
        env.ep_seed = wp.from_torch(self._ep_seed)
        env.force_dir = wp.from_torch(self._force_dir)
        env.force_phase = wp.from_torch(self._force_phase)
        env.force_cycle = wp.from_torch(self._force_cycle)
        env.deform_dirty = wp.from_torch(self._deform_dirty)
        self._env = env

        self._seg = build_seg_arrays(self._segs, dev)
        self._hf = build_pad_arrays(self._pads, self._hfield_data, dev,
                                    self.num_envs, self.grid_subd,
                                    self.spline_degree,
                                    relief_weights(self.m, self._pads, self.relief_length))

        goal = kern.GoalArrays()
        goal.quat = wp.from_torch(self._goal_quat_t, dtype=wp.quat)
        goal.quat_obs = wp.from_torch(self._goal_obs_t, dtype=wp.quat)
        goal.obj_quat = wp.from_torch(self._obj_quat, dtype=wp.quat)
        for name, t in (("obj_pos", self._obj_pos), ("obj_vel", self._obj_vel),
                        ("obj_angvel", self._obj_angvel),
                        ("obj_off_w", self._obj_off_w)):
            setattr(goal, name, wp.from_torch(t, dtype=wp.vec3))
        for name, t in (("t0", self._goal_t0), ("dwell", self._dwell),
                        ("scored", self._scored), ("rebase", self._rebase),
                        ("dist", self._dist), ("reached", self._reached),
                        ("paid", self._paid), ("elapsed", self._elapsed),
                        ("gate", self._gate), ("pos_dist", self._pos_dist)):
            setattr(goal, name, wp.from_torch(t))
        self._goal = goal

        ct = kern.ContactArrays()
        for name in ("kind", "raw_hand", "raw_world", "hist_hand", "hist_world",
                     "touch_hand", "touch_world", "mult"):
            t = self._geom_kind if name == "kind" else getattr(self, "_" + name)
            setattr(ct, name, wp.from_torch(t))
        self._ct = ct
        self._bits = (1 << self.contact_steps) - 1

        c = kern.EnvConst()
        c.seed = int(self.seed)
        c.nq, c.nv, c.nu = int(self.m.nq), int(self.m.nv), int(self.m.nu)
        c.n_seg = int(self.n_links)
        c.n_pad = int(self.n_pad)
        c.n_cell = int(self.n_cell)
        c.obj_bid = int(self._obj_bid)
        c.obj_qadr = int(self._obj_qadr)
        c.obj_dadr = int(self._obj_dadr)
        c.obj_root = int(self._obj_root)
        c.palm_bid = int(self._palm_bid)
        c.palm_root = int(self._palm_root)
        c.palm_quat0 = wp.quat(*(float(v) for v in self._palm_quat0))
        c.spawn_pos = wp.vec3(*(float(v) for v in self._spawn_pos))
        c.init_quat = wp.quat(*(float(v) for v in self.init_quat))
        c.randomise_init = int(self.randomise_init_quat)
        c.goal_fixed = int(self.goal_fixed)
        c.goal_quat = wp.quat(*(float(v) for v in self.goal_quat))
        c.goal_world = int(self.goal_frame == "world")
        c.rot_tol = float(self.goal_rot_tol)
        c.pos_on = 0                            # a goal is an orientation only here
        c.rel_pos = wp.vec3(0.0, 0.0, 0.0)
        c.pos_tol = 0.0
        c.goal_pos_w = wp.vec3(0.0, 0.0, 0.0)
        c.dwell_limit = int(self.goal_dwell_steps)
        c.deform_hfield_s = float(self.deform_hfield_s)
        c.deform_links_s = float(self.deform_links_s)
        c.force_every = max(int(self.force_every), 1)   # force_kernel takes the phase modulo it
        c.force_hold = int(self.force_hold)
        self._consts = c

        self._refresh_tunables()

    def _refresh_tunables(self) -> None:
        vals = torch.zeros(kern.N_TUNABLES, dtype=torch.float32)
        vals[kern.T_MAX_LINK_OFFSET] = self.max_link_offset
        vals[kern.T_HFIELD_OFFSET_FRACTION] = self.hfield_offset_fraction
        vals[kern.T_FORCE_MAG] = self.force_mag
        vals[kern.T_GOAL_DIST] = self.goal_dist
        self._tunables.copy_(vals)

    def _warn_shape_limits(self) -> None:
        # Report where the configured ranges run into the model's limits
        hit = floor_reports(self.m, self._segs, self.max_link_offset)
        if hit and not RotateInHandParallelEnv._length_floor_warned:
            RotateInHandParallelEnv._length_floor_warned = True   # once per process, the floor can be intended
            warnings.warn(
                f"max_link_offset={self.max_link_offset} can shorten "
                f"{len(hit)} link(s) past min_link_length="
                f"{self.min_link_length}, where the deform clamps them:\n  "
                + "\n  ".join(hit), RuntimeWarning, stacklevel=3)
        if self.debug:
            over = overflow_reports(self.m, self._pads, self._obj_bid)
            if over:
                warnings.warn(
                    f"heightfield contacts may overflow on {len(over)} pad(s). Contacts "
                    "past the cap are dropped, so coarsen the pad or split it into "
                    "several geoms:\n  " + "\n  ".join(over), UserWarning, stacklevel=3)

    # ------------------------------------------------------------------ #
    # Spaces
    # ------------------------------------------------------------------ #
    def act_dim(self) -> int:
        # Control action only - the embodiment action is a separate step() argument
        return int(self.m.nu)

    def obs_dim(self) -> int:
        # Hand qpos/qvel, object pos (3), quat (4), lin/ang vel (6), goal quat (4)
        # and angle (1), the can_deform flags, touching (1), extras, and the offsets
        return (len(self._hand_qadr) + len(self._hand_dadr) + 18
                + len(self._can_deform_t) + 1 + self.EXTRA_OBS + self.n_offsets)

    def _make_spaces(self) -> None:
        # The applied embodiment is observed, within its clip range
        low = np.full(self.obs_dim(), -1.0, dtype=np.float32)
        high = np.full(self.obs_dim(), 1.0, dtype=np.float32)
        if self.n_offsets:
            low[-self.n_offsets:], high[-self.n_offsets:] = self.offset_bounds
        self.single_observation_space = spaces.Box(
            low=low, high=high, dtype=np.float32, seed=self.seed)
        self.single_action_space = spaces.Box(
            low=np.full(self.act_dim(), -1.0, dtype=np.float32),
            high=np.full(self.act_dim(), 1.0, dtype=np.float32),
            dtype=np.float32, seed=self.seed)
        self.observation_space = batch_space(
            self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    @property
    def offset_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) of each embodiment entry: [0, 1] for pad cells, then the links'."""
        lo, hi = slot_bounds(self._segs, self.n_offsets)
        lo[:self.n_cell] = 0.0
        hi[:self.n_cell] = 1.0
        return lo, hi

    # ------------------------------------------------------------------ #
    # Kernel launches
    # ------------------------------------------------------------------ #
    def _launch_tick(self) -> None:
        wp.launch(kern.tick_kernel, dim=1, inputs=[self._env])   # advance the RNG tick

    def _launch_deform(self) -> None:
        wp.launch(kern.deform_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._dfm, self._seg, self._hf,
                          self._consts, self._wp_tunables])

    def _stage_actions(self, ctrl_actions, reset_actions) -> None:
        super()._stage_actions(ctrl_actions, reset_actions)
        if reset_actions is not None:
            self._deform_dirty.fill_(1)

    def _launch_head(self) -> None:
        """Head of the step program: the RNG tick, then the deform (when dirty), then clear dirty."""
        self._launch_tick()
        self._launch_deform()
        wp.launch(kern.clear_deform_dirty_kernel, dim=1, inputs=[self._env])

    def _launch_reset(self) -> None:
        """Explicit reset: bump the RNG tick, then respawn the masked worlds."""
        self._launch_tick()
        self._launch_respawn()

    def _launch_respawn(self) -> None:
        """Respawn the masked worlds, then deform every world to its current design."""
        wp.launch(kern.reset_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._goal, self._consts,
                          self._wp_tunables])
        self._launch_deform()

    def _launch_force(self) -> None:
        wp.launch(kern.force_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._consts, self._wp_tunables])

    def _launch_ctrl(self) -> None:
        wp.launch(kern.ctrl_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._consts])

    def _launch_task_step(self) -> None:
        """Run the post-physics task kernels inside the CUDA-graphed step program."""
        d = self.gpu_data
        wp.launch(kern.sample_kernel, dim=d.naconmax,
                  inputs=[d.nacon, d.contact.worldid, d.contact.geom,
                          self._ct, self._env.mask, 0])
        wp.launch(kern.window_kernel, dim=self.num_envs,
                  inputs=[self._env, self._goal, self._ct, self._bits,
                          self.TABLE_TOUCH, 0])
        wp.launch(kern.task_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._goal, self._consts,
                          self._wp_tunables, 0])

    def _launch_task(self, only_reset: bool = False) -> None:
        """Run the post-physics kernels on torch's stream: contacts first, since
        the task kernel reads the gate they set. ``only_reset`` limits them to the
        worlds in ``reset_mask``."""
        d = self.gpu_data
        only = int(only_reset)
        with self._wp_scope():
            wp.launch(kern.sample_kernel, dim=d.naconmax,
                      inputs=[d.nacon, d.contact.worldid, d.contact.geom,
                              self._ct, self._env.mask, only])
            wp.launch(kern.window_kernel, dim=self.num_envs,
                      inputs=[self._env, self._goal, self._ct, self._bits,
                              self.TABLE_TOUCH, only])
            wp.launch(kern.task_kernel, dim=self.num_envs,
                      inputs=[self._env, self._sim, self._goal, self._consts,
                              self._wp_tunables, only])

    # ------------------------------------------------------------------ #
    # Obs / reward
    # ------------------------------------------------------------------ #
    def _compute_obs(self, only_reset: bool = False, from_step: bool = False) -> torch.Tensor:
        if not from_step:
            self._launch_task(only_reset)
        qpos, qvel = self.qpos_torch(), self.qvel_torch()
        blocks = [
            qpos[:, self._qadr_t] * self._qscale_t,
            qvel[:, self._dadr_t] / self.vel_scale,
            self._obj_pos / self.max_rel_dist,
            self._obj_quat.clamp(-1.0, 1.0),     # float32 can land 1 ulp past 1
            self._obj_vel / self.vel_scale,
            self._obj_angvel / self.vel_scale,
            self._goal_obs_t.clamp(-1.0, 1.0),
            (self._dist / _PI).unsqueeze(1),
        ]
        if len(self._can_deform_t):
            blocks.append(self._can_deform_flags())
        blocks.append(self._touch_hand.unsqueeze(1))
        blocks += self._extra_obs()
        blocks.append(self._offsets_t)
        obs = torch.cat(blocks, dim=1)
        if self.check_finite:
            self._assert_finite(obs, "obs")
        # A diverged world never meets a terminal condition (comparisons with NaN
        # are False), so flag it for _reward_terminated_truncated to terminate,
        # and return a zero placeholder row
        self._diverged = ~torch.isfinite(obs).all(dim=1)
        self._diverged_total += (self._diverged & self.reset_mask if only_reset else self._diverged).sum()
        return obs.masked_fill_(self._diverged[:, None], 0.0)

    def _extra_obs(self) -> list[torch.Tensor]:
        """Override hook: ``EXTRA_OBS`` columns appended before the offsets."""
        return []

    def _can_deform_flags(self) -> torch.Tensor:
        """1 while a block's deform window is open, per block in the embodiment
        action. A block deformed only at respawn reads 0 from the first obs."""
        t = wp.to_torch(self.gpu_data.time).unsqueeze(1)
        return (t < self._can_deform_t).float()

    def sample_observations(self, n: int = 1) -> torch.Tensor:
        """``n`` episode-start observations, from resets of the whole batch. Use a
        separate env."""
        rows, collected = [], 0
        while collected < int(n):
            rows.append(self.reset().clone())
            collected += rows[-1].shape[0]
        return torch.cat(rows)[:int(n)]

    def _rebase_episode(self, mask: torch.Tensor) -> None:
        """Start the shaping of the worlds in ``mask`` from the pose they respawned in, and clear their goal count.
        Reads the distances the respawn's ``_compute_obs`` wrote."""
        self._prev_dist[mask] = self._dist[mask]
        self._dist0[mask] = self._dist[mask].clamp(min=self.goal_rot_tol)
        self._goals_total[mask] = 0

    def _reward_terminated_truncated(self):
        # Written by task_kernel during _compute_obs. reached and paid refer to the
        # goal in force when the step began, dist to the one in force now
        reached, scored = self._reached, self._scored > 0
        respawned = self.reset_mask
        redrawn = (self._rebase > 0) & ~respawned   # task_kernel clears rebase, so respawns come from reset_mask
        live = self._shaping_live(respawned)

        # Potential-based shaping: the step's fall in angle, relative to the angle
        # the goal was set at, so one full rotation gives max_pose_return. Inside
        # the tolerance reached is 0, so the last leg is given, and leaving takes it back
        gain, prev = self._step_gain(self._prev_dist, reached, live)
        r_pose = self._pose_reward(
            gain / self._dist0 * self.max_pose_return, live)
        r_goal = torch.where(self._paid > 0, self.max_goal_return
                             / self.goal_dwell_steps, 0.0)
        reward = r_pose + r_goal + self._extra_reward(respawned)

        # Only now does a new goal become the one measured against
        fresh = ~live | redrawn
        self._dist0 = torch.where(
            fresh, self._dist.clamp(min=self.goal_rot_tol), self._dist0)
        self._prev_dist = torch.where(fresh, self._dist, prev)

        # Cleared before adding, so a goal met on the respawn step still counts
        self._goals_total = torch.where(
            respawned, torch.zeros_like(self._goals_total),
            self._goals_total) + scored.long()

        dropped = self._object_lost()
        # Only random goals can stall, a fixed goal's clock is total_limit_s
        stalled = (torch.zeros_like(dropped) if self.goal_fixed
                   else self._elapsed > self.goal_limit_s)

        # Dropping, stalling, divergence and (for a fixed goal) meeting it are true
        # terminals. total_limit_s is the one truncation
        terminated = dropped | stalled | self._diverged
        if self.goal_fixed:
            terminated = terminated | scored
        truncated = (self.world_steps >= self.max_episode_steps) & ~terminated
        if self.check_finite:
            self._assert_finite(reward, "reward")
        reward = reward.masked_fill(self._diverged, 0.0)

        self.last_scored = scored
        self.last_dropped = dropped
        self.last_stalled = stalled
        self.last_time_limit = truncated
        self.last_diverged = self._diverged
        return reward, terminated, truncated

    def _shaping_live(self, respawned: torch.Tensor) -> torch.Tensor:
        """Override hook: worlds whose shaping runs this step. Called once per step."""
        return ~respawned

    def _extra_reward(self, respawned: torch.Tensor) -> torch.Tensor | float:
        """Override hook: reward added on top. Called once per step."""
        return 0.0

    def _pose_reward(self, progress: torch.Tensor,
                     live: torch.Tensor) -> torch.Tensor:
        """Override hook: the shaping reward. A gain is given times the touch
        multiplier, a loss in full, so over any loop the gains are at most the
        losses."""
        return torch.where(progress > 0, self._mult * progress, progress)

    @staticmethod
    def _step_gain(prev: torch.Tensor, now: torch.Tensor, live: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """(gain, new previous) of a distance: ``prev - now`` in the ``live``
        worlds, negative where it grew, 0 elsewhere; and a copy of ``now``, which
        may be a buffer a kernel rewrites in place."""
        return torch.where(live, prev - now, 0.0), now.clone()

    def _object_lost(self) -> torch.Tensor:
        """Override hook: worlds whose object fell half a metre below the palm."""
        return self._obj_off_w[:, 2] < -0.5

    @property
    def is_touching(self) -> torch.Tensor:
        """This step's reward multiplier per world: 1 or 0 (0.5 with ``TABLE_TOUCH``)."""
        return self._mult

    @property
    def goals_reached(self) -> torch.Tensor:
        """Goals met this episode, per world."""
        return self._goals_total

    @property
    def diverged_worlds(self) -> int:
        """Worlds ended so far by non-finite physics, should stay 0. Syncs with the device."""
        return int(self._diverged_total)

    # ------------------------------------------------------------------ #
    # Live tunables + host-side views
    # ------------------------------------------------------------------ #
    def change_max_link_offset(self, new_offset: float) -> None:
        """Set the metres per unit link embodiment action."""
        self.max_link_offset = float(new_offset)
        self._refresh_tunables()
        self._deform_dirty.fill_(1)

    def change_hfield_offset_fraction(self, new_fraction: float) -> None:
        """Set the share of its pad's range a cell at 1 rises to, from the next pad write."""
        _check_fraction(new_fraction)
        self.hfield_offset_fraction = float(new_fraction)
        if self._painter is not None:
            self._painter.fraction = self.hfield_offset_fraction
        self._refresh_tunables()
        self._deform_dirty.fill_(1)

    def change_goal_dist(self, new_dist: float) -> None:
        """Set the angle [rad] of new random goals, from the next one drawn."""
        self.goal_dist = float(new_dist)
        self._refresh_tunables()

    def change_force_mag(self, new_mag: float) -> None:
        """Set the perturbation force magnitude [N] (0 to disable)."""
        self.force_mag = float(new_mag)
        self._refresh_tunables()

    @property
    def hfield_ids(self) -> np.ndarray:
        """Every hfield a deform can change (pad reliefs and box-grown pad sizes)."""
        return np.union1d(self._pads["hfield"], self._segs["pad_hfield"])

    @property
    def env_offsets(self) -> np.ndarray:
        """Host copy of each world's applied embodiment in metres, each block on its own scale."""
        reach = np.repeat(np.asarray(self._pads["range"], dtype=np.float32),
                          np.asarray(self._pads["ncell"], dtype=np.int64))
        scale = np.concatenate([
            reach * np.float32(self.hfield_offset_fraction),
            np.full(self.n_slots, self.max_link_offset, dtype=np.float32)])
        return self.env_offsets_norm * scale

    @property
    def env_offsets_norm(self) -> np.ndarray:
        """Host copy of each world's applied embodiment: cells in [0, 1], links in [-1, 1]."""
        return self._offsets_t.detach().cpu().numpy()

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _init_render_model(self, m: mujoco.MjModel) -> None:
        if self.n_links:                        # else CPU kinematics draws turned links unturned
            release_geom_frames(m, self._segs)
        # Paint once here, since the per-frame repaint only fires when the relief
        # changes. No upload needed before the GL context exists
        if self._painter is not None:
            self._painter.paint(m)

    def _mirror_render_model(self, m: mujoco.MjModel, env_idx: int) -> None:
        """Copy world ``env_idx``'s deformed links and pads into the render copy
        ``m``, re-uploading pads only if their size or relief changed (each upload
        is a blocking round trip to the viewer's render thread)."""
        w, v, gm = int(env_idx), m, self.gpu_model
        reupload = False
        if self.n_links:
            g = self._segs["geom"]
            b = self._segs["body"]
            kid = self._segs["child_id"]
            v.geom_pos[g] = wp.to_torch(gm.geom_pos)[w, g].cpu().numpy()
            v.geom_quat[g] = wp.to_torch(gm.geom_quat)[w, g].cpu().numpy()
            v.body_iquat[b] = wp.to_torch(gm.body_iquat)[w, b].cpu().numpy()
            v.body_ipos[b] = wp.to_torch(gm.body_ipos)[w, b].cpu().numpy()
            v.geom_size[g] = wp.to_torch(gm.geom_size)[w, g].cpu().numpy()
            v.geom_rbound[g] = wp.to_torch(gm.geom_rbound)[w, g].cpu().numpy()
            v.geom_aabb[g] = (wp.to_torch(gm.geom_aabb)[w, g]
                              .reshape(len(g), 6).cpu().numpy())
            if len(kid):
                v.body_pos[kid] = wp.to_torch(gm.body_pos)[w, kid].cpu().numpy()
            pad = self._segs["pad_geom"]
            if len(pad):
                v.geom_rbound[pad] = wp.to_torch(gm.geom_rbound)[w, pad].cpu().numpy()
                v.geom_aabb[pad] = (wp.to_torch(gm.geom_aabb)[w, pad]
                                    .reshape(len(pad), 6).cpu().numpy())
                size = wp.to_torch(self._hfield_size)[w].cpu().numpy()
                if not np.array_equal(v.hfield_size, size):
                    v.hfield_size[:] = size
                    reupload = True
            # The visualizer culls against bvh_aabb, fixed at compile time. For a
            # one-geom body, the body's box is that geom's aabb
            for bid, gid in zip(b, g):
                adr = int(v.body_bvhadr[bid])
                if (adr >= 0 and v.body_bvhnum[bid] == 1
                        and v.body_geomnum[bid] == 1):
                    v.bvh_aabb[adr] = v.geom_aabb[gid]
        if self.n_cell:
            row = wp.to_torch(self._hfield_data)[w].cpu().numpy()
            if not np.array_equal(v.hfield_data, row):
                v.hfield_data[:] = row
                if self._painter is not None:
                    self._painter.paint(v)
                reupload = True
        if reupload:
            self._push_hfields(v)

    def _push_hfields(self, m: mujoco.MjModel) -> None:
        """Re-upload pads and their textures to the GL context drawing ``m``. The
        passive viewer's context is out of reach, so its copy is only flagged and
        ``mjview_utils.sync_viewer`` uploads."""
        if m is self._render_m and self.renderer is not None:
            con = getattr(self.renderer, "_mjr_context", None)
            if con is not None:
                for h in self.hfield_ids:
                    mujoco.mjr_uploadHField(m, con, int(h))
                for t in self.pad_color_tex_ids:
                    mujoco.mjr_uploadTexture(m, con, int(t))
            return
        self._viewer_hfields_dirty = True

    @property
    def pad_color_tex_ids(self) -> tuple[int, ...]:
        """The pads' colour textures, empty unless ``color_deforms`` is on."""
        return () if self._painter is None else self._painter.tex_ids

    def _add_debug_visualizations(self, scene: mujoco.MjvScene,
                                  env_idx: int) -> None:
        """Draw the object's frame and the goal's (longer, translucent), from world
        poses on the device, ``goal_viz_scale`` times the default length. A
        palm-frame goal is drawn at ``_goal_frame_origin``, carried by the palm;
        a world-frame one at that point for the palm's reset pose, where it stays."""
        w = int(env_idx)
        xpos = wp.to_torch(self.gpu_data.xpos)[w]
        xquat = wp.to_torch(self.gpu_data.xquat)[w]
        origin = xpos[self._obj_bid].cpu().numpy().astype(np.float64)
        obj_q = xquat[self._obj_bid].cpu().numpy().astype(np.float64)
        palm_q = xquat[self._palm_bid].cpu().numpy().astype(np.float64)
        palm_p = xpos[self._palm_bid].cpu().numpy().astype(np.float64)
        goal_q = self._goal_quat_t[w].detach().cpu().numpy().astype(np.float64)
        if self.goal_frame == "palm":
            goal_p = self._goal_frame_origin(palm_p, palm_q, origin)
            mujoco.mju_mulQuat(goal_q, palm_q, goal_q.copy())
        else:
            goal_p = self._goal_frame_origin(self._palm_pos, self._palm_quat0,
                                             self._spawn_pos)
        arm = max(self._obj_reach, 1e-3) * 2.0
        axis_len = arm * self.goal_viz_scale
        for quat, at, length, alpha, width in (
                (obj_q, origin, axis_len, 1.0, 0.10),
                (goal_q, goal_p, axis_len * 1.6, 0.4, 0.06)):
            mat = _quat_mat(quat)
            for axis in range(3):
                if scene.ngeom >= scene.maxgeom:
                    return
                rgba = np.zeros(4, dtype=np.float32)
                rgba[axis], rgba[3] = 1.0, alpha
                gm = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    gm, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                    np.zeros(3), np.eye(3).flatten(), rgba)
                mujoco.mjv_connector(
                    gm, mujoco.mjtGeom.mjGEOM_CAPSULE, axis_len * width,
                    at, at + mat[:, axis] * length)
                scene.ngeom += 1

    def _goal_frame_origin(self, palm_pos: np.ndarray, palm_quat: np.ndarray,
                           obj_pos: np.ndarray) -> np.ndarray:
        """Override hook: where the goal frame is drawn, the object for an
        orientation-only goal."""
        return obj_pos

    def _setup_camera(self, camera: mujoco.MjvCamera) -> None:
        """Look at the spawn point from eight times its distance to the palm."""
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = self._spawn_pos
        camera.elevation = -20.0
        camera.azimuth = 120.0
        camera.distance = 8.0 * max(
            float(np.linalg.norm(self._spawn_pos - self._palm_pos)), 0.02)
