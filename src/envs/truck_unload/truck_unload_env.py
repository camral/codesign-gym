"""Truck unloader: a loader robot empties a container of coloured boxes onto a
belt, and stacker robots deliver their colour from the belt to their own goal,
with a different body design in every world.

Every world runs the same scene: a container on a truck bed with a ramp, the
loader in front of it, a conveyor belt running away from the loader, and
``n_stackers`` stackers along the belt, each with a platform, shelf or floor
area to deliver to. Each robot is an arm, optionally on a mobile base, and has a
box-shaped field read from a ``bounds`` site in its MJCF, which places the
robot and decides which boxes it is responsible for. Each robot is rewarded per
box for reaching it, carrying it to its goal, releasing it, and leaving it there.
A lost box, an idle line, or a robot leaving its field ends the episode.

Code layout:

- truck_unload_env.py: scene assembly (``build_scene``), buffer and kernel-argument
setup, spaces, observation, reward and termination, and rendering;
- codesign_env.py: the co-design wrapper;
- truck_unload_kernels.py: the device work, per step and after physics;
- container.py, conveyor_belt.py, stacking_platform.py: the scene parts;
- offsettable_logic.py: the link naming convention and the link deform;
- parallel_env.py: the batched base class and its step program.

``TruckUnloadParallelEnv`` is the native batched env, and
``TruckUnloadCoDesignEnv`` in codesign_env.py puts the co-design contract on top of it.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import mujoco
import numpy as np
import torch
import warp as wp
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from . import truck_unload_kernels as kern
from .container import BOX_RGB, COLOR_NAMES, Container, add_box, add_container
from .conveyor_belt import Conveyor, add_conveyor, add_divider
from .offsettable_logic import (DeformArrays, build_seg_arrays, find_segments,
                                floor_reports, length_safe_lo, n_slots,
                                release_geom_frames, slot_bounds)
from .parallel_env import ParallelMujocoEnv
from .stacking_platform import Platform, add_floor_area, add_platform

_ASSETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))

# A box site on the worldbody: a robot's field, without mass or contacts
BOUNDS_SITE = "bounds"
BASE_BODY = "base"                         # body of a mobile base the arm is mounted on
TIP_SITE = "arm_tip"
GRASP_ACT = "grasp_act"                    # grasp actuator, ctrlrange starting at 0
GRASP_GEOM = "grasp_geom_"                 # gripper contact faces, grasp_geom_<n>

_RGBA_LOADER_FIELD = (0.3, 0.6, 1.0, 0.1)
_RGBA_STACKER_FIELD = (1.0, 0.7, 0.2, 0.1)

_CAMERA_MODES = ("fixed", "track")
_TARGETS = ("platform", "shelf", "floor")

# (nq, nv) per joint type, for slicing a robot's joints out of qpos/qvel
_JNT_DIMS = {int(mujoco.mjtJoint.mjJNT_FREE): (7, 6),
             int(mujoco.mjtJoint.mjJNT_BALL): (4, 3),
             int(mujoco.mjtJoint.mjJNT_SLIDE): (1, 1),
             int(mujoco.mjtJoint.mjJNT_HINGE): (1, 1)}


@dataclass
class Layout:
    """Every extent the scene was laid out with, and the box table it implies."""
    z_playing_field: float
    loader_bounds: tuple[np.ndarray, np.ndarray]
    container: Container
    conveyor: Conveyor
    field_lo: np.ndarray            # (n_robots, 3), robot 0 the loader
    field_hi: np.ndarray
    box_names: list[str]
    box_color: np.ndarray
    box_pos: np.ndarray
    n_grid: int                     # boxes in the container, the rest on the belt
    loader_order: np.ndarray
    goal_sites: list[str]
    stacker_bounds: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    platforms: list[Platform] = field(default_factory=list)


def _box_extents(model: mujoco.MjModel, name: str,
                 source: str) -> tuple[np.ndarray, np.ndarray]:
    """(centre, world half-extents) of the box site ``name`` on the worldbody.
    The half-extents are the AABB of the rotated box."""
    try:
        site = model.site(name)
    except KeyError:
        raise ValueError(f"{source} has no {name!r} site") from None
    if site.type != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError(f"{source}: site {name!r} must be a box, got type {site.type}")
    if int(site.bodyid[0]) != 0:
        raise ValueError(f"{source}: site {name!r} must hang off the worldbody, "
                         f"not body {model.body(int(site.bodyid[0])).name!r}")
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(site.quat, dtype=float))
    return (np.asarray(site.pos, dtype=float),
            np.abs(mat.reshape(3, 3)) @ np.asarray(site.size))


def _load_robot(arm_xml: str, base_xml: str | None, z_playing_field: float):
    """Load one robot's specs: ``((base, arm), bounds centre, bounds half-extents,
    mount height)``.

    An arm alone stands on the playing field and its own bounds count. On a base,
    the base's bounds count and the arm's are deleted, since they would move with
    the base and clash by name.
    """
    arm = mujoco.MjSpec.from_file(os.path.join(_ASSETS, arm_xml))
    if base_xml is None:
        pos, half = _box_extents(arm.compile(), BOUNDS_SITE, arm_xml)
        return (None, arm), pos, half, float(z_playing_field)
    arm_bounds = arm.site(BOUNDS_SITE)
    if arm_bounds is not None:
        arm.delete(arm_bounds)
    arm.compile()   # an attached child's keyframes are namespaced correctly only if compiled first
    base = mujoco.MjSpec.from_file(os.path.join(_ASSETS, base_xml))
    if base.body(BASE_BODY) is None:
        raise ValueError(f"{base_xml} has no {BASE_BODY!r} body to mount "
                         f"{arm_xml} on")
    pos, half = _box_extents(base.compile(), BOUNDS_SITE, base_xml)
    return (base, arm), pos, half, 0.0


def _attach_robot(spec: mujoco.MjSpec, parts, prefix: str, origin) -> None:
    """Attach a robot from ``_load_robot`` at ``origin``. The base and the arm go
    straight into the scene, since nesting the arm in the base spec would merge
    its keyframes into the loader's."""
    base, arm = parts
    frame = spec.worldbody.add_frame(pos=list(origin))
    if base is None:
        spec.attach(arm, prefix=prefix, frame=frame)
        return
    spec.attach(base, prefix=prefix, frame=frame)
    spec.attach(arm, prefix=prefix,
                frame=spec.body(prefix + BASE_BODY).add_frame())


def _add_field(spec: mujoco.MjSpec, name: str, lower, upper, rgba) -> None:
    """Draw a robot's field as a visual-only box site."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    spec.worldbody.add_site(
        name=name, type=mujoco.mjtGeom.mjGEOM_BOX, group=3,
        pos=((lower + upper) / 2.0).tolist(),
        size=((upper - lower) / 2.0).tolist(), rgba=list(rgba))


def _check_visual_meshes(spec: mujoco.MjSpec) -> None:
    """Raise if a visual mesh geom (no contacts) changes a body's mass or inertia."""
    mesh = mujoco.mjtGeom.mjGEOM_MESH
    visual = lambda g: g.type == mesh and g.contype == 0 and g.conaffinity == 0
    if not any(visual(g) and g.mass != 0 for g in spec.geoms):
        return
    massless = spec.copy()
    for g in massless.geoms:
        if visual(g):
            g.mass = 0.0
    ref, zero = spec.copy().compile(), massless.compile()
    bad = [ref.body(b).name for b in range(ref.nbody)
           if not (np.allclose(ref.body_mass[b], zero.body_mass[b])
                   and np.allclose(ref.body_inertia[b], zero.body_inertia[b])
                   and np.allclose(ref.body_ipos[b], zero.body_ipos[b]))]
    if bad:
        raise ValueError(f"visual mesh geoms change the mass or inertia of bodies {bad}. "
                         f'Give each visual mesh geom mass="0".')


def _serpentine(box_array) -> np.ndarray:
    """Order the loader empties the grid in: x slabs from the opening inwards,
    rows from the top down, alternating y within a row."""
    nx, ny, nz = (int(n) for n in box_array)
    order, flip = [], False
    for ix in range(nx - 1, -1, -1):
        for iz in range(nz - 1, -1, -1):
            ys = range(ny) if flip else range(ny - 1, -1, -1)
            order += [ix * ny * nz + iy * nz + iz for iy in ys]
            flip = not flip
    return np.asarray(order, dtype=np.int64)


def build_scene(
    *,
    loader_xml: str,
    loader_base_xml: str | None,
    stacker_xml: str,
    stacker_base_xml: str | None,
    scene_xml: str,
    n_stackers: int,
    platform_walls: bool,
    stacker_targets,
    stacker_colors,
    floor_area_n: int,
    container_opening: str,
    baseplate_dims,
    ramp_tilt_degs: float,
    ramp_lip: float,
    ramp_x_offset: float,
    container_z_offset: float,
    z_playing_field: float,
    belt_velocity: float,
    divider_dims,
    wall_thickness: float,
    box_size: float,
    box_array,
    box_clearance: float,
    box_mass: float,
    field_clearance: float,
    stacker_field_reach: float,
    timestep: float,
    rng: np.random.Generator,
) -> tuple[mujoco.MjSpec, Layout]:
    """Assemble the scene and return it with its ``Layout``.

    The loader stands at the origin and everything is laid out from its bounds:
    the container and belt run along x through their centre, with the ramp's
    lower edge on their -x face. Each stacker's bounds touch the belt's +y side
    and the previous field in x, with its delivery target behind it. One box per
    stacker starts on the belt, so stackers can learn without the loader.
    """
    z_pf = float(z_playing_field)
    palette = BOX_RGB if stacker_colors is None else [BOX_RGB[COLOR_NAMES[c]] for c in stacker_colors]

    loader, l_pos, l_half, l_mount = _load_robot(loader_xml, loader_base_xml, z_pf)
    loader_origin = [0.0, 0.0, l_mount]
    l_pos = l_pos + loader_origin
    l_lo, l_hi = l_pos - l_half, l_pos + l_half
    line_y = float(l_pos[1])

    spec = mujoco.MjSpec.from_file(os.path.join(_ASSETS, scene_xml))
    _attach_robot(spec, loader, "", loader_origin)

    container = add_container(
        spec, ramp_edge_x=float(l_lo[0]), center_y=line_y, z_playing_field=z_pf,
        box_size=box_size, box_array=box_array, box_clearance=box_clearance,
        box_mass=box_mass, wall_thickness=wall_thickness,
        opening=container_opening, baseplate_dims=baseplate_dims,
        ramp_tilt_degs=ramp_tilt_degs, ramp_lip=ramp_lip,
        ramp_x_offset=ramp_x_offset, z_offset=container_z_offset,
        n_colors=max(n_stackers, 1), palette=palette, rng=rng)

    belt_half_w = 0.75 * box_size
    belt_lo_y, belt_hi_y = line_y - belt_half_w, line_y + belt_half_w

    # Loader field: from the container's back wall to two boxes past its bounds
    field_lo = [np.array([container.lower[0], l_lo[1], z_pf - field_clearance])]
    field_hi = [np.array([l_hi[0] + 2.0 * box_size, l_hi[1], l_hi[2]])]

    bounds, platforms, goal_sites = [], [], ["loader_goal"]
    prev_field_x = float(field_hi[0][0])
    for i in range(n_stackers):
        st, s_pos, s_half, s_mount = _load_robot(stacker_xml, stacker_base_xml, z_pf)
        origin = np.array([prev_field_x + s_half[0] - s_pos[0],
                           belt_hi_y + s_half[1] - s_pos[1],
                           s_mount])
        _attach_robot(spec, st, f"stacker{i}_", origin.tolist())
        b_pos = origin + s_pos
        b_lo, b_hi = b_pos - s_half, b_pos + s_half

        target = stacker_targets[i]
        if target == "floor":
            platform = add_floor_area(
                spec, name=f"platform{i}", center_x=float(origin[0]),
                y_min=float(b_hi[1]), n=int(floor_area_n), z_playing_field=z_pf,
                box_size=box_size, box_clearance=box_clearance,
                wall_thickness=wall_thickness, goal_name=f"stacker{i}_goal")
        else:
            # A shelf level per box the stacker owns, the belt box included
            levels = (int(np.sum(container.box_color == i)) + 1) if target == "shelf" else 0
            platform = add_platform(
                spec, name=f"platform{i}", center_x=float(origin[0]),
                y_min=float(b_hi[1]), size=1.5 * box_size, z_playing_field=z_pf,
                box_size=box_size, goal_name=f"stacker{i}_goal",
                walls=platform_walls, wall_height=box_size,
                wall_thickness=wall_thickness, shelf_levels=levels,
                box_clearance=box_clearance)

        bounds.append((b_lo, b_hi))
        platforms.append(platform)
        goal_sites.append(platform.goal_site)
        # Stacker field: across the belt (plus stacker_field_reach for a long
        # arm) to behind the target, and tall enough for the top of a shelf
        field_lo.append(np.array([b_lo[0], belt_lo_y - field_clearance - stacker_field_reach,
                                  z_pf - field_clearance]))
        field_hi.append(np.array([b_hi[0], platform.upper[1] + field_clearance,
                                  max(b_hi[2], platform.upper[2] + field_clearance)]))
        prev_field_x = float(b_hi[0])

    conveyor = add_conveyor(
        spec, x_min=float(l_hi[0]),
        x_max=float(field_hi[-1][0]) + 2.0 * box_size, center_y=line_y,
        width=2.0 * belt_half_w, z_playing_field=z_pf,
        belt_velocity=belt_velocity, box_size=box_size, wall_thickness=wall_thickness)

    # Divider between the loader and stacker fields, with a hole two boxes
    # high so a box passes and an arm does not
    add_divider(
        spec, x=float(field_hi[0][0]), center_y=line_y,
        opening_half_w=float(conveyor.outer_half_w),
        gap_z=z_pf + 2.0 * box_size, height=float(divider_dims[1]),
        side_length=float(divider_dims[0]), wall_thickness=wall_thickness)

    names = list(container.boxes)
    colors = list(container.box_color)
    positions = list(container.box_pos)
    n_grid = len(names)
    for i in range(n_stackers):
        pos = [float(field_lo[i + 1][0]) + box_size, line_y, z_pf + box_size / 2.0]
        name = f"belt_box_{i}"
        add_box(spec, name=name, pos=pos, box_size=box_size, box_mass=box_mass,
                color=i, palette=palette)
        names.append(name)
        colors.append(i)
        positions.append(pos)

    _add_field(spec, "loader_field", field_lo[0], field_hi[0], _RGBA_LOADER_FIELD)
    for i in range(n_stackers):
        _add_field(spec, f"stacker{i}_field", field_lo[i + 1], field_hi[i + 1],
                   _RGBA_STACKER_FIELD)

    _check_visual_meshes(spec)
    spec.option.timestep = timestep
    layout = Layout(
        z_playing_field=z_pf, loader_bounds=(l_lo, l_hi), container=container,
        conveyor=conveyor, field_lo=np.stack(field_lo), field_hi=np.stack(field_hi),
        box_names=names, box_color=np.asarray(colors, dtype=np.int64),
        box_pos=np.asarray(positions, dtype=np.float64), n_grid=n_grid,
        loader_order=_serpentine(box_array), goal_sites=goal_sites,
        stacker_bounds=bounds, platforms=platforms)
    return spec, layout


class TruckUnloadParallelEnv(ParallelMujocoEnv):
    """Batched truck-unloading env with one body design per world.

    Besides the constructor settings, an instance holds:

    - the scene ``layout``, and model ids and addresses cached at setup (tips,
    goals, grasp actuators and geoms, each robot's joints, boxes);
    - the link table of the embodiment;
    - per-world device buffers the kernels hold warp views of, packed once into
    the kernel argument structs, including the box and robot state that the
    post-physics kernels write;
    - host-side episode state for the reward (shaping baselines, reward given
    per box, idle clocks, delivery counts), and ``last_*`` outcome flags.
    """

    _length_floor_warned = False      # see _warn_length_floor


    # Written per world by the deform, and site_pos also by the rising goal sites
    PER_WORLD_MODEL_FIELDS = ("geom_pos", "geom_quat", "geom_size", "geom_rbound",
                              "geom_aabb", "body_pos", "body_ipos", "body_iquat",
                              "body_mass", "body_inertia", "site_pos")

    def __init__(
        self,
        num_envs: int = 4,
        *,
        # Robot MJCFs. A stacker is attached with a "stacker<i>_" prefix, an arm
        # with a base is mounted on its BASE_BODY. The XMLs' timestep is replaced
        loader_xml: str = "scara_loader_arm.xml",
        loader_base_xml: str | None = "base_xy_loader.xml",
        stacker_xml: str = "scara_stacker_arm.xml",
        stacker_base_xml: str | None = "base_xy_stacker.xml",
        scene_xml: str = "scene_truck_unload.xml",
        # Keyframes combined into the reset pose, matched by suffix since
        # attaching prefixes them
        init_key: str | tuple[str, ...] | list[str] = ("init_pos", "init_pos2"),
        max_offset: float = 1.0,                  # metres per unit of link embodiment action
        min_link_length: float = 0.02,            # deformed links are clamped to this
        min_length_pad: float = 0.02,             # embodiment clip keeps links this far above it

        n_stackers: int = 1,

        # Shares of a box's return given for reaching it, carrying it to the goal
        # and releasing it. The rest is given over goal_dwell_steps at the goal
        reach_weight: float = 0.25,
        carry_weight: float = 0.25,
        ungrasp_weight: float = 0.25,
        ungrasp_threshold: float = 0.1,           # how far the tip must clear a released box
        goal_threshold: float = 0.05,             # box-goal distance that counts as at the goal
        goal_dwell_steps: int = 10,               # total steps at the goal that deliver a box
        loader_goal_return: float = 5.0,          # return of all the loader's boxes
        stacker_goal_return: float = 5.0,         # return of all boxes of all stackers

        time_limit_s: float = 120.0,
        inactivity_limit_s: float = 15.0,         # no box loaded, or none stacked, for this long ends it
        # How the time limit is reported. False terminates, since bootstrapping
        # through this clock rewards stalling. Set time_in_obs with either
        truncate_at_time_limit: bool = False,
        terminate_out_of_field: bool = True,      # clear only for recording video

        platform_walls: bool = False,             # walls on three sides of a platform
        stacker_targets=("shelf",),                # per stacker: "platform", "shelf" or "floor"
        stacker_colors=("red",),                   # per stacker: a name from container.COLOR_NAMES
        floor_area_n: int = 3,
        container_opening: str = "all",           # "top", "front" or "all"
        baseplate_dims=(2.0, 1.0),                # length along x, width along y
        ramp_tilt_degs: float = 20.0,
        ramp_lip: float = 0.0,                    # height of a plate under the ramp's lower edge
        ramp_x_offset: float = 0.8,               # landing between the ramp and the container
        container_z_offset: float = 0.45,         # raises the container floor above the baseplate
        z_playing_field: float = 0.45,
        belt_velocity: float = 0.2,               # must be nonzero, see add_conveyor
        belt_velocity_spread: float = 0.2,        # each respawn draws the belt speed from belt_velocity * (1 +/- this)
        divider_dims=(3.0, 3.0),                  # side panel length, height
        wall_thickness: float = 0.025,
        box_size: float = 0.1,                    # side length
        box_array=(2, 2, 1),                      # boxes in the container along x, y, z
        box_clearance: float = 0.02,
        box_jitter: float | None = None,          # +/- x, y nudge of each container box at respawn (default 0.45 * box_clearance)
        box_mass: float = 0.5,
        field_clearance: float = 0.1,
        stacker_field_reach: float = 1.2,         # field past the belt's far edge, for long arms

        max_rel_dist: float = 1.75,               # obs scale of relative vectors
        vel_scale: float = 10.0,                  # obs scale of velocities
        time_in_obs: bool = True,                 # observe the share of the episode left
        timestep: float = 0.005,

        camera_mode: str = "fixed",               # see _setup_camera
        phys_substeps: int = 8,
        # parallel-runner params
        device: str = "cuda",
        viz_env_id: int = 0,
        render_mode: str | None = None,
        debug: bool = False,
        use_graph: bool = True,
        check_finite: bool = False,
        njmax: int | None = None,                 # None: sized from the box count
        nconmax: int | None = None,
        seed: int = 1234,
    ):
        if ungrasp_threshold < 0.0:
            raise ValueError(f"ungrasp_threshold is a distance and must be "
                             f">= 0, got {ungrasp_threshold}")
        weights = (reach_weight, carry_weight, ungrasp_weight)
        if min(weights) < 0.0 or sum(weights) > 1.0:
            raise ValueError(f"reach, carry and ungrasp weights are fractions of a "
                             f"box's return: each must be >= 0 and their sum <= 1, "
                             f"got {weights}")
        if camera_mode not in _CAMERA_MODES:
            raise ValueError(f"unknown camera_mode {camera_mode!r}; "
                             f"have {list(_CAMERA_MODES)}")
        if n_stackers < 1:
            raise ValueError("n_stackers must be at least 1")
        if min_link_length <= 0.0:
            raise ValueError("min_link_length is a length floor and must be "
                             "positive; a link at zero length has no mass, no "
                             "inertia and no bounding volume")
        targets = (("platform",) * int(n_stackers) if stacker_targets is None
                   else tuple(str(t) for t in stacker_targets))
        if len(targets) != n_stackers or any(t not in _TARGETS for t in targets):
            raise ValueError(f"stacker_targets must name one of {list(_TARGETS)} per stacker "
                             f"({n_stackers}), got {stacker_targets!r}")
        colors = None if stacker_colors is None else tuple(str(c) for c in stacker_colors)
        if colors is not None and (len(colors) != n_stackers or any(c not in COLOR_NAMES for c in colors)):
            raise ValueError(f"stacker_colors must name one of {sorted(COLOR_NAMES)} per stacker "
                             f"({n_stackers}), got {stacker_colors!r}")
        if stacker_field_reach < 0.0:
            raise ValueError("stacker_field_reach is a distance and must be >= 0")
        if int(floor_area_n) < 1:
            raise ValueError("floor_area_n must be at least 1")
        self._xmls = (loader_xml, loader_base_xml, stacker_xml, stacker_base_xml,
                      scene_xml)
        self._scene = dict(
            n_stackers=int(n_stackers), platform_walls=bool(platform_walls),
            stacker_targets=targets, stacker_colors=colors,
            floor_area_n=int(floor_area_n),
            container_opening=str(container_opening),
            baseplate_dims=tuple(float(v) for v in baseplate_dims),
            ramp_tilt_degs=float(ramp_tilt_degs), ramp_lip=float(ramp_lip),
            ramp_x_offset=float(ramp_x_offset),
            container_z_offset=float(container_z_offset),
            z_playing_field=float(z_playing_field),
            belt_velocity=float(belt_velocity),
            divider_dims=tuple(float(v) for v in divider_dims),
            wall_thickness=float(wall_thickness), box_size=float(box_size),
            box_array=tuple(int(n) for n in box_array),
            box_clearance=float(box_clearance), box_mass=float(box_mass),
            field_clearance=float(field_clearance),
            stacker_field_reach=float(stacker_field_reach), timestep=float(timestep))
        self.n_stackers = int(n_stackers)
        self.n_robots = self.n_stackers + 1
        self.box_size = float(box_size)
        self.belt_velocity = float(belt_velocity)
        self.belt_velocity_spread = float(belt_velocity_spread)
        # Two neighbours can close at most twice the jitter, so below half the clearance boxes never overlap
        self.box_jitter = 0.45 * float(box_clearance) if box_jitter is None else float(box_jitter)
        if not 0.0 <= self.box_jitter < 0.5 * float(box_clearance):
            raise ValueError(f"box_jitter must be in [0, box_clearance / 2) = [0, {0.5 * float(box_clearance)}), got {self.box_jitter}")
        if not 0.0 <= self.belt_velocity_spread < 1.0:
            raise ValueError(f"belt_velocity_spread must be in [0, 1), got {self.belt_velocity_spread}")
        self.init_key = ((init_key,) if isinstance(init_key, str)
                         else tuple(init_key))
        self.max_offset = float(max_offset)
        self.min_link_length = float(min_link_length)
        self.min_length_pad = float(min_length_pad)
        self.reach_weight = float(reach_weight)
        self.carry_weight = float(carry_weight)
        self.ungrasp_weight = float(ungrasp_weight)
        self.ungrasp_threshold = float(ungrasp_threshold)
        self.goal_threshold = float(goal_threshold)
        self.goal_dwell_steps = max(int(goal_dwell_steps), 1)
        self.loader_goal_return = float(loader_goal_return)
        self.stacker_goal_return = float(stacker_goal_return)
        self.time_limit_s = float(time_limit_s)
        self.inactivity_limit_s = float(inactivity_limit_s)
        self.truncate_at_time_limit = bool(truncate_at_time_limit)
        self.terminate_out_of_field = bool(terminate_out_of_field)
        self.camera_mode = str(camera_mode)
        self.max_rel_dist = float(max_rel_dist)
        self.vel_scale = float(vel_scale)
        self.time_in_obs = bool(time_in_obs)
        self.timestep = float(timestep)
        self.check_finite = bool(check_finite)
        self.seed_rng = np.random.default_rng(seed)

        n_boxes = int(np.prod(self._scene["box_array"])) + self.n_stackers
        nconmax = 12 * n_boxes + 36 if nconmax is None else int(nconmax)
        njmax = 2 * nconmax if njmax is None else int(njmax)

        super().__init__(
            num_envs, device=device, viz_env_id=viz_env_id,
            max_episode_steps=int(round(time_limit_s / timestep)), debug=debug,
            seed=seed, phys_substeps=phys_substeps, render_mode=render_mode,
            use_graph=use_graph, njmax=njmax, nconmax=nconmax)

        self._cache_model_addresses()
        self._alloc_device_state()
        self._build_kernel_params()
        self._make_spaces()

        # Step program: tick -> ctrl -> physics -> kinematics/com_vel -> task kernels.
        # Finished worlds respawn after it, outside the graph (ParallelMujocoEnv.step).
        self.register_step_launch(self._launch_tick)
        self.register_step_launch(self._launch_ctrl)
        self.register_post_step_launch(self._launch_task_step)

    # ------------------------------------------------------------------ #
    # Model assembly
    # ------------------------------------------------------------------ #
    def _build_base_spec(self) -> mujoco.MjSpec:
        loader_xml, loader_base_xml, stacker_xml, stacker_base_xml, scene_xml = self._xmls
        spec, self.layout = build_scene(
            loader_xml=loader_xml, loader_base_xml=loader_base_xml,
            stacker_xml=stacker_xml, stacker_base_xml=stacker_base_xml,
            scene_xml=scene_xml,
            rng=self.seed_rng, **self._scene)
        return spec

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    def _cache_model_addresses(self) -> None:
        """Cache model ids and addresses per robot and box, find the embodiment
        parts, and size the embodiment action and episode clocks."""
        m, lay = self.m, self.layout
        self._prefix = [""] + [f"stacker{i}_" for i in range(self.n_stackers)]
        self._tip_sid, self._tip_bid, self._root_bid = [], [], []
        self._goal_sid, self._grasp_col = [], []
        for r, pre in enumerate(self._prefix):
            sid = m.site(pre + TIP_SITE).id
            bid = int(m.site_bodyid[sid])
            self._tip_sid.append(sid)
            self._tip_bid.append(bid)
            self._root_bid.append(int(m.body_rootid[bid]))
            self._goal_sid.append(m.site(lay.goal_sites[r]).id)
            self._grasp_col.append(self._grasp_actuator(m, pre + GRASP_ACT))
        self._grasp_geoms = [self._grasp_parts(m, pre, a)
                             for pre, a in zip(self._prefix, self._grasp_col)]
        self._n_part = [max(part for _, part in g) + 1 for g in self._grasp_geoms]

        # Each robot's own joints, for its qpos/qvel block of the obs
        self._rob_qadr, self._rob_dadr = [], []
        for r in range(self.n_robots):
            qadr, dadr = [], []
            for j in range(m.njnt):
                if int(m.body_rootid[m.jnt_bodyid[j]]) != self._root_bid[r]:
                    continue
                n_q, n_v = _JNT_DIMS[int(m.jnt_type[j])]
                qadr += list(range(int(m.jnt_qposadr[j]),
                                   int(m.jnt_qposadr[j]) + n_q))
                dadr += list(range(int(m.jnt_dofadr[j]),
                                   int(m.jnt_dofadr[j]) + n_v))
            self._rob_qadr.append(np.asarray(qadr, dtype=np.int64))
            self._rob_dadr.append(np.asarray(dadr, dtype=np.int64))

        self._belt_gid = m.geom(lay.conveyor.geom).id
        # The belt surface and both rollers move; a respawn draws one speed for all of them
        self._belt_gids = np.array([g for g in (self._belt_gid, m.geom(f"{lay.conveyor.geom}_cyl_f").id,
                                                m.geom(f"{lay.conveyor.geom}_cyl_b").id)], dtype=np.int32)
        if not hasattr(self.gpu_model, "geom_surfacevel"):
            raise RuntimeError("Truck Unload needs mujoco-warp >= 3.12 for the moving conveyor (geom surface velocity); "
                               "run: pip install -U 'mujoco-warp>=3.12'")
        self._make_field_per_world("geom_surfacevel")
        self._box_qadr = np.array([m.joint(f"{n}_free").qposadr[0]
                                   for n in lay.box_names], dtype=np.int64)
        self._box_dadr = np.array([m.joint(f"{n}_free").dofadr[0]
                                   for n in lay.box_names], dtype=np.int64)
        self.n_boxes = len(lay.box_names)

        self._segs = find_segments(m, self.min_link_length)
        self._suffix_lo = self._segs["axis_lo"].copy()   # before _tighten_length_floor raises it
        self.n_links = len(self._segs["body"])
        self.n_offsets = n_slots(self._segs)             # a link takes one slot per axis in its name
        self._tighten_length_floor()
        self._warn_length_floor()
        self._init_qpos = self._reset_pose(m)

        # Episode clocks in control steps
        self._idle_limit = max(int(round(
            self.inactivity_limit_s / (self.timestep * self.phys_substeps))), 1)
        self._episode_ctrl_steps = max(
            -(-int(self.max_episode_steps) // int(self.phys_substeps)), 1)

    @staticmethod
    def _grasp_actuator(m: mujoco.MjModel, name: str) -> int:
        """Id of a grasp actuator: a finger drive (joint) or an adhesion (body),
        whose ctrlrange starts at 0 so the action's dead zone opens the gripper."""
        a = m.actuator(name)
        kinds = (int(mujoco.mjtTrn.mjTRN_JOINT), int(mujoco.mjtTrn.mjTRN_BODY))
        if int(a.trntype[0]) not in kinds:
            raise ValueError(f"{name} must drive a joint (a finger) or a body "
                             f"(an adhesion), got trntype {int(a.trntype[0])}")
        lo = float(a.ctrlrange[0])
        if not a.ctrllimited[0] or lo != 0.0:
            raise ValueError(f"{name} must have ctrlrange starting at 0, so that "
                             f"its dead zone opens the gripper; got "
                             f"{a.ctrlrange.tolist()}")
        return int(a.id)

    @staticmethod
    def _grasp_parts(m: mujoco.MjModel, prefix: str,
                     act: int) -> list[tuple[int, int]]:
        """(geom id, gripper part) of every geom the grasp test watches.

        The ``grasp_geom_<n>`` geoms are the gripper's contact faces, and each body
        carrying them is one part a held box must touch. Without any, an adhesion
        gripper falls back to the geoms of the body it pulls with.
        """
        named = [g for g in range(m.ngeom)
                 if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                 .startswith(prefix + GRASP_GEOM)]
        if not named:
            if int(m.actuator_trntype[act]) != int(mujoco.mjtTrn.mjTRN_BODY):
                raise ValueError(
                    f"{prefix + GRASP_ACT} drives a joint, so the gripper's "
                    f"contact faces cannot be guessed: name them "
                    f"{prefix + GRASP_GEOM}<n>")
            body = int(m.actuator_trnid[act][0])
            named = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == body]
            if not named:
                raise ValueError(f"the body {prefix + GRASP_ACT} pulls with has "
                                 f"no geoms, so nothing can touch it")
        parts = {}
        for g in named:
            parts.setdefault(int(m.geom_bodyid[g]), len(parts))
        return [(g, parts[int(m.geom_bodyid[g])]) for g in named]

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
        n, nr, nb, dev = self.num_envs, self.n_robots, self.n_boxes, self.device
        self._alloc_action_buffers(self.act_dim(), self.n_offsets)
        self._offsets_t = torch.zeros((n, self.n_offsets), device=dev)
        self._offset_scale = torch.full((n,), float(self.max_offset), device=dev)   # max_offset each world was built with
        self._tick = torch.zeros(1, dtype=torch.int32, device=dev)
        self._lane_seed = torch.full((n,), int(self.seed), dtype=torch.int32, device=dev)
        self._tunables = torch.zeros(kern.N_TUNABLES, device=dev)
        self._wp_tunables = wp.from_torch(self._tunables)
        self._qpos_init = torch.as_tensor(self._init_qpos, device=dev)

        i32 = dict(dtype=torch.int32, device=dev)
        self._box_state = torch.zeros((n, nb), **i32)
        self._left_region = torch.zeros((n, nb), **i32)
        # What each robot was given for each box this episode, taken back if a
        # delivery is undone
        self._paid = torch.zeros((n, nr, nb), device=dev)
        self._box_owner = torch.as_tensor(self.layout.box_color.astype(np.int64) + 1, device=dev)
        self._touch = torch.zeros((n, nr, nb, max(self._n_part)), **i32)
        self._on_bed = torch.zeros((n, nb), **i32)
        deadzone = np.zeros(self.m.nu, dtype=np.int32)
        deadzone[self._grasp_col] = 1
        self._ctrl_deadzone = torch.as_tensor(deadzone, device=dev)

        # Task state and outputs, written by task_kernel
        self._tgt = torch.full((n, nr), -1, **i32)
        self._loader_idx = torch.zeros(n, **i32)
        self._dwell = torch.zeros((n, nr), **i32)
        self._lost = torch.zeros((n, nr), **i32)
        self._scored = torch.zeros((n, nr), **i32)
        self._undeliv = torch.zeros((n, nr), **i32)
        self._rebase_t = torch.zeros((n, nr), **i32)

        self._tip_pos = torch.zeros((n, nr, 3), device=dev)
        self._tip_quat = torch.zeros((n, nr, 4), device=dev)
        self._tip_lvel = torch.zeros((n, nr, 3), device=dev)
        self._tip_avel = torch.zeros((n, nr, 3), device=dev)
        self._tgt_pos = torch.zeros((n, nr, 3), device=dev)
        self._tgt_quat = torch.zeros((n, nr, 4), device=dev)
        self._tgt_lvel = torch.zeros((n, nr, 3), device=dev)
        self._tgt_avel = torch.zeros((n, nr, 3), device=dev)
        self._goal_pos = torch.zeros((n, nr, 3), device=dev)
        self._d_tip = torch.zeros((n, nr), device=dev)
        self._d_goal = torch.zeros((n, nr), device=dev)
        self._grasped = torch.zeros((n, nr), device=dev)
        self._close = torch.zeros((n, nr), device=dev)
        self._away = torch.zeros((n, nr), device=dev)
        self._has_tgt = torch.zeros((n, nr), device=dev)

        # Host-side episode state - no kernel views these, so they may be rebound
        self._idle_loader = torch.zeros(n, dtype=torch.long, device=dev)
        self._idle_stack = torch.zeros(n, dtype=torch.long, device=dev)
        self._prev_d_tip = torch.zeros((n, nr), device=dev)
        self._best_d_goal = torch.zeros((n, nr), device=dev)   # closest the target has come to its goal
        self._low_grasp = torch.ones((n, nr), device=dev)      # lowest grasp command while at the goal
        self._d_tip0 = torch.ones((n, nr), device=dev)
        self._d_goal0 = torch.ones((n, nr), device=dev)
        self._diverged = torch.zeros(n, dtype=torch.bool, device=dev)
        self._diverged_total = torch.zeros((), dtype=torch.long, device=dev)
        self._delivered_total = torch.zeros((n, nr), dtype=torch.long, device=dev)
        # Deliveries that make an episode a success, per robot
        stacker_deliveries = np.bincount(
            self.layout.box_color, minlength=self.n_stackers)
        self._success_deliveries = torch.as_tensor(
            [self.layout.n_grid, *stacker_deliveries.tolist()],
            dtype=torch.long, device=dev)
        self.last_success = torch.zeros(n, dtype=torch.bool, device=dev)

        self._field_c = torch.as_tensor(
            (self.layout.field_lo + self.layout.field_hi) / 2.0,
            dtype=torch.float32, device=dev)
        self._field_h = torch.as_tensor(
            (self.layout.field_hi - self.layout.field_lo) / 2.0,
            dtype=torch.float32, device=dev).clamp(min=1e-6)
        self._grasp_idx = torch.as_tensor(self._grasp_col, dtype=torch.long,
                                          device=dev)
        # Return of one box per robot. Every box has a stacker, so the stackers
        # share their return over all boxes
        box_return = np.full(self.n_robots, self.stacker_goal_return / self.n_boxes,
                             dtype=np.float32)
        box_return[0] = self.loader_goal_return / self.layout.n_grid
        self._box_return = torch.as_tensor(box_return, device=dev)
        goal_fraction = 1.0 - self.reach_weight - self.carry_weight - self.ungrasp_weight
        self._goal_rew = self._box_return * (goal_fraction / self.goal_dwell_steps)
        self._qadr_t = [torch.as_tensor(a, device=dev) for a in self._rob_qadr]
        self._dadr_t = [torch.as_tensor(a, device=dev) for a in self._rob_dadr]
        off, scl = self._qpos_norm()
        self._qpos_off = [torch.as_tensor(off[a], device=dev)
                          for a in self._rob_qadr]
        self._qpos_scl = [torch.as_tensor(scl[a], device=dev)
                          for a in self._rob_qadr]

    def _qpos_norm(self) -> tuple[np.ndarray, np.ndarray]:
        """(offset, scale) mapping qpos into about [-1, 1]: a robot's free-joint
        position by its field, quaternions unchanged, limited joints by range.
        The boxes' free joints are never observed."""
        m, lay = self.m, self.layout
        off = np.zeros(m.nq, dtype=np.float32)
        scl = np.ones(m.nq, dtype=np.float32)
        centre = (lay.field_lo + lay.field_hi) / 2.0
        half = np.maximum((lay.field_hi - lay.field_lo) / 2.0, 1e-6)
        for j in range(m.njnt):
            adr = int(m.jnt_qposadr[j])
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                root = int(m.body_rootid[m.jnt_bodyid[j]])
                if root in self._root_bid:
                    r = self._root_bid.index(root)
                    off[adr:adr + 3] = centre[r]
                    scl[adr:adr + 3] = 1.0 / half[r]
                continue
            if m.jnt_limited[j]:
                scl[adr] = 1.0 / max(np.abs(m.jnt_range[j]).max(), 1e-6)
        return off, scl

    def _build_kernel_params(self) -> None:
        dev, lay = self.device, self.layout
        sim = kern.SimArrays()
        for attr in ("qpos", "qvel", "qacc", "qacc_warmstart", "time", "ctrl",
                     "xquat", "cvel", "subtree_com", "site_xpos"):
            setattr(sim, attr, getattr(self.gpu_data, attr))
        sim.geom_surfacevel = self.gpu_model.geom_surfacevel
        self._sim = sim

        dfm = DeformArrays()
        for attr in self.PER_WORLD_MODEL_FIELDS:
            setattr(dfm, attr, getattr(self.gpu_model, attr))
        self._dfm = dfm

        env = kern.EnvArrays()
        env.mask = wp.from_torch(self.reset_mask)
        env.qpos_init = wp.from_torch(self._qpos_init)
        env.deform = wp.from_torch(self._reset_act_buf)
        env.ctrl_act = wp.from_torch(self._act_buf)
        env.ctrl_scale = wp.from_torch(self.ctrl_scale)
        env.ctrl_mid = wp.from_torch(self.ctrl_mid)
        env.ctrl_deadzone = wp.from_torch(self._ctrl_deadzone)
        env.offsets = wp.from_torch(self._offsets_t)
        env.offset_scale = wp.from_torch(self._offset_scale)
        env.tick = wp.from_torch(self._tick)
        env.lane_seed = wp.from_torch(self._lane_seed)
        env.belt_gid = wp.array(self._belt_gids, dtype=wp.int32, device=dev)
        self._env = env

        self._seg = build_seg_arrays(self._segs, dev)

        box = kern.BoxArrays()
        box.qadr = wp.array(self._box_qadr.astype(np.int32), dtype=wp.int32, device=dev)
        box.dadr = wp.array(self._box_dadr.astype(np.int32), dtype=wp.int32, device=dev)
        box.color = wp.array(lay.box_color.astype(np.int32), dtype=wp.int32, device=dev)
        box.spawn_z = wp.array(lay.box_pos[:, 2].astype(np.float32),
                               dtype=wp.float32, device=dev)
        box.order = wp.array(lay.loader_order.astype(np.int32), dtype=wp.int32,
                             device=dev)
        box.state = wp.from_torch(self._box_state)
        box.left_region = wp.from_torch(self._left_region)
        self._box = box

        adh = kern.GraspArrays()
        contact = self.gpu_data.contact
        adh.contact_geom = contact.geom
        adh.contact_worldid = contact.worldid
        adh.nacon = self.gpu_data.nacon
        geom_robot = np.full(self.m.ngeom, -1, dtype=np.int32)
        geom_part = np.zeros(self.m.ngeom, dtype=np.int32)
        for r, geoms in enumerate(self._grasp_geoms):
            for g, part in geoms:
                geom_robot[g], geom_part[g] = r, part
        geom_box = np.full(self.m.ngeom, -1, dtype=np.int32)
        for b, name in enumerate(lay.box_names):
            geom_box[self.m.geom(f"{name}_geom").id] = b
        adh.geom_robot = wp.array(geom_robot, dtype=wp.int32, device=dev)
        adh.geom_part = wp.array(geom_part, dtype=wp.int32, device=dev)
        adh.geom_box = wp.array(geom_box, dtype=wp.int32, device=dev)
        adh.touch = wp.from_torch(self._touch)
        self._adh = adh

        bed = kern.BedArrays()
        geom_bed = np.zeros(self.m.ngeom, dtype=np.int32)
        for name in lay.container.bed_geoms:
            geom_bed[self.m.geom(name).id] = 1
        bed.geom_bed = wp.array(geom_bed, dtype=wp.int32, device=dev)
        bed.on_bed = wp.from_torch(self._on_bed)
        self._bed = bed

        rob = kern.RobotArrays()
        for name, val in (("tip_sid", self._tip_sid), ("tip_bid", self._tip_bid),
                          ("root_bid", self._root_bid),
                          ("goal_sid", self._goal_sid),
                          ("grasp_act", self._grasp_col),
                          ("n_part", self._n_part)):
            setattr(rob, name, wp.array(np.asarray(val, dtype=np.int32),
                                        dtype=wp.int32, device=dev))
        goal_pos0 = self.m.site_pos[np.asarray(self._goal_sid)]
        rob.goal_pos0 = wp.array(goal_pos0.astype(np.float32), dtype=wp.vec3,
                                 device=dev)
        rob.field_lo = wp.array(lay.field_lo.astype(np.float32), dtype=wp.vec3,
                                device=dev)
        rob.field_hi = wp.array(lay.field_hi.astype(np.float32), dtype=wp.vec3,
                                device=dev)
        # Robot 0's goal is tested separately, its entries here are unused
        kind = [kern.GOAL_SITE] + [kern.GOAL_REGION if p.kind == "region" else kern.GOAL_SITE for p in lay.platforms]
        g_lo = [np.zeros(3)] + [p.region_lower if p.kind == "region" else np.zeros(3) for p in lay.platforms]
        g_hi = [np.zeros(3)] + [p.region_upper if p.kind == "region" else np.zeros(3) for p in lay.platforms]
        rob.goal_kind = wp.array(np.asarray(kind, dtype=np.int32), dtype=wp.int32, device=dev)
        rob.goal_step = wp.array(np.asarray([0.0] + [p.goal_step for p in lay.platforms], dtype=np.float32), dtype=wp.float32, device=dev)
        rob.goal_lo = wp.array(np.stack(g_lo).astype(np.float32), dtype=wp.vec3, device=dev)
        rob.goal_hi = wp.array(np.stack(g_hi).astype(np.float32), dtype=wp.vec3, device=dev)
        rob.tgt = wp.from_torch(self._tgt)
        rob.loader_idx = wp.from_torch(self._loader_idx)
        rob.dwell = wp.from_torch(self._dwell)
        rob.lost = wp.from_torch(self._lost)
        rob.scored = wp.from_torch(self._scored)
        rob.undelivered = wp.from_torch(self._undeliv)
        rob.rebase = wp.from_torch(self._rebase_t)
        for name, t, dtype in (
                ("tip_pos", self._tip_pos, wp.vec3),
                ("tip_quat", self._tip_quat, wp.quat),
                ("tip_lvel", self._tip_lvel, wp.vec3),
                ("tip_avel", self._tip_avel, wp.vec3),
                ("tgt_pos", self._tgt_pos, wp.vec3),
                ("tgt_quat", self._tgt_quat, wp.quat),
                ("tgt_lvel", self._tgt_lvel, wp.vec3),
                ("tgt_avel", self._tgt_avel, wp.vec3),
                ("goal_pos", self._goal_pos, wp.vec3),
                ("d_tip", self._d_tip, None), ("d_goal", self._d_goal, None),
                ("grasped", self._grasped, None), ("close", self._close, None),
                ("away", self._away, None), ("has_tgt", self._has_tgt, None)):
            setattr(rob, name, wp.from_torch(t) if dtype is None
                    else wp.from_torch(t, dtype=dtype))
        self._rob = rob

        c = kern.EnvConst()
        c.seed = int(self.seed)
        c.nq, c.nv, c.nu = int(self.m.nq), int(self.m.nv), int(self.m.nu)
        c.n_seg = int(self.n_links)
        c.n_robot = int(self.n_robots)
        c.n_box = int(self.n_boxes)
        c.n_grid = int(lay.n_grid)
        c.dwell_limit = int(self.goal_dwell_steps)
        c.goal_thresh = float(self.goal_threshold)
        # The cylinder around a box the tip must leave, grown by ungrasp_threshold
        c.away_xy = float(self.ungrasp_threshold
                          + self.box_size * np.sqrt(2.0) / 2.0)
        c.away_z = float(self.ungrasp_threshold + self.box_size / 2.0)
        c.belt_x_min = float(lay.conveyor.lower[0])
        c.cont_lo = wp.vec3(*(float(v) for v in lay.container.lower))
        c.cont_hi = wp.vec3(*(float(v) for v in lay.container.upper))
        self._consts = c

        self._refresh_tunables()

    def _refresh_tunables(self) -> None:
        # Order must match the T_* indices in truck_unload_kernels
        self._tunables.copy_(torch.tensor([self.max_offset, self.belt_velocity, self.belt_velocity_spread, self.box_jitter],
                                          dtype=torch.float32))

    def act_dim(self) -> int:
        # Control action only - the embodiment action is a separate step() argument
        return int(self.m.nu)

    def obs_dim(self) -> int:
        # Per robot: its qpos/qvel, tip pose and twist (13), target pose and twist
        # (13), three relative vectors (9), grasped and has-target (2). Then the clock
        per_robot = sum(len(a) + len(d)
                        for a, d in zip(self._rob_qadr, self._rob_dadr))
        return per_robot + 37 * self.n_robots + int(self.time_in_obs)

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
        unit, flag, blocks = (-1.0, 1.0), (0.0, 1.0), []
        for r in range(self.n_robots):
            blocks += [(len(self._qadr_t[r]), None), (len(self._dadr_t[r]), None), (3, None), (4, unit), (3, None), (3, None),
                       (3, None), (4, unit), (3, None), (3, None), (3, None), (3, None), (3, None), (1, flag), (1, flag)]
        if self.time_in_obs:
            blocks.append((1, flag))
        return self._bounds_from_blocks(blocks)

    def _make_spaces(self) -> None:
        # Only quaternion, flag and clock entries have finite obs bounds. The
        # embodiment is not observed
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

    @property
    def offset_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) of each embodiment entry, with lo raised so no link shortens past
        ``min_link_length + min_length_pad``."""
        return slot_bounds(self._segs, self.n_offsets)

    # ------------------------------------------------------------------ #
    # Kernel launches
    # ------------------------------------------------------------------ #
    def _launch_tick(self) -> None:
        wp.launch(kern.tick_kernel, dim=1, inputs=[self._env])   # advance the RNG tick

    def _launch_reset(self) -> None:
        """Explicit reset: bump the RNG tick, then respawn the masked worlds."""
        self._launch_tick()
        self._launch_respawn()

    def _launch_respawn(self) -> None:
        wp.launch(kern.reset_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._dfm, self._seg, self._box,
                          self._rob, self._consts, self._wp_tunables])

    def _launch_ctrl(self) -> None:
        wp.launch(kern.ctrl_kernel, dim=self.num_envs,
                  inputs=[self._env, self._sim, self._consts])

    def _launch_task_step(self) -> None:
        """Run the post-physics task kernels inside the CUDA-graphed step program."""
        wp.launch(kern.clear_task_bufs_kernel, dim=self.num_envs,
                  inputs=[self._adh, self._bed, self._box, self._rob, self._consts])
        wp.launch(kern.contact_kernel, dim=self.gpu_data.naconmax,
                  inputs=[self._adh, self._bed, self._env.mask, 0])
        wp.launch(kern.box_kernel, dim=(self.num_envs, self.n_boxes),
                  inputs=[self._sim, self._box, self._rob, self._bed,
                          self._consts, self._env.mask, 0])
        wp.launch(kern.task_kernel, dim=(self.num_envs, self.n_robots),
                  inputs=[self._sim, self._dfm, self._box, self._rob,
                          self._adh, self._consts, self._env.mask, 0])

    def _launch_task(self, only_reset: bool = False) -> None:
        """Run the post-physics kernels on torch's stream: contacts, then boxes,
        then robots, since targets are chosen from the boxes still in play.
        ``only_reset`` limits them to the worlds in ``reset_mask``."""
        if only_reset:
            m = self.reset_mask
            self._touch[m] = 0
            self._on_bed[m] = 0
            self._undeliv[m] = 0
            self._left_region[m] = 0
        else:
            self._touch.zero_()
            self._on_bed.zero_()
            self._undeliv.zero_()
            self._left_region.zero_()
        only = int(only_reset)
        with self._wp_scope():
            wp.launch(kern.contact_kernel, dim=self.gpu_data.naconmax,
                      inputs=[self._adh, self._bed, self._env.mask, only])
            wp.launch(kern.box_kernel, dim=(self.num_envs, self.n_boxes),
                      inputs=[self._sim, self._box, self._rob, self._bed,
                              self._consts, self._env.mask, only])
            wp.launch(kern.task_kernel, dim=(self.num_envs, self.n_robots),
                      inputs=[self._sim, self._dfm, self._box, self._rob,
                              self._adh, self._consts, self._env.mask, only])

    # ------------------------------------------------------------------ #
    # Obs / reward
    # ------------------------------------------------------------------ #
    def _compute_obs(self, only_reset: bool = False, from_step: bool = False) -> torch.Tensor:
        if not from_step:
            self._launch_task(only_reset)
        qpos, qvel = self.qpos_torch(), self.qvel_torch()
        blocks = []
        for r in range(self.n_robots):
            c, h = self._field_c[r], self._field_h[r]
            m = self._has_tgt[:, r:r + 1]      # target terms are zero without a target
            tgt = (self._tgt_pos[:, r] - c) / h
            blocks += [
                (qpos[:, self._qadr_t[r]] - self._qpos_off[r]) * self._qpos_scl[r],
                qvel[:, self._dadr_t[r]] / self.vel_scale,
                (self._tip_pos[:, r] - c) / h,
                self._tip_quat[:, r],
                self._tip_lvel[:, r] / self.vel_scale,
                self._tip_avel[:, r] / self.vel_scale,
                tgt * m,
                self._tgt_quat[:, r] * m,
                self._tgt_lvel[:, r] / self.vel_scale * m,
                self._tgt_avel[:, r] / self.vel_scale * m,
                (self._tgt_pos[:, r] - self._tip_pos[:, r]) / self.max_rel_dist * m,
                (self._tgt_pos[:, r] - self._goal_pos[:, r]) / self.max_rel_dist * m,
                (self._tip_pos[:, r] - self._goal_pos[:, r]) / self.max_rel_dist,
                self._grasped[:, r:r + 1],
                m,
            ]
        if self.time_in_obs:
            blocks.append(self._time_left())
        obs = torch.cat(blocks, dim=1)
        if self.check_finite:
            self._assert_finite(obs, "obs")
        # A diverged world never meets a terminal condition (comparisons with NaN
        # are False), so flag it for _reward_terminated_truncated to terminate,
        # and return a zero placeholder row
        self._diverged = ~torch.isfinite(obs).all(dim=1)
        self._diverged_total += (self._diverged & self.reset_mask if only_reset else self._diverged).sum()
        return obs.masked_fill_(self._diverged[:, None], 0.0)

    def _time_left(self) -> torch.Tensor:
        """Share of the episode still ahead, 1 after a reset and 0 on the step the
        time limit is reached."""
        spent = self.world_steps.float() / float(
            self._episode_ctrl_steps * self.phys_substeps)
        return (1.0 - spent).clamp_(0.0, 1.0).unsqueeze(1)

    def sample_observations(self, n: int = 1) -> torch.Tensor:
        """``n`` episode-start observations, from resets of the whole batch. Use a
        separate env."""
        rows, collected = [], 0
        while collected < int(n):
            out = self.reset()                  # the co-design reset returns (obs, rejected, info)
            rows.append((out[0] if isinstance(out, tuple) else out).clone())
            collected += rows[-1].shape[0]
        return torch.cat(rows)[:int(n)]

    def _rebase_episode(self, mask: torch.Tensor) -> None:
        """Start each robot's shaping and the reward given per box of the worlds in ``mask`` from their respawn pose, so
        the first step rewards only its own progress. Reads the distances the respawn's ``_compute_obs`` wrote."""
        m = mask[:, None]
        self._prev_d_tip = torch.where(m, self._d_tip, self._prev_d_tip)
        self._best_d_goal = torch.where(m, self._d_goal, self._best_d_goal)
        self._low_grasp = torch.where(m, 1.0, self._low_grasp)
        self._d_tip0 = torch.where(m, self._d_tip.clamp(min=self.goal_threshold), self._d_tip0)
        self._d_goal0 = torch.where(m, self._d_goal.clamp(min=self.goal_threshold), self._d_goal0)
        self._paid.masked_fill_(mask[:, None, None], 0.0)

    def _reward_terminated_truncated(self):
        # Every geometric quantity was written by task_kernel during _compute_obs
        d_tip, d_goal = self._d_tip, self._d_goal
        grasp = self._act_buf[:, self._grasp_idx].clamp(0.0, 1.0)   # grasp level past the dead zone
        grasped = self._grasped > 0.5
        close = self._close > 0.5
        away = self._away > 0.5
        has_tgt = self._has_tgt > 0.5
        box_return = self._box_return

        # A robot with a new target re-bases its shaping from here. The grasp
        # record starts at the top, so the first release gives the full share
        fresh = self._rebase_t > 0
        self._prev_d_tip = torch.where(fresh, d_tip, self._prev_d_tip)
        self._best_d_goal = torch.where(fresh, d_goal, self._best_d_goal)
        self._low_grasp = torch.where(fresh, 1.0, self._low_grasp)
        self._d_tip0 = torch.where(fresh, d_tip.clamp(min=self.goal_threshold), self._d_tip0)
        self._d_goal0 = torch.where(fresh, d_goal.clamp(min=self.goal_threshold), self._d_goal0)

        # Each phase gives its share of the box's return over a full move, relative
        # to the distance when the target was set. Reach is potential-based. Carry
        # rewards only new bests, so carry-release-recarry loops give nothing.
        # Release rewards lowering the grasp command, at most once in total
        r_reach = (self._prev_d_tip - d_tip) / self._d_tip0 * (self.reach_weight * box_return)
        r_carry = (self._best_d_goal - d_goal).clamp(min=0.0) / self._d_goal0 * (self.carry_weight * box_return)
        pays_drop = close & (grasped | ~away)
        drop = torch.where(pays_drop, (self._low_grasp - grasp).clamp(min=0.0), 0.0)
        self._low_grasp = self._low_grasp - drop
        r_drop = drop * (self.ungrasp_weight * box_return)
        r_goal = self._goal_rew.expand_as(d_tip)

        # At the goal, the robot is releasing until it lets go and clears the box,
        # and only then does the goal term give
        per_robot = torch.where(close,
                                torch.where(pays_drop, r_drop, r_goal),
                                torch.where(grasped, r_carry, r_reach))
        paid_now = per_robot * has_tgt.float()
        # Book the reward given to each robot's target box. A box that left the region it
        # was delivered to takes back everything given for it, so knocking it
        # out and re-delivering it nets nothing
        tgt = self._tgt.long()
        self._paid.masked_fill_(self.reset_mask[:, None, None], 0.0)
        self._paid.scatter_add_(2, tgt.clamp(min=0).unsqueeze(2), (paid_now * (tgt >= 0).float()).unsqueeze(2))
        owner = torch.arange(self.n_robots, device=tgt.device)[None, :, None] == self._box_owner[None, None, :]
        undone = self._left_region.bool()[:, None, :] & owner
        clawback = (self._paid * undone.float()).sum(dim=2)
        self._paid.masked_fill_(undone, 0.0)
        self.last_reward_per_robot = paid_now - clawback   # for callers scoring robots apart
        reward = self.last_reward_per_robot.sum(dim=1)
        self._prev_d_tip = d_tip.clone()
        self._best_d_goal = torch.minimum(self._best_d_goal, d_goal)

        scored = self._scored > 0
        respawned = self.reset_mask
        zero = torch.zeros_like(self._idle_loader)
        self._delivered_total = torch.where(
            respawned[:, None], torch.zeros_like(self._delivered_total),
            self._delivered_total + scored.long() - self._undeliv.long())
        self._update_success()
        # Idle clocks: no box loaded, or none stacked by any stacker. One stacker
        # waiting for its colour is not idleness
        self._idle_loader = torch.where(respawned | scored[:, 0], zero,
                                        self._idle_loader + 1)
        stacked = scored[:, 1:].any(dim=1)
        self._idle_stack = torch.where(respawned | stacked, zero,
                                       self._idle_stack + 1)
        idle = ((self._idle_loader >= self._idle_limit)
                | (self._idle_stack >= self._idle_limit))
        dropped = self._lost.sum(dim=1) > 0          # one lost box ends the episode
        left_field = ((self._tip_pos - self._field_c).abs()
                      > self._field_h).any(dim=2).any(dim=1)

        # Idleness, a lost box, a tip leaving its field, and divergence are true
        # terminals. The time limit is reported per truncate_at_time_limit, and
        # never overrides another ending
        terminated = idle | dropped | self._diverged
        if self.terminate_out_of_field:
            terminated = terminated | left_field
        ran_out = (self.world_steps >= self.max_episode_steps) & ~terminated
        if self.truncate_at_time_limit:
            truncated = ran_out
        else:
            terminated = terminated | ran_out
            truncated = torch.zeros_like(terminated)
        if self.check_finite:
            self._assert_finite(reward, "reward")
        reward = reward.masked_fill(self._diverged, 0.0)
        # Still sums to the reward
        self.last_reward_per_robot = self.last_reward_per_robot.masked_fill(self._diverged[:, None], 0.0)

        self.last_scored = scored
        self.last_idle = idle
        self.last_dropped = dropped
        self.last_left_field = left_field
        self.last_time_limit = ran_out          # time ran out, however it is reported
        self.last_diverged = self._diverged
        return reward, terminated, truncated

    @property
    def boxes_delivered(self) -> torch.Tensor:
        """Deliveries this episode per world and robot (loader first), net of
        boxes that left their region."""
        return self._delivered_total

    def _update_success(self) -> None:
        # A world succeeds once every robot has made all its deliveries
        self.last_success.copy_(
            (self._delivered_total >= self._success_deliveries).all(dim=1))

    @property
    def diverged_worlds(self) -> int:
        """Worlds ended so far by non-finite physics, should stay 0. Syncs with the device."""
        return int(self._diverged_total)

    # ------------------------------------------------------------------ #
    # Live tunables + host-side views
    # ------------------------------------------------------------------ #
    def change_max_offset(self, new_offset: float) -> None:
        """Set the metres per unit embodiment action, from the next respawn of each world."""
        self.max_offset = float(new_offset)
        self._tighten_length_floor()
        self._warn_length_floor()
        self._refresh_tunables()

    def _tighten_length_floor(self) -> None:
        """Clip every embodiment entry so no link gets shorter than
        ``min_link_length + min_length_pad`` at the current ``max_offset``, on the
        host table and on the device once uploaded."""
        lo = length_safe_lo(self._segs, self._suffix_lo, self.max_offset, self.min_link_length + self.min_length_pad)
        self._segs["axis_lo"] = lo
        if getattr(self, "_seg", None) is not None:
            self._seg.axis_lo.assign(lo.astype(np.float32))

    def _warn_length_floor(self) -> None:
        """Warn about links ``max_offset`` can drive into their length floor, once
        per process, since the floor can be intended."""
        hit = floor_reports(self.m, self._segs, self.max_offset)
        if hit and not TruckUnloadParallelEnv._length_floor_warned:
            TruckUnloadParallelEnv._length_floor_warned = True
            warnings.warn(
                f"max_offset={self.max_offset} can shorten "
                f"{len(hit)} link(s) past min_link_length="
                f"{self.min_link_length}, where the deform clamps them:\n  "
                + "\n  ".join(hit), RuntimeWarning, stacklevel=3)

    def set_belt_velocity(self, vel: float) -> None:
        """Set the belt speed every world uses now, and the centre of the speeds later respawns draw."""
        self.belt_velocity = float(vel)
        self._refresh_tunables()
        sv = wp.to_torch(self.gpu_model.geom_surfacevel)
        sv[:, torch.as_tensor(self._belt_gids, dtype=torch.long), 0] = float(vel)

    @property
    def env_offsets(self) -> np.ndarray:
        """Host copy of each world's applied embodiment in metres, at the max_offset it was built with."""
        return (self._offsets_t * self._offset_scale[:, None]).detach().cpu().numpy()

    @property
    def env_offsets_norm(self) -> np.ndarray:
        """Host copy of each world's applied embodiment, normalised like the embodiment action."""
        return self._offsets_t.detach().cpu().numpy()

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def _init_render_model(self, m: mujoco.MjModel) -> None:
        # Without this, CPU kinematics draws turned links unturned
        release_geom_frames(m, self._segs)

    def _mirror_render_model(self, m: mujoco.MjModel, env_idx: int) -> None:
        """Copy world ``env_idx``'s deformed links and goal sites into the render copy ``m``."""
        w, v, gm = int(env_idx), m, self.gpu_model
        g, b, kid = self._segs["geom"], self._segs["body"], self._segs["child_id"]
        v.geom_pos[g] = wp.to_torch(gm.geom_pos)[w, g].cpu().numpy()
        v.geom_quat[g] = wp.to_torch(gm.geom_quat)[w, g].cpu().numpy()
        v.body_iquat[b] = wp.to_torch(gm.body_iquat)[w, b].cpu().numpy()
        v.body_ipos[b] = wp.to_torch(gm.body_ipos)[w, b].cpu().numpy()
        v.geom_size[g] = wp.to_torch(gm.geom_size)[w, g].cpu().numpy()
        v.geom_rbound[g] = wp.to_torch(gm.geom_rbound)[w, g].cpu().numpy()
        v.geom_aabb[g] = wp.to_torch(gm.geom_aabb)[w, g].reshape(len(g), 6).cpu().numpy()
        if len(kid):
            v.body_pos[kid] = wp.to_torch(gm.body_pos)[w, kid].cpu().numpy()
        v.site_pos[:] = wp.to_torch(gm.site_pos)[w].cpu().numpy()
        # The visualizer culls against bvh_aabb, fixed at compile time. For a
        # one-geom body, the body's box is that geom's aabb
        for bid, gid in zip(b, g):
            adr = int(v.body_bvhadr[bid])
            if adr >= 0 and v.body_bvhnum[bid] == 1 and v.body_geomnum[bid] == 1:
                v.bvh_aabb[adr] = v.geom_aabb[gid]

    def _add_debug_visualizations(self, scene: mujoco.MjvScene, env_idx: int) -> None:
        """Draw each robot's tip -> target -> goal chain."""
        tip = self._tip_pos[env_idx].detach().cpu().numpy().astype(np.float64)
        tgt = self._tgt_pos[env_idx].detach().cpu().numpy().astype(np.float64)
        goal = self._goal_pos[env_idx].detach().cpu().numpy().astype(np.float64)
        has = self._has_tgt[env_idx].detach().cpu().numpy()
        for r in range(self.n_robots):
            if has[r] < 0.5:
                continue
            for a, z, rgba in ((tip[r], tgt[r], [0.2, 0.9, 0.3, 0.6]),
                               (tgt[r], goal[r], [0.9, 0.6, 0.2, 0.6])):
                if scene.ngeom >= scene.maxgeom:
                    return
                gm = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    gm, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                    np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
                mujoco.mjv_connector(gm, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.005, a, z)
                scene.ngeom += 1

    def _setup_camera(self, camera: mujoco.MjvCamera) -> None:
        """``fixed`` looks down the whole line, ``track`` follows the loader's tip."""
        lay = self.layout
        x0, x1 = float(lay.field_lo[0][0]), float(lay.conveyor.upper[0])
        y1 = float(lay.field_hi[-1][1])
        if self.camera_mode == "track":
            camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            camera.trackbodyid = self._tip_bid[0]
            camera.azimuth, camera.elevation = 20.0, -25.0
            camera.distance = 2.0 * max(y1, 2.0)
            return
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [(x0 + x1) / 2.0, y1 / 2.0, lay.z_playing_field]
        camera.elevation = -25.0
        camera.azimuth = 120.0
        camera.distance = 1.3 * max(x1 - x0, y1, 2.0)
